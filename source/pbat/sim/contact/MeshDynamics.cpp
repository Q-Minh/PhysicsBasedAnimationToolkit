#include "MeshDynamics.h"

#include "Friction.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <exception>
#include <fmt/core.h>
#include <limits>
#include <numeric>
#include <span>
#include <tbb/parallel_for.h>

namespace pbat::sim::contact {

void MeshDynamics::Construct(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    MultiMesh<IndexType> meshes,
    geometry::sdf::Forest<ScalarType> sdfForest,
    ScalarType nReserveRatio)
{
    mMeshes    = std::move(meshes);
    mSdfForest = std::move(sdfForest);
    mSdf       = geometry::sdf::Composite<ScalarType>(
        std::span<geometry::sdf::Node<ScalarType> const>(
            mSdfForest.nodes.data(),
            mSdfForest.nodes.size()),
        std::span<geometry::sdf::Transform<ScalarType> const>(
            mSdfForest.transforms.data(),
            mSdfForest.transforms.size()),
        std::span<std::pair<int, int> const>(
            mSdfForest.children.data(),
            mSdfForest.children.size()),
        std::span<int const>(mSdfForest.roots.data(), mSdfForest.roots.size()));
    if (mSdf.Status() != geometry::sdf::ECompositeStatus::Valid)
    {
        throw std::invalid_argument(
            fmt::format(
                "MeshDynamics::SetGeometry: invalid SDF forest with status {}",
                static_cast<int>(mSdf.Status())));
    }
    // Preallocate contact constraint storage
    CFP.resize(mMeshes.F.cols() + 1);
    CHEP.resize(mMeshes.F.cols() * 3 + 1);
    V2CV.resize(mMeshes.V.size());
    FA.resize(mMeshes.F.cols());
    HEA.resize(mMeshes.F.cols() * 3);
    VA.resize(mMeshes.V.size());
    CF.reserve(static_cast<size_t>(mMeshes.F.cols() * nReserveRatio));
    CHE.reserve(static_cast<size_t>(mMeshes.F.cols() * 3 * nReserveRatio));
    CV.reserve(static_cast<size_t>(mMeshes.V.size() * nReserveRatio));
}

void MeshDynamics::InitializeMeshMeshContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    geometry::Device const& device,
    OgcParams const& ogcParams)
{
    mOffsetGeometryContact.Initialize(
        device,
        X,
        mMeshes.V,
        mMeshes.F,
        mMeshes.E,
        mMeshes.VP,
        mMeshes.FP,
        mMeshes.EP,
        ogcParams);
}

void MeshDynamics::InitializeMeshEnvironmentContactDetection(MeshSdfContactParams const& params)
{
    mMeshSdfContact.Initialize(mMeshes.V, mMeshes.F, params);
    CFP.setZero();
    CHEP.setZero();
    V2CV.setConstant(IndexType(-1));
}

void MeshDynamics::UpdateEnvironmentContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    // 1. Run mesh-SDF contact detection
    mMeshSdfContact.PrepareIteration();
    mMeshSdfContact.TriangleSdfContactDetection(X, mMeshes.F, mSdf);
    mMeshSdfContact.DeduplicateContactSet(mMeshes.F, mMeshes.GHEF, mMeshes.GXV);
    // 2. Recompute geometric quantities
    auto Xa = X(Eigen::placeholders::all, mMeshes.F.row(0)).bottomRows<3>();
    auto Xb = X(Eigen::placeholders::all, mMeshes.F.row(1)).bottomRows<3>();
    auto Xc = X(Eigen::placeholders::all, mMeshes.F.row(2)).bottomRows<3>();
    for (auto f = 0; f < FA.size(); ++f)
        FA(f) = ScalarType(0.5) * (Xb.col(f) - Xa.col(f)).cross((Xc.col(f) - Xa.col(f))).norm();
    for (auto he = 0; he < HEA.size(); ++he)
        HEA(he) = ScalarType(0.5) * (FA(mMeshes.GHEF(0, he)) + FA(mMeshes.GHEF(1, he)));
    for (auto v = 0; v < VA.size(); ++v)
    {
        VA(v)      = ScalarType(0);
        auto begin = mMeshes.GVHEp(v);
        auto end   = mMeshes.GVHEp(v + 1);
        for (IndexType he : mMeshes.GVHEadj(Eigen::seqN(begin, end - begin)))
            VA(v) += FA(geometry::FaceOfHalfEdge(he));
    }
    // 3. Update constraint lists
    UpdateTriangleContactConstraints(X);
    UpdateHalfEdgeContactConstraints(X);
    UpdateVertexContactConstraints(X);
}

void MeshDynamics::PrepareEnvironmentContactsForDualIteration()
{
    tbb::parallel_for(std::size_t{0}, CF.size(), [&](std::size_t c) {
        CF[c].lambda *= mEnvContactDynamicsParams.gamma;
        CF[c].k =
            (mEnvContactDynamicsParams.gamma * CF[c].k).cwiseMax(mEnvContactDynamicsParams.kstart);
    });
    tbb::parallel_for(std::size_t{0}, CHE.size(), [&](std::size_t c) {
        CHE[c].lambda *= mEnvContactDynamicsParams.gamma;
        CHE[c].k =
            (mEnvContactDynamicsParams.gamma * CHE[c].k).cwiseMax(mEnvContactDynamicsParams.kstart);
    });
    tbb::parallel_for(std::size_t{0}, CV.size(), [&](std::size_t c) {
        CV[c].lambda *= mEnvContactDynamicsParams.gamma;
        CV[c].k =
            (mEnvContactDynamicsParams.gamma * CV[c].k).cwiseMax(mEnvContactDynamicsParams.kstart);
    });
}

void MeshDynamics::DualUpdateEnvironmentContacts(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    auto const fDualUpdate = [&](Eigen::Vector<ScalarType, 3> const& xc,
                                 EnvironmentContactConstraint& c) {
        // Update Lagrange multipliers
        bool const bViolating = (c.C(0) < ScalarType(0));
        c.lambda(0)           = std::clamp(
            c.lambda(0) - bViolating * (c.k(0) * c.C(0)),
            ScalarType(0),
            mEnvContactDynamicsParams.Fnmax);
        ScalarType const muFn = mEnvContactDynamicsParams.mu * c.lambda(0);
        c.lambda(1)           = std::clamp(c.lambda(1) - c.k(1) * c.C(1), -muFn, muFn);
        c.lambda(2)           = std::clamp(c.lambda(2) - c.k(2) * c.C(2), -muFn, muFn);
        // Update stiffnesses
        c.k.array() += mEnvContactDynamicsParams.beta * c.k.array() * c.C.array().abs();
    };
    tbb::parallel_for(Eigen::Index{0}, mMeshes.F.cols(), [&](Eigen::Index f) {
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, mMeshes.F.col(f));
        std::vector<Eigen::Vector<ScalarType, 2>> const& contactPoints =
            mMeshSdfContact.mTriangleContactPoints[f];
        auto begin = CFP[f];
        auto n     = CFP[f + 1] - begin;
        for (IndexType k = 0; k < n; ++k)
        {
            IndexType const cidx                   = begin + k;
            Eigen::Vector<ScalarType, 2> const& uv = contactPoints[k];
            Eigen::Vector<ScalarType, 3> const xc =
                (ScalarType(1) - uv(0) - uv(1)) * xf.col(0) + uv(0) * xf.col(1) + uv(1) * xf.col(2);
            fDualUpdate(xc, CF[cidx]);
        }
    });
    tbb::parallel_for(Eigen::Index{0}, 3 * mMeshes.F.cols(), [&](Eigen::Index he) {
        Eigen::Vector<ScalarType, 3> const xi = X.col(geometry::IncomingVertex(mMeshes.F, he));
        Eigen::Vector<ScalarType, 3> const xj = X.col(geometry::OutgoingVertex(mMeshes.F, he));
        std::vector<ScalarType> const& contactPoints = mMeshSdfContact.mHalfEdgeContactPoints[he];
        auto begin                                   = CHEP[he];
        auto n                                       = CHEP[he + 1] - begin;
        for (IndexType k = 0; k < n; ++k)
        {
            IndexType const cidx                  = begin + k;
            ScalarType const u                    = contactPoints[k];
            Eigen::Vector<ScalarType, 3> const xc = (ScalarType(1) - u) * xi + u * xj;
            fDualUpdate(xc, CHE[cidx]);
        }
    });
    tbb::parallel_for(std::size_t(0), CV.size(), [&](std::size_t v) {
        fDualUpdate(X.col(mMeshes.V(v)), CV[v]);
    });
}

namespace detail {

template <class TScalar>
void ComputeEnvironmentContactBasis(
    Eigen::Vector<TScalar, 3> const& xc,
    geometry::sdf::Composite<TScalar> const& mSdf,
    TScalar hfd,
    Eigen::Vector<TScalar, 3>& O,
    Eigen::Matrix<TScalar, 3, 3>& B)
{
    namespace mini = math::linalg::mini;
    // A true SDF has |grad SDF| = 1, but in practice, it is not always a valid assumption.
    // A distance estimate can be given by sd / |grad SDF|.
    // The contact basis' origin should be on the surface of the SDF, i.e.
    // O = xc - (sd / |grad SDF|) * (grad SDF / |grad SDF|) = xc - sd * (grad SDF / |grad
    // SDF|^2)
    Eigen::Vector<TScalar, 3> n = mini::ToEigen(mSdf.Grad(mini::FromEigen(xc), hfd));
    TScalar normGrad            = n.norm();
    n /= normGrad;
    TScalar sd = mSdf.Eval(mini::FromEigen(xc)) / normGrad;
    O          = xc - sd * n;
    // Contact basis
    B.col(0)         = n;
    B.rightCols<2>() = mini::ToEigen(TangentialBasis(mini::FromEigen(n)));
}

template <class TScalar>
MeshDynamics::EnvironmentContactConstraint CreateNewEnvironmentContactConstraint(
    Eigen::Vector<TScalar, 3> const& xc,
    geometry::sdf::Composite<TScalar> const& mSdf,
    TScalar hfd,
    MeshDynamics::ScalarType area,
    MeshDynamics::ScalarType kstart,
    MeshDynamics::ScalarType mu)
{
    namespace mini = math::linalg::mini;
    MeshDynamics::EnvironmentContactConstraint c;
    // Contact basis and origin
    ComputeEnvironmentContactBasis(xc, mSdf, hfd, c.O, c.B);
    // Initialize Lagrange multipliers and stiffness as if no warm-starting
    c.lambda.setZero();
    c.k.setConstant(kstart * area);
    // Friction coefficient
    c.mu = mu;
    return c;
}

template <class TScalar>
MeshDynamics::EnvironmentContactConstraint CreateNewEnvironmentContactConstraint(
    Eigen::Vector<TScalar, 3> const& xc,
    geometry::sdf::Composite<TScalar> const& mSdf,
    TScalar hfd,
    MeshDynamics::ScalarType area,
    MeshDynamics::ScalarType kstart,
    MeshDynamics::ScalarType mu,
    Eigen::Vector<MeshDynamics::ScalarType, 3>& totalConstraintError)
{
    namespace mini = math::linalg::mini;
    MeshDynamics::EnvironmentContactConstraint c =
        CreateNewEnvironmentContactConstraint(xc, mSdf, hfd, area, kstart, mu);
    c.C = c.B.transpose() * (xc - c.O);
    totalConstraintError += c.C.cwiseAbs();
    return c;
}

template <class TScalar>
void ConstraintErrorProportionalWarmStart(
    Eigen::Vector<TScalar, 3> const& totalLambda,
    Eigen::Vector<TScalar, 3> const& totalStiffness,
    Eigen::Vector<TScalar, 3> const& totalConstraintError,
    TScalar Fnmax,
    TScalar mu,
    MeshDynamics::EnvironmentContactConstraint& c)
{
    // Signed constraint proportional weights
    Eigen::Vector<TScalar, 3> const wc = c.C.array() / totalConstraintError.array();
    c.lambda(0)                        = std::min(wc(0) * totalLambda(0), Fnmax);
    // Coulomb friction bounds
    TScalar const muFn = mu * c.lambda(0);
    c.lambda(1)        = std::clamp(wc(1) * totalLambda(1), -muFn, muFn);
    c.lambda(2)        = std::clamp(wc(2) * totalLambda(2), -muFn, muFn);
    // Stiffness should be non-negative
    c.k = wc.array().abs() * totalStiffness.array();
}

} // namespace detail

void MeshDynamics::EnvironmentContactConstraint::Eval(Eigen::Vector<ScalarType, 3> const& xc)
{
    C = B.transpose() * (xc - O);
}

Eigen::Vector<MeshDynamics::ScalarType, 3>
MeshDynamics::EnvironmentContactConstraint::ForceEstimate(ScalarType lambdaNmax) const
{
    ScalarType const muFn = mu * lambda(0);
    return Eigen::Vector<MeshDynamics::ScalarType, 3>{
        std::min(lambda(0) - k(0) * C(0), lambdaNmax),
        std::clamp(lambda(1) - k(1) * C(1), -muFn, muFn),
        std::clamp(lambda(2) - k(2) * C(2), -muFn, muFn)};
};

void MeshDynamics::UpdateTriangleContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    IndexType const nPastTriangleConstraints = static_cast<IndexType>(CF.size());
    // 1. Warm-start triangle contact constraints
    IndexType const nTriangles = CFP.size() - 1;
    for (IndexType f = 0; f < nTriangles; ++f)
    {
        // Get contacts on f
        std::span<EnvironmentContactConstraint const> pastConstraints(
            CF.data() + CFP[f],
            CFP[f + 1] - CFP[f]);
        // Capture aggregate unsigned contact forces
        Eigen::Vector<ScalarType, 3> totalLambda    = Eigen::Vector<ScalarType, 3>::Zero();
        Eigen::Vector<ScalarType, 3> totalStiffness = Eigen::Vector<ScalarType, 3>::Zero();
        for (EnvironmentContactConstraint const& c : pastConstraints)
        {
            totalLambda(0) += c.lambda(0);           // normal contact force should be non-negative
            totalLambda(1) += std::abs(c.lambda(1)); // frictional forces have no sign requirement
            totalLambda(2) += std::abs(c.lambda(2)); // frictional forces have no sign requirement
            totalStiffness += c.k;
        }
        // Create new contact constraints for this triangle at end of current constraint list, we
        // will swap and erase later to keep only the new constraints.
        std::size_t const iNewConstraintStart             = CF.size();
        ScalarType const farea                            = FA(f);
        Eigen::Vector<ScalarType, 3> totalConstraintError = Eigen::Vector<ScalarType, 3>::Zero();
        for (Eigen::Vector<ScalarType, 2> const& uv : mMeshSdfContact.mTriangleContactPoints[f])
        {
            // Fetch contact point
            Eigen::Matrix<ScalarType, 3, 3> xf = X(Eigen::placeholders::all, mMeshes.F.col(f));
            Eigen::Vector<ScalarType, 3> xc =
                (1 - uv(0) - uv(1)) * xf.col(0) + uv(0) * xf.col(1) + uv(1) * xf.col(2);
            // Add new constraint
            CF.push_back(
                detail::CreateNewEnvironmentContactConstraint(
                    xc,
                    mSdf,
                    mMeshSdfContact.mParams.hfd,
                    farea,
                    mEnvContactDynamicsParams.kstart,
                    mEnvContactDynamicsParams.mu,
                    totalConstraintError));
        }
        bool const bWarmStart = not pastConstraints.empty();
        if (bWarmStart)
        {
            // Distribute past contact forces and stiffness to new constraints proportionally to
            // their constraint errors
            for (std::size_t ci = iNewConstraintStart; ci < CF.size(); ++ci)
            {
                EnvironmentContactConstraint& c = CF[ci];
                detail::ConstraintErrorProportionalWarmStart(
                    totalLambda,
                    totalStiffness,
                    totalConstraintError,
                    mEnvContactDynamicsParams.Fnmax,
                    mEnvContactDynamicsParams.mu,
                    c);
            }
        }
    }
    // 2. Swap new constraints to front
    CF.erase(CF.begin(),
             CF.begin() + nPastTriangleConstraints); // Remove old constraints

    // 3. Update prefix sums
    CFP[0] = 0;
    std::transform(
        mMeshSdfContact.mTriangleContactPoints.begin(),
        mMeshSdfContact.mTriangleContactPoints.end(),
        CFP.begin() + 1,
        [](std::vector<Eigen::Vector<ScalarType, 2>> const& tcp) {
            return static_cast<IndexType>(tcp.size());
        });
    std::inclusive_scan(CFP.begin() + 1, CFP.end(), CFP.begin() + 1);
}

void MeshDynamics::UpdateHalfEdgeContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    IndexType const nPastHalfEdgeConstraints = static_cast<IndexType>(CHE.size());
    // 1. Warm-start half-edge contact constraints
    IndexType const nHalfEdges = CHEP.size() - 1;
    for (IndexType he = 0; he < nHalfEdges; ++he)
    {
        // Get contacts on f
        std::span<EnvironmentContactConstraint const> pastConstraints(
            CHE.data() + CHEP[he],
            CHEP[he + 1] - CHEP[he]);
        // Capture aggregate unsigned contact forces
        Eigen::Vector<ScalarType, 3> totalLambda    = Eigen::Vector<ScalarType, 3>::Zero();
        Eigen::Vector<ScalarType, 3> totalStiffness = Eigen::Vector<ScalarType, 3>::Zero();
        for (EnvironmentContactConstraint const& c : pastConstraints)
        {
            totalLambda(0) += c.lambda(0);           // normal contact force should be non-negative
            totalLambda(1) += std::abs(c.lambda(1)); // frictional forces have no sign requirement
            totalLambda(2) += std::abs(c.lambda(2)); // frictional forces have no sign requirement
            totalStiffness += c.k;
        }
        // Create new contact constraints for this triangle at end of current constraint list, we
        // will swap and erase later to keep only the new constraints.
        std::size_t const iNewConstraintStart             = CF.size();
        ScalarType const hearea                           = HEA(he);
        Eigen::Vector<ScalarType, 3> totalConstraintError = Eigen::Vector<ScalarType, 3>::Zero();
        for (ScalarType u : mMeshSdfContact.mHalfEdgeContactPoints[he])
        {
            // Fetch contact point
            IndexType const i               = geometry::IncomingVertex(mMeshes.F, he);
            IndexType const j               = geometry::OutgoingVertex(mMeshes.F, he);
            Eigen::Vector<ScalarType, 3> xc = (1 - u) * X.col(i) + u * X.col(j);
            // Add new constraint
            CF.push_back(
                detail::CreateNewEnvironmentContactConstraint(
                    xc,
                    mSdf,
                    mMeshSdfContact.mParams.hfd,
                    hearea,
                    mEnvContactDynamicsParams.kstart,
                    mEnvContactDynamicsParams.mu,
                    totalConstraintError));
        }
        bool const bWarmStart = not pastConstraints.empty();
        if (bWarmStart)
        {
            // Distribute past contact forces and stiffness to new constraints proportionally to
            // their constraint errors
            for (std::size_t ci = iNewConstraintStart; ci < CHE.size(); ++ci)
            {
                EnvironmentContactConstraint& c = CHE[ci];
                detail::ConstraintErrorProportionalWarmStart(
                    totalLambda,
                    totalStiffness,
                    totalConstraintError,
                    mEnvContactDynamicsParams.Fnmax,
                    mEnvContactDynamicsParams.mu,
                    c);
            }
        }
    }
    // 2. Swap new constraints to front
    CHE.erase(CHE.begin(),
              CHE.begin() + nPastHalfEdgeConstraints); // Remove old constraints
    // 3. Update prefix sums
    CHEP[0] = 0;
    std::transform(
        mMeshSdfContact.mHalfEdgeContactPoints.begin(),
        mMeshSdfContact.mHalfEdgeContactPoints.end(),
        CHEP.begin() + 1,
        [](std::vector<ScalarType> const& hecp) { return static_cast<IndexType>(hecp.size()); });
    std::inclusive_scan(CHEP.begin() + 1, CHEP.end(), CHEP.begin() + 1);
}

void MeshDynamics::UpdateVertexContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    IndexType const nPastVertexConstraints = static_cast<IndexType>(CV.size());
    // 1. Warm-start vertex contact constraints
    IndexType const nVertices = mMeshes.V.size();
    for (IndexType v = 0, n = 0; v < nVertices; ++v)
    {
        IndexType const ci             = V2CV[v];
        bool const bWasVertexInContact = ci >= 0;
        bool const bIsVertexInContact  = mMeshSdfContact.mVertexContactPoints[v];
        if (bIsVertexInContact)
        {
            Eigen::Vector<ScalarType, 3> const xc = X.col(mMeshes.V(v));
            ScalarType const varea                = VA(v);
            bool const bWarmStart                 = bWasVertexInContact;
            if (bWarmStart)
            {
                EnvironmentContactConstraint& c = CV[ci];
                detail::ComputeEnvironmentContactBasis(
                    xc,
                    mSdf,
                    mMeshSdfContact.mParams.hfd,
                    c.O,
                    c.B);
                CV.push_back(c);
            }
            else
            {
                CV.push_back(
                    detail::CreateNewEnvironmentContactConstraint(
                        xc,
                        mSdf,
                        mMeshSdfContact.mParams.hfd,
                        varea,
                        mEnvContactDynamicsParams.kstart,
                        mEnvContactDynamicsParams.mu));
            }
        }
        // Update map
        V2CV[v] = (not bIsVertexInContact) * -1 + (bIsVertexInContact)*n;
        n += bIsVertexInContact;
    }
    // 2. Swap new constraints to front
    CV.erase(CV.begin(),
             CV.begin() + nPastVertexConstraints); // Remove old constraints
}

} // namespace pbat::sim::contact

#include "pbat/geometry/model/Cube.h"
#include "pbat/graph/Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshDynamics initialization")
{
    using namespace pbat;
    using namespace pbat::geometry::model;

    // 1. Arrange

    // Build three disjoint tetrahedral cubes stacked along +Z to create a MultiMesh
    auto [X1, T1] = Cube();
    auto [X2, T2] = Cube();
    auto [X3, T3] = Cube();
    X2.row(2).array() += Scalar(2);
    X3.row(2).array() += Scalar(4);

    // Concatenate into a single mesh (adjust indices)
    Index n1 = static_cast<Index>(X1.cols());
    Index n2 = static_cast<Index>(X2.cols());
    Index n3 = static_cast<Index>(X3.cols());
    MatrixX X(3, n1 + n2 + n3);
    X << X1, X2, X3;
    IndexMatrixX T(4, T1.cols() + T2.cols() + T3.cols());
    T << T1, (T2.array() + n1), (T3.array() + n1 + n2);

    // Compute connected components and reindex
    IndexVectorX XCC(X.cols()), ECC(T.cols()), Xord(X.cols()), Eord(T.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, T, XCC, ECC, Xord, Eord);
    graph::ReindexMeshByConnectedComponents(X, T, XCC, ECC, Xord, Eord);

    // Create MultiMesh via BoundaryTriangulation
    sim::contact::MultiMesh<Index> meshes(T.bottomRows<4>(), XCC, nComponents);
    // Create an SDF Forest with some primitives
    geometry::sdf::Forest<Scalar> sdfForest;
    // Add some primitive nodes
    sdfForest.nodes.push_back(geometry::sdf::Sphere<Scalar>{1.5}); // radius = 1.5
    // Add transforms for each node
    sdfForest.transforms.push_back(geometry::sdf::Transform<Scalar>::Identity());
    sdfForest.transforms.back().t(2) -= Scalar(1.2);
    // Set up tree structure (all roots, no hierarchy for simplicity)
    sdfForest.roots.push_back(0);
    sdfForest.children.push_back({-1, -1}); // Leaf node

    // 2. Act

    // Construct MeshDynamics
    sim::contact::MeshDynamics meshDynamics;
    Scalar const nReserveRatio = 2.0;
    meshDynamics.Construct(X, std::move(meshes), std::move(sdfForest), nReserveRatio);

    // 3. Assert

    // Check that basic structures are initialized
    CHECK_GT(meshDynamics.mMeshes.V.size(), 0);
    // Check that SDF forest is set
    CHECK_EQ(meshDynamics.mSdfForest.nodes.size(), 1);
    CHECK_EQ(meshDynamics.mSdfForest.transforms.size(), 1);
    CHECK_EQ(meshDynamics.mSdfForest.roots.size(), 1);
    CHECK_EQ(meshDynamics.mSdfForest.children.size(), 1);

    // Check that SDF composite is valid
    CHECK(meshDynamics.mSdf.Status() == geometry::sdf::ECompositeStatus::Valid);

    // Check that geometric arrays are sized correctly
    CHECK_EQ(meshDynamics.FA.size(), meshDynamics.mMeshes.F.cols());
    CHECK_EQ(meshDynamics.HEA.size(), 3 * meshDynamics.mMeshes.F.cols());
    CHECK_EQ(meshDynamics.VA.size(), meshDynamics.mMeshes.V.size());

    // Check that prefix arrays are sized correctly
    CHECK_EQ(meshDynamics.CFP.size(), meshDynamics.mMeshes.F.cols() + 1);
    CHECK_EQ(meshDynamics.CHEP.size(), 3 * meshDynamics.mMeshes.F.cols() + 1);
    CHECK_EQ(meshDynamics.V2CV.size(), meshDynamics.mMeshes.V.size());

    // Check that contact constraint vectors are initially empty
    CHECK_EQ(meshDynamics.CF.size(), 0);
    CHECK_EQ(meshDynamics.CHE.size(), 0);
    CHECK_EQ(meshDynamics.CV.size(), 0);

    SUBCASE("Mesh-SDF environment contacts")
    {
        // Act
        meshDynamics.InitializeMeshEnvironmentContactDetection(
            sim::contact::MeshSdfContactParams{});
        meshDynamics.UpdateEnvironmentContactConstraints(X);
        meshDynamics.PrepareEnvironmentContactsForDualIteration();

        // Assert
        CHECK_EQ(meshDynamics.CV.size(), 1); // One vertex should be in contact
        CHECK_EQ(meshDynamics.CV.front().mu, meshDynamics.mEnvContactDynamicsParams.mu);
        Eigen::Vector<Scalar, 3> A = X.col(meshDynamics.mMeshes.F(0, 0));
        Eigen::Vector<Scalar, 3> B = X.col(meshDynamics.mMeshes.F(1, 0));
        Eigen::Vector<Scalar, 3> C = X.col(meshDynamics.mMeshes.F(2, 0));
        Scalar areaf0              = Scalar(0.5) * (B - A).cross(C - A).norm();
        CHECK_EQ(meshDynamics.FA(0), areaf0);
        Scalar areahe0 = Scalar(0.5) * (meshDynamics.FA(geometry::FaceOfHalfEdge(0)) +
                                        meshDynamics.FA(geometry::FaceOfHalfEdge(1)));
        CHECK_EQ(meshDynamics.HEA(0), areahe0);
    }
}