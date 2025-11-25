#include "MeshDynamics.h"

#include "Friction.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/profiling/Profiling.h"

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
    MultiMesh<IndexType> meshes,
    geometry::sdf::Forest<ScalarType> sdfForest)
{
    SetStaticGeometry(std::move(sdfForest));
    SetDynamicGeometry(std::move(meshes));
}

void MeshDynamics::SetStaticGeometry(geometry::sdf::Forest<ScalarType> sdfForest)
{
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
}

void MeshDynamics::SetDynamicGeometry(MultiMesh<IndexType> meshes)
{
    mMeshes = std::move(meshes);
}

void MeshDynamics::AllocateEnvironmentContactDataStructures()
{
    // Preallocate contact constraint storage
    auto const nTriangles = mMeshes.F.cols();
    auto const nHalfEdges = nTriangles * 3;
    auto const nVertices  = mMeshes.V.size();
    FA.resize(nTriangles);
    HEA.resize(nHalfEdges);
    VA.resize(nVertices);
    CF.resize(static_cast<size_t>(nTriangles));
    CHE.resize(static_cast<size_t>(nHalfEdges));
    CV.resize(static_cast<size_t>(nVertices));
}

void MeshDynamics::InitializeMeshMeshContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    geometry::Device const& device)
{
    mOffsetGeometryContact
        .Initialize(device, X, mMeshes.V, mMeshes.F, mMeshes.E, mMeshes.VP, mMeshes.FP, mMeshes.EP);
}

void MeshDynamics::InitializeMeshEnvironmentContactDetection()
{
    mMeshSdfContact.Initialize(mMeshes.V, mMeshes.F);
    AllocateEnvironmentContactDataStructures();
    for (auto& c : CF)
    {
        c.C      = ScalarType(0);
        c.lambda = ScalarType(0);
    }
    for (auto& c : CHE)
    {
        c.C      = ScalarType(0);
        c.lambda = ScalarType(0);
    }
    for (auto& c : CV)
    {
        c.C      = ScalarType(0);
        c.lambda = ScalarType(0);
    }
}

void MeshDynamics::UpdateEnvironmentContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateEnvironmentContactConstraints");
    // 1. Run mesh-SDF contact detection
    mMeshSdfContact.PrepareIteration();
    mMeshSdfContact.TriangleSdfContactDetection(
        X,
        mMeshes.V,
        mMeshes.F,
        mMeshes.GHEF,
        mMeshes.EHE,
        mMeshes.GXV,
        mSdf);
    // 2. Recompute geometric quantities
    UpdateGeometricQuantities(X);
    // 3. Update constraint lists
    UpdateTriangleContactConstraints(X);
    UpdateHalfEdgeContactConstraints(X);
    UpdateVertexContactConstraints(X);
}

void MeshDynamics::UpdateGeometricQuantities(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateGeometricQuantities");
    auto Xa = X(Eigen::placeholders::all, mMeshes.F.row(0)).bottomRows<3>();
    auto Xb = X(Eigen::placeholders::all, mMeshes.F.row(1)).bottomRows<3>();
    auto Xc = X(Eigen::placeholders::all, mMeshes.F.row(2)).bottomRows<3>();
    tbb::parallel_for(Eigen::Index{0}, FA.size(), [&](Eigen::Index f) {
        FA(f) = ScalarType(0.5) * (Xb.col(f) - Xa.col(f)).cross((Xc.col(f) - Xa.col(f))).norm();
    });
    tbb::parallel_for(Eigen::Index{0}, HEA.size(), [&](Eigen::Index he) {
        HEA(he) = ScalarType(0.5) * (FA(mMeshes.GHEF(0, he)) + FA(mMeshes.GHEF(1, he)));
    });
    tbb::parallel_for(Eigen::Index{0}, VA.size(), [&](Eigen::Index v) {
        VA(v)      = ScalarType(0);
        auto begin = mMeshes.GVHEp(v);
        auto end   = mMeshes.GVHEp(v + 1);
        for (IndexType he : mMeshes.GVHEadj(Eigen::seqN(begin, end - begin)))
            VA(v) += FA(geometry::FaceOfHalfEdge(he));
    });
}

void MeshDynamics::PrepareEnvironmentContactsForDualIteration()
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshDynamics.PrepareEnvironmentContactsForDualIteration");
    auto const nTriangles = mMeshes.F.cols();
    tbb::parallel_for(Eigen::Index{0}, nTriangles, [&](Eigen::Index f) {
        EnvironmentContact& c = CF[f];
        if (mMeshSdfContact.mTriangleContactMask(f))
        {
            c.lambda *= mEnvContactDynamicsParams.gamma;
            c.k = mEnvContactDynamicsParams.gamma * std::max(c.k, mEnvContactDynamicsParams.kstart);
        }
        else
        {
            c.lambda = ScalarType(0);
            c.k      = mEnvContactDynamicsParams.kstart;
        }
    });
    auto const nEdges = mMeshes.EHE.cols();
    tbb::parallel_for(Eigen::Index{0}, nEdges, [&](Eigen::Index e) {
        IndexType const hei   = mMeshes.EHE(0, e);
        IndexType const hej   = mMeshes.EHE(1, e);
        EnvironmentContact& c = CHE[hei];
        if (mMeshSdfContact.mHalfEdgeContactMask(hei))
        {
            c.lambda *= mEnvContactDynamicsParams.gamma;
            c.k = mEnvContactDynamicsParams.gamma * std::max(c.k, mEnvContactDynamicsParams.kstart);
        }
        else
        {
            c.lambda = ScalarType(0);
            c.k      = mEnvContactDynamicsParams.kstart;
        }
        if (hej >= 0)
            CHE[hej] = CHE[hei];
    });
    auto const nVerts = mMeshes.V.size();
    tbb::parallel_for(Eigen::Index{0}, nVerts, [&](Eigen::Index v) {
        EnvironmentContact& c = CV[v];
        if (mMeshSdfContact.mVertexContactMask(v))
        {
            c.lambda *= mEnvContactDynamicsParams.gamma;
            c.k = mEnvContactDynamicsParams.gamma * std::max(c.k, mEnvContactDynamicsParams.kstart);
        }
        else
        {
            c.lambda = ScalarType(0);
            c.k      = mEnvContactDynamicsParams.kstart;
        }
    });
}

void MeshDynamics::DualUpdateEnvironmentContacts(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.DualUpdateEnvironmentContacts");
    auto const fDualUpdate = [&](Eigen::Vector<ScalarType, 3> const& xc, EnvironmentContact& c) {
        // Update Lagrange multipliers
        c.Eval(xc);
        bool const bViolating = (c.C < ScalarType(0));
        c.lambda              = std::clamp(
            c.lambda - bViolating * (c.k * c.C),
            ScalarType(0),
            mEnvContactDynamicsParams.Fnmax);
        // Update stiffness
        c.k = std::min(
            c.k + bViolating * mEnvContactDynamicsParams.beta * std::abs(c.C),
            mEnvContactDynamicsParams.kmax);
    };
    auto const nTriangles = mMeshes.F.cols();
    tbb::parallel_for(Eigen::Index{0}, nTriangles, [&](Eigen::Index f) {
        if (not mMeshSdfContact.mTriangleContactMask(f))
            return;
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, mMeshes.F.col(f));
        Eigen::Vector<ScalarType, 2> const uv    = mMeshSdfContact.mTriangleContactPoints.col(f);
        Eigen::Vector<ScalarType, 3> const xc =
            (ScalarType(1) - uv(0) - uv(1)) * xf.col(0) + uv(0) * xf.col(1) + uv(1) * xf.col(2);
        fDualUpdate(xc, CF[f]);
    });
    auto const nEdges = mMeshes.EHE.cols();
    tbb::parallel_for(Eigen::Index{0}, nEdges, [&](Eigen::Index e) {
        IndexType const hei = mMeshes.EHE(0, e);
        IndexType const hej = mMeshes.EHE(1, e);
        if (not mMeshSdfContact.mHalfEdgeContactMask(hei))
            return;
        Eigen::Vector<ScalarType, 3> const xi = X.col(geometry::IncomingVertex(mMeshes.F, hei));
        Eigen::Vector<ScalarType, 3> const xj = X.col(geometry::OutgoingVertex(mMeshes.F, hei));
        ScalarType const u                    = mMeshSdfContact.mHalfEdgeContactPoints(hei);
        Eigen::Vector<ScalarType, 3> const xc = (1 - u) * xi + u * xj;
        fDualUpdate(xc, CHE[hei]);
        if (hej >= 0)
            CHE[hej] = CHE[hei];
    });
    auto const nVerts = mMeshes.V.size();
    tbb::parallel_for(Eigen::Index{0}, nVerts, [&](Eigen::Index v) {
        if (not mMeshSdfContact.mVertexContactMask(v))
            return;
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
    B.col(0)                  = n;
    B.template rightCols<2>() = mini::ToEigen(TangentialBasis(mini::FromEigen(n)));
}

} // namespace detail

void MeshDynamics::EnvironmentContact::Eval(Eigen::Vector<ScalarType, 3> const& xc)
{
    auto n = B.col(0);
    C      = n.transpose() * (xc - O);
}

MeshDynamics::ScalarType
MeshDynamics::EnvironmentContact::ForceEstimate(ScalarType lambdaNmax) const
{
    ScalarType const muFn = mu * lambda;
    return std::min(lambda - k * C, lambdaNmax);
};

void MeshDynamics::UpdateTriangleContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    auto const nTriangles = mMeshes.F.cols();
    tbb::parallel_for(Eigen::Index{0}, nTriangles, [&](std::size_t f) {
        EnvironmentContact& c   = CF[f];
        bool const bIsInContact = mMeshSdfContact.mTriangleContactMask(f);
        if (bIsInContact)
        {
            Eigen::Matrix<ScalarType, 3, 3> const xf =
                X(Eigen::placeholders::all, mMeshes.F.col(f));
            Eigen::Vector<ScalarType, 2> const uv = mMeshSdfContact.mTriangleContactPoints.col(f);
            Eigen::Vector<ScalarType, 3> const xc =
                (ScalarType(1) - uv(0) - uv(1)) * xf.col(0) + uv(0) * xf.col(1) + uv(1) * xf.col(2);
            detail::ComputeEnvironmentContactBasis(xc, mSdf, mMeshSdfContact.mParams.hfd, c.O, c.B);
            c.mu = mEnvContactDynamicsParams.mu;
        }
        else
        {
            c.C      = ScalarType(0);
            c.lambda = ScalarType(0);
        }
    });
}

void MeshDynamics::UpdateHalfEdgeContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    auto const nHalfEdges = mMeshes.F.cols() * 3;
    tbb::parallel_for(Eigen::Index{0}, nHalfEdges, [&](Eigen::Index he) {
        EnvironmentContact& c   = CHE[he];
        bool const bIsInContact = mMeshSdfContact.mHalfEdgeContactMask(he);
        if (bIsInContact)
        {
            Eigen::Vector<ScalarType, 3> const xi = X.col(geometry::IncomingVertex(mMeshes.F, he));
            Eigen::Vector<ScalarType, 3> const xj = X.col(geometry::OutgoingVertex(mMeshes.F, he));
            ScalarType const u                    = mMeshSdfContact.mHalfEdgeContactPoints(he);
            Eigen::Vector<ScalarType, 3> const xc = (1 - u) * xi + u * xj;
            detail::ComputeEnvironmentContactBasis(xc, mSdf, mMeshSdfContact.mParams.hfd, c.O, c.B);
            c.mu = mEnvContactDynamicsParams.mu;
        }
        else
        {
            c.C      = ScalarType(0);
            c.lambda = ScalarType(0);
        }
    });
}

void MeshDynamics::UpdateVertexContactConstraints(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X)
{
    auto const nVertices = mMeshes.V.size();
    tbb::parallel_for(Eigen::Index{0}, nVertices, [&](Eigen::Index v) {
        EnvironmentContact& c   = CV[v];
        bool const bIsInContact = mMeshSdfContact.mVertexContactMask(v);
        if (bIsInContact)
        {
            Eigen::Vector<ScalarType, 3> const xc = X.col(mMeshes.V(v));
            detail::ComputeEnvironmentContactBasis(xc, mSdf, mMeshSdfContact.mParams.hfd, c.O, c.B);
            c.mu = mEnvContactDynamicsParams.mu;
        }
        else
        {
            c.C      = ScalarType(0);
            c.lambda = ScalarType(0);
        }
    });
}

} // namespace pbat::sim::contact

#include "pbat/geometry/model/Cube.h"
#include "pbat/graph/Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshDynamics")
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
    meshDynamics.Construct(std::move(meshes), std::move(sdfForest));

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
    // Check that contact constraint vectors are initially empty
    CHECK_EQ(meshDynamics.CF.size(), 0);
    CHECK_EQ(meshDynamics.CHE.size(), 0);
    CHECK_EQ(meshDynamics.CV.size(), 0);

    SUBCASE("Mesh-SDF environment contacts")
    {
        // Act
        meshDynamics.InitializeMeshEnvironmentContactDetection();
        meshDynamics.UpdateEnvironmentContactConstraints(X);
        meshDynamics.PrepareEnvironmentContactsForDualIteration();

        // Assert
        // Check that geometric arrays are sized correctly
        CHECK_EQ(meshDynamics.FA.size(), meshDynamics.mMeshes.F.cols());
        CHECK_EQ(meshDynamics.HEA.size(), 3 * meshDynamics.mMeshes.F.cols());
        CHECK_EQ(meshDynamics.VA.size(), meshDynamics.mMeshes.V.size());

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