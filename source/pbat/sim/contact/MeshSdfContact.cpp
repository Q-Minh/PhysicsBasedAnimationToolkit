#include "MeshSdfContact.h"

#include "pbat/common/Atomic.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/optimization/TriangleConstrainedTrustRegionSr1.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <random>
#include <span>
#include <tbb/parallel_for.h>

namespace pbat::sim::contact {

MeshSdfContact::MeshSdfContact(
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    MeshSdfContactParams const& params)
    : MeshSdfContact()
{
    Initialize(V, F, params);
}

void MeshSdfContact::Initialize(
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    MeshSdfContactParams const& params)
{
    mParams               = params;
    auto const nTriangles = F.cols();
    auto const nHalfEdges = 3 * F.cols();
    auto const nVertices  = V.size();
    mTriangleContactPoints.resize(nTriangles);
    mHalfEdgeContactPoints.resize(nHalfEdges);
    mVertexContactPoints.resize(nVertices);
    mHalfEdgeLocks.resize(nHalfEdges);
    for (auto& contactPoints : mTriangleContactPoints)
        contactPoints.reserve(mParams.nMaxContactsPerTriangle);
    for (auto& contactPoints : mHalfEdgeContactPoints)
        contactPoints.reserve(2 * mParams.nMaxContactsPerTriangle);
}

void MeshSdfContact::PrepareIteration()
{
    for (auto& contactPoints : mTriangleContactPoints)
        contactPoints.clear();
    for (auto& contactPoints : mHalfEdgeContactPoints)
        contactPoints.clear();
    mVertexContactPoints.setConstant(false);
    mHalfEdgeLocks.setConstant(false);
}

void MeshSdfContact::TriangleSdfContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    geometry::sdf::Composite<ScalarType> const& sdf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.TriangleSdfContactDetection");
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        // Setup thread-safe random number generation
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        thread_local std::uniform_real_distribution<ScalarType> dis(ScalarType(0), ScalarType(1));
        // 1. Cull triangles that are too far from the surface before doing anything, assume
        // Lipschitz bound of 1 on the SDF.
        namespace mini                           = pbat::math::linalg::mini;
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, F.col(f));
        Eigen::Vector<ScalarType, 3> const xbary = xf.rowwise().sum() / ScalarType(3);
        ScalarType signedDistanceAtBarycenter    = sdf.Eval(mini::FromEigen(xbary));
        Eigen::Vector<ScalarType, 3> const distancesToVertices =
            (xf.colwise() - xbary).colwise().norm();
        ScalarType const maxDistanceToVertex = distancesToVertices.maxCoeff();
        ScalarType const signedDistanceLowerBound =
            signedDistanceAtBarycenter - maxDistanceToVertex;
        // TODO: Parameterize Lipschitz bound
        if (signedDistanceLowerBound > mParams.r)
            return;
        // 2. Run SDF minimization using TR-SR1 on randomly sampled points on the triangle to find
        // penetration points (from which contact points can be computed).
        mini::SMatrix<ScalarType, 3, 2> const DX =
            mini::FromEigen(xf.rightCols<2>().colwise() - xf.col(0));
        mini::SVector<ScalarType, 3> const A = mini::FromEigen(xf.col(0));
        math::optimization::TriangleConstrainedTrustRegionSr1Params<ScalarType> trParams{};
        ScalarType const triangleSizeMeasure = 2 * maxDistanceToVertex;
        trParams.sigmaB                      = mParams.sigmaB * triangleSizeMeasure;
        trParams.R0                          = mParams.sigmaR * triangleSizeMeasure;
        trParams.nMaxIters                   = mParams.nMaxOptimizationIterationsPerTriangle;
        for (auto c = 0; c < mParams.nMaxContactsPerTriangle; ++c)
        {
            // Randomly sample uniformly in triangle for initial iterate
            mini::SVector<ScalarType, 2> xk;
            xk(0) = dis(gen);
            xk(1) = dis(gen);
            if (xk(0) + xk(1) > ScalarType(1))
            {
                xk(0) = ScalarType(1) - xk(0);
                xk(1) = ScalarType(1) - xk(1);
            }
            // Run TR-SR1 optimization
            bool const bConverged = math::optimization::TriangleConstrainedTrustRegionSr1(
                [&](mini::SVector<ScalarType, 2> const& bary) {
                    mini::SVector<ScalarType, 3> p = A + DX * bary;
                    return sdf.Eval(p);
                } /*f(x)*/,
                [&](mini::SVector<ScalarType, 2> const& bary) -> mini::SVector<ScalarType, 2> {
                    mini::SVector<ScalarType, 3> p  = A + DX * bary;
                    mini::SVector<ScalarType, 3> gx = sdf.Grad(p, mParams.hfd);
                    return DX.Transpose() * gx;
                } /*g(x)*/,
                [&](mini::SVector<ScalarType, 2> const& /*bary*/,
                    ScalarType ared,
                    ScalarType pred,
                    bool /*bStepAccepted*/) {
                    // Because the SDF is non-smooth everywhere, we need to provide a custom
                    // convergence check to the minimizer. Maybe we could check if the actual
                    // reduction was small and non-positive, and that the predicted reduction is
                    // small enough, and then we could consider the minimization converged. The
                    // notion of "small enough" should be relative to the triangle's size, and we
                    // can note that the SDF is supposed to compute true distance values, hence we
                    // can measure reductions in units of meters. If the actual reduction is
                    // positive, then we obviously don't have a good approximation to the problem,
                    // hence we should continue iterating. The intuition is that if even our most
                    // optimistic reduction, which comes from the predicted reduction, is small, and
                    // the directions we take are not bad directions (i.e. they are not energy
                    // increasing), and we are not actually getting much closer to the surface (i.e.
                    // actual reduction is small), then we are likely close enough to a local
                    // minimum. Unfortunately, in the pathological yet frequent case of a triangle
                    // lying on a plane SDF, the predicted reductions will have a hard time
                    // decreasing, since we keep the hessian estimate positive definite, but the
                    // plane is flat. In that case, however, the gradient will actually be zero,
                    // since the range of the triangle is orthogonal to the SDF gradient, so we can
                    // also check for small gradient norm.
                    return (ared >= ScalarType(0)) and
                               (ared <= mParams.tauAred * triangleSizeMeasure) and
                               (pred <= mParams.tauPred * triangleSizeMeasure) or
                           (mini::SquaredNorm(trParams.gk) <= trParams.gzero * trParams.gzero);
                } /*fCheckConvergence*/,
                xk /*xk*/,
                trParams /*params*/);
            // Add (potentially duplicate) contact point if we found a penetrating point
            if (trParams.fk <= mParams.r)
                mTriangleContactPoints[f].push_back(mini::ToEigen(xk));
        }
    });
}

void MeshSdfContact::DeduplicateContactSet(
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.DeduplicateContactSet");
    DeduplicateTriangleContacts(F);
    ExtractLowerDimensionalContactsFromTriangleContacts(F, GHEF, GXV);
    DeduplicateHalfEdgeContacts();
}

void MeshSdfContact::Serialize(io::Archive& archive) const
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    group.WriteData("mTriangleContactPoints", mTriangleContactPoints);
    group.WriteData("mHalfEdgeContactPoints", mHalfEdgeContactPoints);
    group.WriteData("mVertexContactPoints", mVertexContactPoints.cast<int>().eval());
    group.WriteData("mHalfEdgeLocks", mHalfEdgeLocks.cast<int>().eval());
    mParams.Serialize(group);
}

void MeshSdfContact::Deserialize(io::Archive& archive)
{
    io::Archive group      = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    mTriangleContactPoints = group.ReadData<std::vector<std::vector<Eigen::Vector<ScalarType, 2>>>>(
        "mTriangleContactPoints");
    mHalfEdgeContactPoints =
        group.ReadData<std::vector<std::vector<ScalarType>>>("mHalfEdgeContactPoints");
    mVertexContactPoints =
        group.ReadData<Eigen::Vector<int, Eigen::Dynamic>>("mVertexContactPoints").cast<bool>();
    mHalfEdgeLocks =
        group.ReadData<Eigen::Vector<int, Eigen::Dynamic>>("mHalfEdgeLocks").cast<bool>();
    mParams.Deserialize(group);
}

void MeshSdfContact::DeduplicateTriangleContacts(
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.DeduplicateTriangleContacts");
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        // Sort and de-duplicate contact points
        std::sort(
            mTriangleContactPoints[f].begin(),
            mTriangleContactPoints[f].end(),
            [](Eigen::Vector<ScalarType, 2> const& a, Eigen::Vector<ScalarType, 2> const& b) {
                return (a[0] < b[0]) or ((a[0] == b[0]) and (a[1] < b[1]));
            });
        auto it = std::unique(
            mTriangleContactPoints[f].begin(),
            mTriangleContactPoints[f].end(),
            [&](Eigen::Vector<ScalarType, 2> const& a, Eigen::Vector<ScalarType, 2> const& b) {
                return a.isApprox(b, mParams.coordZero);
            });
        mTriangleContactPoints[f].erase(it, mTriangleContactPoints[f].end());
    });
}

void MeshSdfContact::ExtractLowerDimensionalContactsFromTriangleContacts(
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshSdfContact.ExtractLowerDimensionalContactsFromTriangleContacts");
    tbb::parallel_for(std::size_t(0), mTriangleContactPoints.size(), [&](std::size_t f) {
        // Contact points are expressed as (1-u-v)*a + u*b + v*c where a,b,c are the triangle
        // vertices, and satisfy u >= 0, v >= 0, u + v <= 1.
        std::vector<Eigen::Vector<ScalarType, 2>>& contactPoints = mTriangleContactPoints[f];
        std::size_t nTriangleContactPoints                       = contactPoints.size();
        for (std::size_t c = 0; c < nTriangleContactPoints;)
        {
            Eigen::Vector<ScalarType, 2> const& uv = contactPoints[c];
            // bEdgeMask[0] == true -> vertex 0 has barycentric coord 0
            // bEdgeMask[1] == true -> vertex 1 has barycentric coord 0
            // bEdgeMask[2] == true -> vertex 2 has barycentric coord 0
            std::array<bool, 3> const bEdgeMask{
                (uv[0] + uv[1] == ScalarType(1)),
                uv[0] == ScalarType(0),
                uv[1] == ScalarType(0)};
            int nZeros           = bEdgeMask[0] + bEdgeMask[1] + bEdgeMask[2];
            bool const bIsVertex = nZeros == 2;
            bool const bIsEdge   = nZeros == 1;
            if (bIsVertex)
            {
                int const ilocal =
                    /* (uv[0] + uv[1] == ScalarType(0)) * 0 + */ (uv[0] == ScalarType(1)) * 1 +
                    (uv[1] == ScalarType(1)) * 2;
                IndexType const v = GXV(F(ilocal, f));
                std::atomic_ref<bool> bIsContacting{mVertexContactPoints(v)};
                bIsContacting.store(true, std::memory_order_relaxed);
                // Remove triangle contact point and add vertex contact point
                std::swap(contactPoints[c], contactPoints[--nTriangleContactPoints]);
            }
            else if (bIsEdge)
            {
                // If vertex 0 has barycentric coord 0, then the half-edge starts from vertex 1
                // If vertex 1 has barycentric coord 0, then the half-edge starts from vertex 2
                // If vertex 2 has barycentric coord 0, then the half-edge starts from vertex 0
                int const helocal = bEdgeMask[0] * 1 + bEdgeMask[1] * 2 /* + bEdgeMask[2] * 0*/;
                // The triangle's vertices are weighted as (1-u-v), u, v, respectively.
                // Let the half-edge barycentric coordinates be s,t, then if the contacting point is
                // on
                // 1. the first half-edge (helocal == 0), then s=(1-u-v), t=u
                // 2. the second half-edge (helocal == 1), then s=u, t=v
                // 3. the third half-edge (helocal == 2), then s=v, t=(1-u-v)
                // Since points on the edge are formulated as xc = s*a + t*b where a,b are the
                // edge's vertices, and s+t=1, then we need only store t as xc = (1-t)*a + t*b. In
                // other words, we store t for the half-edge contact point. The opposite half-edge
                // has its vertex indices reversed, so we store s for it.
                // clang-format off
                ScalarType const s = (helocal == 0)*(ScalarType(1) - uv[0] - uv[1]) +
                                     (helocal == 1)*uv[0] +
                                     (helocal == 2)*uv[1];
                ScalarType const t = (helocal == 0)*uv[0] +
                                     (helocal == 1)*uv[1] +
                                     (helocal == 2)*(ScalarType(1) - uv[0] - uv[1]);
                // clang-format on
                IndexType const hei = 3 * f + helocal;
                IndexType const hej = geometry::OppositeHalfEdge(F, hei, GHEF);
                common::AtomicExecute(mHalfEdgeLocks(hei), [&]() {
                    mHalfEdgeContactPoints[hei].push_back(t);
                });
                common::AtomicExecute(mHalfEdgeLocks(hej), [&]() {
                    mHalfEdgeContactPoints[hej].push_back(s);
                });
                // Remove triangle contact point and add half-edge contact point
                std::swap(contactPoints[c], contactPoints[--nTriangleContactPoints]);
            }
            else
            {
                ++c;
            }
        }
        contactPoints.erase(contactPoints.begin() + nTriangleContactPoints, contactPoints.end());
    });
}

void MeshSdfContact::DeduplicateHalfEdgeContacts()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.DeduplicateHalfEdgeContacts");
    tbb::parallel_for(std::size_t(0), mHalfEdgeContactPoints.size(), [&](std::size_t he) {
        // Sort and de-duplicate contact points
        std::sort(mHalfEdgeContactPoints[he].begin(), mHalfEdgeContactPoints[he].end());
        auto it = std::unique(mHalfEdgeContactPoints[he].begin(), mHalfEdgeContactPoints[he].end());
        mHalfEdgeContactPoints[he].erase(it, mHalfEdgeContactPoints[he].end());
    });
}

MeshSdfContactParams&
MeshSdfContactParams::WithInitializationStrategy(ScalarType _sigmaR, ScalarType _sigmaB)
{
    sigmaR = _sigmaR;
    sigmaB = _sigmaB;
    return *this;
}

MeshSdfContactParams& MeshSdfContactParams::WithTerminationCriteria(
    ScalarType _tauAred,
    ScalarType _tauPred,
    int _nMaxOptimizationIterationsPerTriangle)
{
    tauAred                               = _tauAred;
    tauPred                               = _tauPred;
    nMaxOptimizationIterationsPerTriangle = _nMaxOptimizationIterationsPerTriangle;
    return *this;
}

MeshSdfContactParams& MeshSdfContactParams::WithContactStorageLimits(int _nMaxContactsPerTriangle)
{
    nMaxContactsPerTriangle = _nMaxContactsPerTriangle;
    return *this;
}

MeshSdfContactParams&
MeshSdfContactParams::WithNumericalParameters(ScalarType _coordZero, ScalarType _hfd, ScalarType _r)
{
    coordZero = _coordZero;
    hfd       = _hfd;
    r         = _r;
    return *this;
}

MeshSdfContactParams& MeshSdfContactParams::Construct(bool bValidate)
{
    if (bValidate)
    {
        if (sigmaR <= ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): sigmaR must be > 0.");
        }
        if (sigmaB <= ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): sigmaB must be > 0.");
        }
        if (tauAred <= ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): tauAred must be > 0.");
        }
        if (tauPred <= ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): tauPred must be > 0.");
        }
        if (nMaxContactsPerTriangle <= 0)
        {
            throw std::invalid_argument(
                "MeshSdfContactParams::Construct(): nMaxContactsPerTriangle must be > 0.");
        }
        if (nMaxOptimizationIterationsPerTriangle <= 0)
        {
            throw std::invalid_argument(
                "MeshSdfContactParams::Construct(): nMaxOptimizationIterationsPerTriangle must be "
                "> 0.");
        }
        if (coordZero <= ScalarType(0))
        {
            throw std::invalid_argument(
                "MeshSdfContactParams::Construct(): coordZero must be > 0.");
        }
        if (hfd <= ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): hfd must be > 0.");
        }
        if (r < ScalarType(0))
        {
            throw std::invalid_argument("MeshSdfContactParams::Construct(): r must be >= 0.");
        }
    }
    return *this;
}

void MeshSdfContactParams::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.contact.MeshSdfContactParams"];
    group.WriteMetaData("sigmaR", sigmaR);
    group.WriteMetaData("sigmaB", sigmaB);
    group.WriteMetaData("tauAred", tauAred);
    group.WriteMetaData("tauPred", tauPred);
    group.WriteMetaData("nMaxContactsPerTriangle", nMaxContactsPerTriangle);
    group.WriteMetaData(
        "nMaxOptimizationIterationsPerTriangle",
        nMaxOptimizationIterationsPerTriangle);
    group.WriteMetaData("coordZero", coordZero);
    group.WriteMetaData("hfd", hfd);
    group.WriteMetaData("r", r);
}

void MeshSdfContactParams::Deserialize(io::Archive& archive)
{
    io::Archive group       = archive["pbat.sim.contact.MeshSdfContactParams"];
    sigmaR                  = group.ReadMetaData<ScalarType>("sigmaR");
    sigmaB                  = group.ReadMetaData<ScalarType>("sigmaB");
    tauAred                 = group.ReadMetaData<ScalarType>("tauAred");
    tauPred                 = group.ReadMetaData<ScalarType>("tauPred");
    nMaxContactsPerTriangle = group.ReadMetaData<int>("nMaxContactsPerTriangle");
    nMaxOptimizationIterationsPerTriangle =
        group.ReadMetaData<int>("nMaxOptimizationIterationsPerTriangle");
    coordZero = group.ReadMetaData<ScalarType>("coordZero");
    hfd       = group.ReadMetaData<ScalarType>("hfd");
    r         = group.ReadMetaData<ScalarType>("r");
}

} // namespace pbat::sim::contact

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshSdfContact") {}