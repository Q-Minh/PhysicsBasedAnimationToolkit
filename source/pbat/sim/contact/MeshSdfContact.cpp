#include "MeshSdfContact.h"

#include "pbat/common/Atomic.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/optimization/GoldenSectionSearch.h"
#include "pbat/math/optimization/TriangleConstrainedTrustRegionSr1.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Geometry>
#include <array>
#include <atomic>
#include <tbb/parallel_for.h>

namespace pbat::sim::contact {

MeshSdfContact::MeshSdfContact(
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    MeshSdfContactParams const& params)
    : MeshSdfContact()
{
    mParams = params;
    Initialize(V, F);
}

void MeshSdfContact::Initialize(
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F)
{
    auto const nTriangles = F.cols();
    auto const nHalfEdges = 3 * nTriangles;
    auto const nVertices  = V.size();
    mTriangleContactMask.resize(nTriangles);
    mTriangleContactPoints.resize(2, nTriangles);
    mHalfEdgeContactMask.resize(nHalfEdges);
    mHalfEdgeContactPoints.resize(nHalfEdges);
    mVertexContactMask.resize(nVertices);
    for (auto f = 0; f < nTriangles; ++f)
        mTriangleContactPoints.col(f).setConstant(ScalarType(0.33));
    for (auto he = 0; he < nHalfEdges; ++he)
        mHalfEdgeContactPoints(he) = ScalarType(0.5);
    mVertexDisplacementBounds.resize(nVertices);
    mVertexLocks.setConstant(nVertices, false);
}

void MeshSdfContact::PrepareIteration()
{
    mTriangleContactMask.setConstant(true);
    mHalfEdgeContactMask.setConstant(true);
    mVertexContactMask.setConstant(true);
}

void MeshSdfContact::TriangleSdfContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV,
    geometry::sdf::Composite<ScalarType> const& sdf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.TriangleSdfContactDetection");
    // 0. Reset locks and bounds
    mVertexLocks.setConstant(false);
    mVertexDisplacementBounds.setConstant(std::numeric_limits<ScalarType>::max());
    // 1. Cull triangles that are too far from the surface before doing anything, assume
    // Lipschitz bound of 1 on the SDF.
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        namespace mini                           = pbat::math::linalg::mini;
        Eigen::Vector<IndexType, 3> const finds  = F.col(f);
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, finds);
        Eigen::Vector<ScalarType, 3> const xbary = xf.rowwise().sum() / ScalarType(3);
        ScalarType signedDistanceAtBarycenter    = sdf.Eval(mini::FromEigen(xbary));
        Eigen::Vector<ScalarType, 3> const distancesToVertices =
            (xf.colwise() - xbary).colwise().norm();
        ScalarType const maxDistanceToVertex = distancesToVertices.maxCoeff();
        ScalarType const signedDistanceLowerBound =
            signedDistanceAtBarycenter - maxDistanceToVertex;
        // TODO: Parameterize Lipschitz bound
        bool const bShouldCull = signedDistanceLowerBound > mParams.r;
        if (bShouldCull)
        {
            // Mark all triangle features as non-contact
            mTriangleContactMask(f) = false;
            for (int j = 0; j < 3; ++j)
            {
                // We need to mark both half-edges of the edge as non-contact,
                // since the edge-adjacent triangle may not cull the edge.
                IndexType const hej = 3 * f + j;
                std::atomic_ref<bool> bIsHejInContact(mHalfEdgeContactMask(hej));
                bIsHejInContact.store(false, std::memory_order_relaxed);
                IndexType const hei = geometry::OppositeHalfEdge(F, hej, GHEF);
                std::atomic_ref<bool> bIsHeiInContact(mHalfEdgeContactMask(hei));
                bIsHeiInContact.store(false, std::memory_order_relaxed);
                // Cull vertex as well
                std::atomic_ref<bool> bIsVertexInContact(mVertexContactMask(finds(j)));
                bIsVertexInContact.store(false, std::memory_order_relaxed);
            }
            // Compute conservative bound if triangle is culled
            for (IndexType i : finds)
                common::AtomicMin(mVertexDisplacementBounds(i), signedDistanceLowerBound);
        }
    });
    // 2. Run SDF minimization using TR-SR1 on potential triangle candidates
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        namespace mini        = pbat::math::linalg::mini;
        bool const bWarmStart = mTriangleContactMask(f);
        mini::SVector<ScalarType, 2> xk =
            bWarmStart ? mini::FromEigen(mTriangleContactPoints.col(f)) :
                         mini::SVector<ScalarType, 2>(ScalarType(0.33), ScalarType(0.33));
        Eigen::Vector<IndexType, 3> const finds  = F.col(f);
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, finds);
        mini::SMatrix<ScalarType, 3, 2> const DX =
            mini::FromEigen(xf.rightCols<2>().colwise() - xf.col(0));
        mini::SVector<ScalarType, 3> const A  = mini::FromEigen(xf.col(0));
        ScalarType const maxSquaredEdgeLength = std::max({
            (xf.col(1) - xf.col(0)).squaredNorm(),
            (xf.col(2) - xf.col(1)).squaredNorm(),
            (xf.col(0) - xf.col(2)).squaredNorm(),
        });
        math::optimization::TriangleConstrainedTrustRegionSr1Params<ScalarType> trParams{};
        ScalarType const triangleSizeMeasure = std::sqrt(maxSquaredEdgeLength);
        trParams.sigmaB                      = mParams.sigmaB * triangleSizeMeasure;
        trParams.R0                          = mParams.sigmaR * triangleSizeMeasure;
        trParams.nMaxIters                   = mParams.nMaxOptimizationIterations;
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
        bool const bIsInContactRadius         = trParams.fk <= mParams.r;
        bool const bIsLowerDimensionalContact = xk(0) <= mParams.coordZero or
                                                xk(1) <= mParams.coordZero or
                                                (1 - xk(0) - xk(1) <= mParams.coordZero);
        if (bIsInContactRadius and (not bIsLowerDimensionalContact))
        {
            mTriangleContactMask(f)       = true;
            mTriangleContactPoints.col(f) = mini::ToEigen(xk);
            // Update vertex displacement bounds if triangle in contact
            for (IndexType i : finds)
                common::AtomicMin(mVertexDisplacementBounds(i), trParams.fk - triangleSizeMeasure);
        }
    });
    // 3. Run SDF minimization on potential edge candidates
    auto const nEdges = EHE.cols();
    tbb::parallel_for(Eigen::Index(0), nEdges, [&](Eigen::Index e) {
        namespace mini        = pbat::math::linalg::mini;
        IndexType const hei   = EHE(0, e);
        IndexType const hej   = EHE(1, e);
        bool const bWarmStart = mHalfEdgeContactMask(hei);
        ScalarType xk         = bWarmStart ? mHalfEdgeContactPoints(hei) : ScalarType(0.5);
        IndexType const i     = geometry::IncomingVertex(F, hei);
        IndexType const j     = geometry::OutgoingVertex(F, hei);
        Eigen::Vector<ScalarType, 3> const x1  = X.col(i);
        Eigen::Vector<ScalarType, 3> const x2  = X.col(j);
        Eigen::Vector<ScalarType, 3> const x12 = x2 - x1;
        ScalarType const edgeLength            = x12.norm();
        math::optimization::GoldenSectionSearchResult<ScalarType> result =
            math::optimization::GoldenSectionSearch(
                [&](ScalarType u) {
                    mini::SVector<ScalarType, 3> p = mini::FromEigen(x1 + u * x12);
                    return sdf.Eval(p);
                },
                ScalarType(0),
                ScalarType(1),
                mParams.tauAred * edgeLength,
                mParams.nMaxOptimizationIterations);
        bool const bIsInContactRadius = result.fmin <= mParams.r;
        bool const bIsVertex =
            (result.xmin <= mParams.coordZero) or (1 - result.xmin <= mParams.coordZero);
        if (bIsInContactRadius and not bIsVertex)
        {
            mHalfEdgeContactMask(hei)   = true;
            mHalfEdgeContactPoints(hei) = result.xmin;
            common::AtomicMin(mVertexDisplacementBounds(i), result.fmin - edgeLength / 2);
            common::AtomicMin(mVertexDisplacementBounds(j), result.fmin - edgeLength / 2);
            if (hej >= 0)
            {
                mHalfEdgeContactMask(hej)   = true;
                mHalfEdgeContactPoints(hej) = 1 - result.xmin;
            }
        }
    });
    // 4. Cull vertices that are not in contact
    auto const nVerts = V.size();
    tbb::parallel_for(Eigen::Index(0), nVerts, [&](Eigen::Index v) {
        namespace mini          = pbat::math::linalg::mini;
        bool const bIsNotCulled = mVertexContactMask(v);
        if (bIsNotCulled)
        {
            ScalarType const sd           = sdf.Eval(mini::FromEigen(X.col(v)));
            bool const bIsInContactRadius = sd <= mParams.r;
            mVertexContactMask(v)         = bIsInContactRadius;
            mVertexDisplacementBounds(v)  = std::min(mVertexDisplacementBounds(v), sd);
        }
        mVertexDisplacementBounds(v) = std::max(mVertexDisplacementBounds(v), ScalarType(0));
    });
}

void MeshSdfContact::Serialize(io::Archive& archive) const
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    group.WriteData("mTriangleContactMask", mTriangleContactMask.cast<int>().eval());
    group.WriteData("mTriangleContactPoints", mTriangleContactPoints);
    group.WriteData("mHalfEdgeContactMask", mHalfEdgeContactMask.cast<int>().eval());
    group.WriteData("mHalfEdgeContactPoints", mHalfEdgeContactPoints);
    group.WriteData("mVertexContactMask", mVertexContactMask.cast<int>().eval());
    group.WriteData("mVertexDisplacementBounds", mVertexDisplacementBounds);
    mParams.Serialize(group);
}

void MeshSdfContact::Deserialize(io::Archive& archive)
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    mTriangleContactMask =
        group.ReadData<Eigen::Vector<int, Eigen::Dynamic>>("mTriangleContactMask").cast<bool>();
    mTriangleContactPoints =
        group.ReadData<Eigen::Matrix<ScalarType, 2, Eigen::Dynamic>>("mTriangleContactPoints");
    mHalfEdgeContactMask =
        group.ReadData<Eigen::Vector<int, Eigen::Dynamic>>("mHalfEdgeContactMask").cast<bool>();
    mHalfEdgeContactPoints =
        group.ReadData<Eigen::Vector<ScalarType, Eigen::Dynamic>>("mHalfEdgeContactPoints");
    mVertexContactMask =
        group.ReadData<Eigen::Vector<int, Eigen::Dynamic>>("mVertexContactMask").cast<bool>();
    mVertexDisplacementBounds =
        group.ReadData<Eigen::Vector<ScalarType, Eigen::Dynamic>>("mVertexDisplacementBounds");
    mParams.Deserialize(group);
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
    tauAred                    = _tauAred;
    tauPred                    = _tauPred;
    nMaxOptimizationIterations = _nMaxOptimizationIterationsPerTriangle;
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
        if (nMaxOptimizationIterations <= 0)
        {
            throw std::invalid_argument(
                "MeshSdfContactParams::Construct(): nMaxOptimizationIterations must be "
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
    group.WriteMetaData("nMaxOptimizationIterations", nMaxOptimizationIterations);
    group.WriteMetaData("coordZero", coordZero);
    group.WriteMetaData("hfd", hfd);
    group.WriteMetaData("r", r);
}

void MeshSdfContactParams::Deserialize(io::Archive& archive)
{
    io::Archive group          = archive["pbat.sim.contact.MeshSdfContactParams"];
    sigmaR                     = group.ReadMetaData<ScalarType>("sigmaR");
    sigmaB                     = group.ReadMetaData<ScalarType>("sigmaB");
    tauAred                    = group.ReadMetaData<ScalarType>("tauAred");
    tauPred                    = group.ReadMetaData<ScalarType>("tauPred");
    nMaxOptimizationIterations = group.ReadMetaData<int>("nMaxOptimizationIterations");
    coordZero                  = group.ReadMetaData<ScalarType>("coordZero");
    hfd                        = group.ReadMetaData<ScalarType>("hfd");
    r                          = group.ReadMetaData<ScalarType>("r");
}

} // namespace pbat::sim::contact

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshSdfContact") {}