#include "MeshSdfContact.h"

#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/optimization/TriangleConstrainedTrustRegionSr1.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Geometry>
#include <random>
#include <span>
#include <tbb/parallel_for.h>

namespace pbat::sim::contact {

MeshSdfContact::MeshSdfContact(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    MeshSdfContactParams const& params)
    : MeshSdfContact()
{
    Initialize(X, F, params);
}

void MeshSdfContact::Initialize(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    MeshSdfContactParams const& params)
{
    mTriangleContactCounts.resize(F.cols());
    mTriangleSdfContacts.resize(3 * mParams.nMaxContactsPerTriangle, F.cols());
    mParams = params;
}

void MeshSdfContact::TriangleSdfContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    geometry::sdf::Composite<ScalarType> const& sdf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshSdfContact.TriangleSdfContactDetection");
    // 0. Setup random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<ScalarType> dis(ScalarType(0), ScalarType(1));
    // 1. Reset contact data
    mTriangleContactCounts.setZero();
    // 2. Run triangle-sdf contact detection in parallel over all triangles
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        namespace mini                           = pbat::math::linalg::mini;
        Eigen::Matrix<ScalarType, 3, 3> const xf = X(Eigen::placeholders::all, F.col(f));
        // Cull triangles that are too far from the surface before doing anything, assume Lipschitz
        // bound of 1 on the SDF.
        // TODO: Parameterize Lipschitz bound
        Eigen::Vector<ScalarType, 3> const xbary = xf.rowwise().sum() / ScalarType(3);
        ScalarType signedDistanceAtBarycenter    = sdf.Eval(mini::FromEigen(xbary));
        Eigen::Vector<ScalarType, 3> const distancesToVertices =
            (xf.colwise() - xbary).colwise().norm();
        ScalarType const maxDistanceToVertex = distancesToVertices.maxCoeff();
        ScalarType const signedDistanceLowerBound =
            signedDistanceAtBarycenter - maxDistanceToVertex;
        if (signedDistanceLowerBound > mParams.r)
            return;
        // Run SDF minimization using TR-SR1 on randomly sampled points on the triangle to find
        // penetration points (from which contact points can be computed), then sort and
        // de-duplicate them.
        mini::SMatrix<ScalarType, 3, 2> const DX =
            mini::FromEigen(xf.rightCols<2>().colwise() - xf.col(0));
        mini::SVector<ScalarType, 3> const A = mini::FromEigen(xf.col(0));
        IndexType nContacts                  = 0;
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
            {
                mTriangleSdfContacts.col(f).segment<3>(3 * nContacts)
                    << ScalarType(1) - xk(0) - xk(1),
                    xk(0), xk(1);
                ++nContacts;
            }
        }
        mTriangleContactCounts(f) = nContacts;
    });
    DeduplicateTriangleContacts(X, F);
}

void MeshSdfContact::Serialize(io::Archive& archive) const
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    group.WriteData("mTriangleContactCounts", mTriangleContactCounts);
    group.WriteData("mTriangleSdfContacts", mTriangleSdfContacts);
    mParams.Serialize(group);
}

void MeshSdfContact::Deserialize(io::Archive& archive)
{
    io::Archive group = archive.GetOrCreateGroup("pbat.sim.contact.MeshSdfContact");
    mTriangleContactCounts =
        group.ReadData<Eigen::Vector<IndexType, Eigen::Dynamic>>("mTriangleContactCounts");
    mTriangleSdfContacts =
        group.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>(
            "mTriangleSdfContacts");
    mParams.Deserialize(group);
}

void MeshSdfContact::DeduplicateTriangleContacts(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F)
{
    tbb::parallel_for(Eigen::Index(0), F.cols(), [&](Eigen::Index f) {
        IndexType nContacts = mTriangleContactCounts(f);
        if (nContacts <= 0)
            return;
        // Sort and de-duplicate contact points
        std::span<std::array<ScalarType, 3>> contactPoints(
            reinterpret_cast<std::array<ScalarType, 3>*>(mTriangleSdfContacts.col(f).data()),
            nContacts);
        std::sort(
            contactPoints.begin(),
            contactPoints.end(),
            [](std::array<ScalarType, 3> const& a, std::array<ScalarType, 3> const& b) {
                return a < b;
            });
        auto it = std::unique(
            contactPoints.begin(),
            contactPoints.end(),
            [&](std::array<ScalarType, 3> const& a, std::array<ScalarType, 3> const& b) {
                return (std::abs(a[0] - b[0]) <= mParams.coordZero) and
                       (std::abs(a[1] - b[1]) <= mParams.coordZero) and
                       (std::abs(a[2] - b[2]) <= mParams.coordZero);
            });
        nContacts = static_cast<IndexType>(std::distance(contactPoints.begin(), it));
        mTriangleContactCounts(f) = nContacts;
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
