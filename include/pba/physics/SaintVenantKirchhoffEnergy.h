
#ifndef PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
#define PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H

#include "pba/aliases.h"

#include <cmath>
#include <tuple>

namespace pba {
namespace physics {

template <int Dims>
struct SaintVenantKirchhoffEnergy;

template <>
struct SaintVenantKirchhoffEnergy<1>
{
    public:
        static auto constexpr kDims = 1;
    
        template <class Derived>
        Scalar
        eval(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Vector<1>
        grad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Matrix<1,1>
        hessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<1>>
        evalWithGrad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<1>, Matrix<1,1>>
        evalWithGradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Vector<1>, Matrix<1,1>>
        gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;
};

template <class Derived>
Scalar
SaintVenantKirchhoffEnergy<1>::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Scalar const a0 = (((1.0/2.0)*((F[0])*(F[0])) - 1.0/2.0)*((1.0/2.0)*((F[0])*(F[0])) - 1.0/2.0));
    psi = (1.0/2.0)*a0*lambda + a0*mu;
    return psi;
}

template <class Derived>
Vector<1>
SaintVenantKirchhoffEnergy<1>::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Vector<1> G;
    auto vecG = G.reshaped();
    Scalar const a0 = ((1.0/2.0)*((F[0])*(F[0])) - 1.0/2.0)*F[0];
    vecG[0] = a0*lambda + 2*a0*mu;
    return G;
}

template <class Derived>
Matrix<1,1>
SaintVenantKirchhoffEnergy<1>::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Matrix<1,1> H;
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = 2*mu;
    Scalar const a2 = (1.0/2.0)*a0 - 1.0/2.0;
    vecH[0] = a0*a1 + a0*lambda + a1*a2 + a2*lambda;
    return H;
}

template <class Derived>
std::tuple<Scalar, Vector<1>>
SaintVenantKirchhoffEnergy<1>::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<1> G;
    auto vecG = G.reshaped();
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) - 1.0/2.0;
    Scalar const a1 = ((a0)*(a0));
    Scalar const a2 = a0*F[0];
    psi = (1.0/2.0)*a1*lambda + a1*mu;
    vecG[0] = a2*lambda + 2*a2*mu;
    return {psi, G};
}

template <class Derived>
std::tuple<Scalar, Vector<1>, Matrix<1,1>>
SaintVenantKirchhoffEnergy<1>::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<1> G;
    Matrix<1,1> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = (1.0/2.0)*a0 - 1.0/2.0;
    Scalar const a2 = ((a1)*(a1));
    Scalar const a3 = a1*lambda;
    Scalar const a4 = 2*mu;
    Scalar const a5 = a1*a4;
    psi = (1.0/2.0)*a2*lambda + a2*mu;
    vecG[0] = a3*F[0] + a5*F[0];
    vecH[0] = a0*a4 + a0*lambda + a3 + a5;
    return {psi, G, H};
}

template <class Derived>
std::tuple<Vector<1>, Matrix<1,1>>
SaintVenantKirchhoffEnergy<1>::gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
{
    Vector<1> G;
    Matrix<1,1> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = (1.0/2.0)*a0 - 1.0/2.0;
    Scalar const a2 = a1*lambda;
    Scalar const a3 = 2*mu;
    Scalar const a4 = a1*a3;
    vecG[0] = a2*F[0] + a4*F[0];
    vecH[0] = a0*a3 + a0*lambda + a2 + a4;
    return {G, H};
}

template <>
struct SaintVenantKirchhoffEnergy<2>
{
    public:
        static auto constexpr kDims = 2;
    
        template <class Derived>
        Scalar
        eval(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Vector<4>
        grad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Matrix<4,4>
        hessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<4>>
        evalWithGrad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<4>, Matrix<4,4>>
        evalWithGradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Vector<4>, Matrix<4,4>>
        gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;
};

template <class Derived>
Scalar
SaintVenantKirchhoffEnergy<2>::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1]));
    Scalar const a1 = (1.0/2.0)*((F[2])*(F[2])) + (1.0/2.0)*((F[3])*(F[3]));
    psi = (1.0/2.0)*lambda*((a0 + a1 - 1)*(a0 + a1 - 1)) + mu*(((a0 - 1.0/2.0)*(a0 - 1.0/2.0)) + ((a1 - 1.0/2.0)*(a1 - 1.0/2.0)) + 2*(((1.0/2.0)*F[0]*F[2] + (1.0/2.0)*F[1]*F[3])*((1.0/2.0)*F[0]*F[2] + (1.0/2.0)*F[1]*F[3])));
    return psi;
}

template <class Derived>
Vector<4>
SaintVenantKirchhoffEnergy<2>::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Vector<4> G;
    auto vecG = G.reshaped();
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1]));
    Scalar const a1 = (1.0/2.0)*((F[2])*(F[2])) + (1.0/2.0)*((F[3])*(F[3]));
    Scalar const a2 = lambda*(a0 + a1 - 1);
    Scalar const a3 = 2*a0 - 1;
    Scalar const a4 = F[0]*F[2] + F[1]*F[3];
    Scalar const a5 = 2*a1 - 1;
    vecG[0] = a2*F[0] + mu*(a3*F[0] + a4*F[2]);
    vecG[1] = a2*F[1] + mu*(a3*F[1] + a4*F[3]);
    vecG[2] = a2*F[2] + mu*(a4*F[0] + a5*F[2]);
    vecG[3] = a2*F[3] + mu*(a4*F[1] + a5*F[3]);
    return G;
}

template <class Derived>
Matrix<4,4>
SaintVenantKirchhoffEnergy<2>::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Matrix<4,4> H;
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = ((F[2])*(F[2]));
    Scalar const a3 = a1 + a2 - 1;
    Scalar const a4 = ((F[3])*(F[3]));
    Scalar const a5 = lambda*((1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a2 + (1.0/2.0)*a4 - 1);
    Scalar const a6 = F[0]*F[1];
    Scalar const a7 = F[2]*F[3];
    Scalar const a8 = a6*lambda + mu*(2*a6 + a7);
    Scalar const a9 = F[0]*F[2];
    Scalar const a10 = F[1]*F[3];
    Scalar const a11 = a9*lambda + mu*(a10 + 2*a9);
    Scalar const a12 = F[0]*F[3];
    Scalar const a13 = F[1]*F[2];
    Scalar const a14 = a12*lambda + a13*mu;
    Scalar const a15 = a0 + a4 - 1;
    Scalar const a16 = a12*mu + a13*lambda;
    Scalar const a17 = a10*lambda + mu*(2*a10 + a9);
    Scalar const a18 = a7*lambda + mu*(a6 + 2*a7);
    vecH[0] = a0*lambda + a5 + mu*(3*a0 + a3);
    vecH[1] = a8;
    vecH[2] = a11;
    vecH[3] = a14;
    vecH[4] = a8;
    vecH[5] = a1*lambda + a5 + mu*(3*a1 + a15);
    vecH[6] = a16;
    vecH[7] = a17;
    vecH[8] = a11;
    vecH[9] = a16;
    vecH[10] = a2*lambda + a5 + mu*(a15 + 3*a2);
    vecH[11] = a18;
    vecH[12] = a14;
    vecH[13] = a17;
    vecH[14] = a18;
    vecH[15] = a4*lambda + a5 + mu*(a3 + 3*a4);
    return H;
}

template <class Derived>
std::tuple<Scalar, Vector<4>>
SaintVenantKirchhoffEnergy<2>::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<4> G;
    auto vecG = G.reshaped();
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1]));
    Scalar const a1 = (1.0/2.0)*((F[2])*(F[2])) + (1.0/2.0)*((F[3])*(F[3]));
    Scalar const a2 = a0 + a1 - 1;
    Scalar const a3 = a0 - 1.0/2.0;
    Scalar const a4 = a1 - 1.0/2.0;
    Scalar const a5 = (1.0/2.0)*F[0]*F[2] + (1.0/2.0)*F[1]*F[3];
    Scalar const a6 = a2*lambda;
    Scalar const a7 = 2*a3;
    Scalar const a8 = 2*a5;
    Scalar const a9 = 2*a4;
    psi = (1.0/2.0)*((a2)*(a2))*lambda + mu*(((a3)*(a3)) + ((a4)*(a4)) + 2*((a5)*(a5)));
    vecG[0] = a6*F[0] + mu*(a7*F[0] + a8*F[2]);
    vecG[1] = a6*F[1] + mu*(a7*F[1] + a8*F[3]);
    vecG[2] = a6*F[2] + mu*(a8*F[0] + a9*F[2]);
    vecG[3] = a6*F[3] + mu*(a8*F[1] + a9*F[3]);
    return {psi, G};
}

template <class Derived>
std::tuple<Scalar, Vector<4>, Matrix<4,4>>
SaintVenantKirchhoffEnergy<2>::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<4> G;
    Matrix<4,4> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = (1.0/2.0)*a0 + (1.0/2.0)*a1;
    Scalar const a3 = ((F[2])*(F[2]));
    Scalar const a4 = ((F[3])*(F[3]));
    Scalar const a5 = (1.0/2.0)*a3 + (1.0/2.0)*a4;
    Scalar const a6 = a2 + a5 - 1;
    Scalar const a7 = a2 - 1.0/2.0;
    Scalar const a8 = a5 - 1.0/2.0;
    Scalar const a9 = F[0]*F[2];
    Scalar const a10 = F[1]*F[3];
    Scalar const a11 = (1.0/2.0)*a10 + (1.0/2.0)*a9;
    Scalar const a12 = a6*lambda;
    Scalar const a13 = 2*a7;
    Scalar const a14 = 2*a11;
    Scalar const a15 = 2*a8;
    Scalar const a16 = a1 + a3 - 1;
    Scalar const a17 = F[0]*F[1];
    Scalar const a18 = F[2]*F[3];
    Scalar const a19 = a17*lambda + mu*(2*a17 + a18);
    Scalar const a20 = a9*lambda + mu*(a10 + 2*a9);
    Scalar const a21 = F[0]*F[3];
    Scalar const a22 = F[1]*F[2];
    Scalar const a23 = a21*lambda + a22*mu;
    Scalar const a24 = a0 + a4 - 1;
    Scalar const a25 = a21*mu + a22*lambda;
    Scalar const a26 = a10*lambda + mu*(2*a10 + a9);
    Scalar const a27 = a18*lambda + mu*(a17 + 2*a18);
    psi = (1.0/2.0)*((a6)*(a6))*lambda + mu*(2*((a11)*(a11)) + ((a7)*(a7)) + ((a8)*(a8)));
    vecG[0] = a12*F[0] + mu*(a13*F[0] + a14*F[2]);
    vecG[1] = a12*F[1] + mu*(a13*F[1] + a14*F[3]);
    vecG[2] = a12*F[2] + mu*(a14*F[0] + a15*F[2]);
    vecG[3] = a12*F[3] + mu*(a14*F[1] + a15*F[3]);
    vecH[0] = a0*lambda + a12 + mu*(3*a0 + a16);
    vecH[1] = a19;
    vecH[2] = a20;
    vecH[3] = a23;
    vecH[4] = a19;
    vecH[5] = a1*lambda + a12 + mu*(3*a1 + a24);
    vecH[6] = a25;
    vecH[7] = a26;
    vecH[8] = a20;
    vecH[9] = a25;
    vecH[10] = a12 + a3*lambda + mu*(a24 + 3*a3);
    vecH[11] = a27;
    vecH[12] = a23;
    vecH[13] = a26;
    vecH[14] = a27;
    vecH[15] = a12 + a4*lambda + mu*(a16 + 3*a4);
    return {psi, G, H};
}

template <class Derived>
std::tuple<Vector<4>, Matrix<4,4>>
SaintVenantKirchhoffEnergy<2>::gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
{
    Vector<4> G;
    Matrix<4,4> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = (1.0/2.0)*a0 + (1.0/2.0)*a1;
    Scalar const a3 = ((F[2])*(F[2]));
    Scalar const a4 = ((F[3])*(F[3]));
    Scalar const a5 = (1.0/2.0)*a3 + (1.0/2.0)*a4;
    Scalar const a6 = lambda*(a2 + a5 - 1);
    Scalar const a7 = 2*a2 - 1;
    Scalar const a8 = F[0]*F[2];
    Scalar const a9 = F[1]*F[3];
    Scalar const a10 = a8 + a9;
    Scalar const a11 = 2*a5 - 1;
    Scalar const a12 = a1 + a3 - 1;
    Scalar const a13 = F[0]*F[1];
    Scalar const a14 = F[2]*F[3];
    Scalar const a15 = a13*lambda + mu*(2*a13 + a14);
    Scalar const a16 = a8*lambda + mu*(2*a8 + a9);
    Scalar const a17 = F[0]*F[3];
    Scalar const a18 = F[1]*F[2];
    Scalar const a19 = a17*lambda + a18*mu;
    Scalar const a20 = a0 + a4 - 1;
    Scalar const a21 = a17*mu + a18*lambda;
    Scalar const a22 = a9*lambda + mu*(a8 + 2*a9);
    Scalar const a23 = a14*lambda + mu*(a13 + 2*a14);
    vecG[0] = a6*F[0] + mu*(a10*F[2] + a7*F[0]);
    vecG[1] = a6*F[1] + mu*(a10*F[3] + a7*F[1]);
    vecG[2] = a6*F[2] + mu*(a10*F[0] + a11*F[2]);
    vecG[3] = a6*F[3] + mu*(a10*F[1] + a11*F[3]);
    vecH[0] = a0*lambda + a6 + mu*(3*a0 + a12);
    vecH[1] = a15;
    vecH[2] = a16;
    vecH[3] = a19;
    vecH[4] = a15;
    vecH[5] = a1*lambda + a6 + mu*(3*a1 + a20);
    vecH[6] = a21;
    vecH[7] = a22;
    vecH[8] = a16;
    vecH[9] = a21;
    vecH[10] = a3*lambda + a6 + mu*(a20 + 3*a3);
    vecH[11] = a23;
    vecH[12] = a19;
    vecH[13] = a22;
    vecH[14] = a23;
    vecH[15] = a4*lambda + a6 + mu*(a12 + 3*a4);
    return {G, H};
}

template <>
struct SaintVenantKirchhoffEnergy<3>
{
    public:
        static auto constexpr kDims = 3;
    
        template <class Derived>
        Scalar
        eval(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Vector<9>
        grad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Matrix<9,9>
        hessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<9>>
        evalWithGrad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<9>, Matrix<9,9>>
        evalWithGradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Vector<9>, Matrix<9,9>>
        gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;
};

template <class Derived>
Scalar
SaintVenantKirchhoffEnergy<3>::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1])) + (1.0/2.0)*((F[2])*(F[2]));
    Scalar const a1 = (1.0/2.0)*((F[3])*(F[3])) + (1.0/2.0)*((F[4])*(F[4])) + (1.0/2.0)*((F[5])*(F[5]));
    Scalar const a2 = (1.0/2.0)*((F[6])*(F[6])) + (1.0/2.0)*((F[7])*(F[7])) + (1.0/2.0)*((F[8])*(F[8]));
    Scalar const a3 = (1.0/2.0)*F[0];
    Scalar const a4 = (1.0/2.0)*F[1];
    Scalar const a5 = (1.0/2.0)*F[2];
    psi = (1.0/2.0)*lambda*((a0 + a1 + a2 - 3.0/2.0)*(a0 + a1 + a2 - 3.0/2.0)) + mu*(((a0 - 1.0/2.0)*(a0 - 1.0/2.0)) + ((a1 - 1.0/2.0)*(a1 - 1.0/2.0)) + ((a2 - 1.0/2.0)*(a2 - 1.0/2.0)) + 2*((a3*F[3] + a4*F[4] + a5*F[5])*(a3*F[3] + a4*F[4] + a5*F[5])) + 2*((a3*F[6] + a4*F[7] + a5*F[8])*(a3*F[6] + a4*F[7] + a5*F[8])) + 2*(((1.0/2.0)*F[3]*F[6] + (1.0/2.0)*F[4]*F[7] + (1.0/2.0)*F[5]*F[8])*((1.0/2.0)*F[3]*F[6] + (1.0/2.0)*F[4]*F[7] + (1.0/2.0)*F[5]*F[8])));
    return psi;
}

template <class Derived>
Vector<9>
SaintVenantKirchhoffEnergy<3>::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Vector<9> G;
    auto vecG = G.reshaped();
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1])) + (1.0/2.0)*((F[2])*(F[2]));
    Scalar const a1 = (1.0/2.0)*((F[3])*(F[3])) + (1.0/2.0)*((F[4])*(F[4])) + (1.0/2.0)*((F[5])*(F[5]));
    Scalar const a2 = (1.0/2.0)*((F[6])*(F[6])) + (1.0/2.0)*((F[7])*(F[7])) + (1.0/2.0)*((F[8])*(F[8]));
    Scalar const a3 = lambda*(a0 + a1 + a2 - 3.0/2.0);
    Scalar const a4 = 2*a0 - 1;
    Scalar const a5 = (1.0/2.0)*F[0];
    Scalar const a6 = (1.0/2.0)*F[1];
    Scalar const a7 = (1.0/2.0)*F[2];
    Scalar const a8 = 2*a5*F[3] + 2*a6*F[4] + 2*a7*F[5];
    Scalar const a9 = 2*a5*F[6] + 2*a6*F[7] + 2*a7*F[8];
    Scalar const a10 = 2*a1 - 1;
    Scalar const a11 = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];
    Scalar const a12 = 2*a2 - 1;
    vecG[0] = a3*F[0] + mu*(a4*F[0] + a8*F[3] + a9*F[6]);
    vecG[1] = a3*F[1] + mu*(a4*F[1] + a8*F[4] + a9*F[7]);
    vecG[2] = a3*F[2] + mu*(a4*F[2] + a8*F[5] + a9*F[8]);
    vecG[3] = a3*F[3] + mu*(a10*F[3] + a11*F[6] + a8*F[0]);
    vecG[4] = a3*F[4] + mu*(a10*F[4] + a11*F[7] + a8*F[1]);
    vecG[5] = a3*F[5] + mu*(a10*F[5] + a11*F[8] + a8*F[2]);
    vecG[6] = a3*F[6] + mu*(a11*F[3] + a12*F[6] + a9*F[0]);
    vecG[7] = a3*F[7] + mu*(a11*F[4] + a12*F[7] + a9*F[1]);
    vecG[8] = a3*F[8] + mu*(a11*F[5] + a12*F[8] + a9*F[2]);
    return G;
}

template <class Derived>
Matrix<9,9>
SaintVenantKirchhoffEnergy<3>::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Matrix<9,9> H;
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = ((F[3])*(F[3]));
    Scalar const a3 = a1 + a2;
    Scalar const a4 = ((F[6])*(F[6]));
    Scalar const a5 = ((F[2])*(F[2]));
    Scalar const a6 = a5 - 1;
    Scalar const a7 = a4 + a6;
    Scalar const a8 = ((F[4])*(F[4]));
    Scalar const a9 = ((F[5])*(F[5]));
    Scalar const a10 = ((F[7])*(F[7]));
    Scalar const a11 = ((F[8])*(F[8]));
    Scalar const a12 = lambda*((1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a10 + (1.0/2.0)*a11 + (1.0/2.0)*a2 + (1.0/2.0)*a4 + (1.0/2.0)*a5 + (1.0/2.0)*a8 + (1.0/2.0)*a9 - 3.0/2.0);
    Scalar const a13 = F[0]*F[1];
    Scalar const a14 = F[3]*F[4];
    Scalar const a15 = F[6]*F[7];
    Scalar const a16 = a13*lambda + mu*(2*a13 + a14 + a15);
    Scalar const a17 = F[0]*F[2];
    Scalar const a18 = F[3]*F[5];
    Scalar const a19 = F[6]*F[8];
    Scalar const a20 = a17*lambda + mu*(2*a17 + a18 + a19);
    Scalar const a21 = F[0]*F[3];
    Scalar const a22 = F[1]*F[4];
    Scalar const a23 = F[2]*F[5];
    Scalar const a24 = a21*lambda + mu*(2*a21 + a22 + a23);
    Scalar const a25 = lambda*F[0];
    Scalar const a26 = mu*F[3];
    Scalar const a27 = a25*F[4] + a26*F[1];
    Scalar const a28 = a25*F[5] + a26*F[2];
    Scalar const a29 = F[0]*F[6];
    Scalar const a30 = F[1]*F[7];
    Scalar const a31 = F[2]*F[8];
    Scalar const a32 = a29*lambda + mu*(2*a29 + a30 + a31);
    Scalar const a33 = mu*F[6];
    Scalar const a34 = a25*F[7] + a33*F[1];
    Scalar const a35 = a25*F[8] + a33*F[2];
    Scalar const a36 = a0 + a8;
    Scalar const a37 = F[1]*F[2];
    Scalar const a38 = F[4]*F[5];
    Scalar const a39 = F[7]*F[8];
    Scalar const a40 = a37*lambda + mu*(2*a37 + a38 + a39);
    Scalar const a41 = lambda*F[1];
    Scalar const a42 = mu*F[4];
    Scalar const a43 = a41*F[3] + a42*F[0];
    Scalar const a44 = a22*lambda + mu*(a21 + 2*a22 + a23);
    Scalar const a45 = a41*F[5] + a42*F[2];
    Scalar const a46 = mu*F[7];
    Scalar const a47 = a41*F[6] + a46*F[0];
    Scalar const a48 = a30*lambda + mu*(a29 + 2*a30 + a31);
    Scalar const a49 = a41*F[8] + a46*F[2];
    Scalar const a50 = a9 - 1;
    Scalar const a51 = a0 + a11;
    Scalar const a52 = lambda*F[2];
    Scalar const a53 = mu*F[5];
    Scalar const a54 = a52*F[3] + a53*F[0];
    Scalar const a55 = a52*F[4] + a53*F[1];
    Scalar const a56 = a23*lambda + mu*(a21 + a22 + 2*a23);
    Scalar const a57 = mu*F[8];
    Scalar const a58 = a52*F[6] + a57*F[0];
    Scalar const a59 = a52*F[7] + a57*F[1];
    Scalar const a60 = a31*lambda + mu*(a29 + a30 + 2*a31);
    Scalar const a61 = a14*lambda + mu*(a13 + 2*a14 + a15);
    Scalar const a62 = a18*lambda + mu*(a17 + 2*a18 + a19);
    Scalar const a63 = F[3]*F[6];
    Scalar const a64 = F[4]*F[7];
    Scalar const a65 = F[5]*F[8];
    Scalar const a66 = a63*lambda + mu*(2*a63 + a64 + a65);
    Scalar const a67 = lambda*F[3];
    Scalar const a68 = a33*F[4] + a67*F[7];
    Scalar const a69 = a33*F[5] + a67*F[8];
    Scalar const a70 = a38*lambda + mu*(a37 + 2*a38 + a39);
    Scalar const a71 = lambda*F[4];
    Scalar const a72 = a26*F[7] + a71*F[6];
    Scalar const a73 = a64*lambda + mu*(a63 + 2*a64 + a65);
    Scalar const a74 = a46*F[5] + a71*F[8];
    Scalar const a75 = a11 + a8;
    Scalar const a76 = lambda*F[5];
    Scalar const a77 = a26*F[8] + a76*F[6];
    Scalar const a78 = a42*F[8] + a76*F[7];
    Scalar const a79 = a65*lambda + mu*(a63 + a64 + 2*a65);
    Scalar const a80 = a15*lambda + mu*(a13 + a14 + 2*a15);
    Scalar const a81 = a19*lambda + mu*(a17 + a18 + 2*a19);
    Scalar const a82 = a39*lambda + mu*(a37 + a38 + 2*a39);
    vecH[0] = a0*lambda + a12 + mu*(3*a0 + a3 + a7);
    vecH[1] = a16;
    vecH[2] = a20;
    vecH[3] = a24;
    vecH[4] = a27;
    vecH[5] = a28;
    vecH[6] = a32;
    vecH[7] = a34;
    vecH[8] = a35;
    vecH[9] = a16;
    vecH[10] = a1*lambda + a12 + mu*(3*a1 + a10 + a36 + a6);
    vecH[11] = a40;
    vecH[12] = a43;
    vecH[13] = a44;
    vecH[14] = a45;
    vecH[15] = a47;
    vecH[16] = a48;
    vecH[17] = a49;
    vecH[18] = a20;
    vecH[19] = a40;
    vecH[20] = a12 + a5*lambda + mu*(a1 + 3*a5 + a50 + a51);
    vecH[21] = a54;
    vecH[22] = a55;
    vecH[23] = a56;
    vecH[24] = a58;
    vecH[25] = a59;
    vecH[26] = a60;
    vecH[27] = a24;
    vecH[28] = a43;
    vecH[29] = a54;
    vecH[30] = a12 + a2*lambda + mu*(3*a2 + a36 + a4 + a50);
    vecH[31] = a61;
    vecH[32] = a62;
    vecH[33] = a66;
    vecH[34] = a68;
    vecH[35] = a69;
    vecH[36] = a27;
    vecH[37] = a44;
    vecH[38] = a55;
    vecH[39] = a61;
    vecH[40] = a12 + a8*lambda + mu*(a10 + a3 + a50 + 3*a8);
    vecH[41] = a70;
    vecH[42] = a72;
    vecH[43] = a73;
    vecH[44] = a74;
    vecH[45] = a28;
    vecH[46] = a45;
    vecH[47] = a56;
    vecH[48] = a62;
    vecH[49] = a70;
    vecH[50] = a12 + a9*lambda + mu*(a2 + a6 + a75 + 3*a9);
    vecH[51] = a77;
    vecH[52] = a78;
    vecH[53] = a79;
    vecH[54] = a32;
    vecH[55] = a47;
    vecH[56] = a58;
    vecH[57] = a66;
    vecH[58] = a72;
    vecH[59] = a77;
    vecH[60] = a12 + a4*lambda + mu*(a10 + a2 + 3*a4 + a51 - 1);
    vecH[61] = a80;
    vecH[62] = a81;
    vecH[63] = a34;
    vecH[64] = a48;
    vecH[65] = a59;
    vecH[66] = a68;
    vecH[67] = a73;
    vecH[68] = a78;
    vecH[69] = a80;
    vecH[70] = a10*lambda + a12 + mu*(a1 + 3*a10 + a4 + a75 - 1);
    vecH[71] = a82;
    vecH[72] = a35;
    vecH[73] = a49;
    vecH[74] = a60;
    vecH[75] = a69;
    vecH[76] = a74;
    vecH[77] = a79;
    vecH[78] = a81;
    vecH[79] = a82;
    vecH[80] = a11*lambda + a12 + mu*(a10 + 3*a11 + a7 + a9);
    return H;
}

template <class Derived>
std::tuple<Scalar, Vector<9>>
SaintVenantKirchhoffEnergy<3>::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    auto vecG = G.reshaped();
    Scalar const a0 = (1.0/2.0)*((F[0])*(F[0])) + (1.0/2.0)*((F[1])*(F[1])) + (1.0/2.0)*((F[2])*(F[2]));
    Scalar const a1 = (1.0/2.0)*((F[3])*(F[3])) + (1.0/2.0)*((F[4])*(F[4])) + (1.0/2.0)*((F[5])*(F[5]));
    Scalar const a2 = (1.0/2.0)*((F[6])*(F[6])) + (1.0/2.0)*((F[7])*(F[7])) + (1.0/2.0)*((F[8])*(F[8]));
    Scalar const a3 = a0 + a1 + a2 - 3.0/2.0;
    Scalar const a4 = a0 - 1.0/2.0;
    Scalar const a5 = a1 - 1.0/2.0;
    Scalar const a6 = a2 - 1.0/2.0;
    Scalar const a7 = (1.0/2.0)*F[0];
    Scalar const a8 = (1.0/2.0)*F[1];
    Scalar const a9 = (1.0/2.0)*F[2];
    Scalar const a10 = a7*F[3] + a8*F[4] + a9*F[5];
    Scalar const a11 = a7*F[6] + a8*F[7] + a9*F[8];
    Scalar const a12 = (1.0/2.0)*F[3]*F[6] + (1.0/2.0)*F[4]*F[7] + (1.0/2.0)*F[5]*F[8];
    Scalar const a13 = a3*lambda;
    Scalar const a14 = 2*a4;
    Scalar const a15 = 2*a10;
    Scalar const a16 = 2*a11;
    Scalar const a17 = 2*a5;
    Scalar const a18 = 2*a12;
    Scalar const a19 = 2*a6;
    psi = (1.0/2.0)*((a3)*(a3))*lambda + mu*(2*((a10)*(a10)) + 2*((a11)*(a11)) + 2*((a12)*(a12)) + ((a4)*(a4)) + ((a5)*(a5)) + ((a6)*(a6)));
    vecG[0] = a13*F[0] + mu*(a14*F[0] + a15*F[3] + a16*F[6]);
    vecG[1] = a13*F[1] + mu*(a14*F[1] + a15*F[4] + a16*F[7]);
    vecG[2] = a13*F[2] + mu*(a14*F[2] + a15*F[5] + a16*F[8]);
    vecG[3] = a13*F[3] + mu*(a15*F[0] + a17*F[3] + a18*F[6]);
    vecG[4] = a13*F[4] + mu*(a15*F[1] + a17*F[4] + a18*F[7]);
    vecG[5] = a13*F[5] + mu*(a15*F[2] + a17*F[5] + a18*F[8]);
    vecG[6] = a13*F[6] + mu*(a16*F[0] + a18*F[3] + a19*F[6]);
    vecG[7] = a13*F[7] + mu*(a16*F[1] + a18*F[4] + a19*F[7]);
    vecG[8] = a13*F[8] + mu*(a16*F[2] + a18*F[5] + a19*F[8]);
    return {psi, G};
}

template <class Derived>
std::tuple<Scalar, Vector<9>, Matrix<9,9>>
SaintVenantKirchhoffEnergy<3>::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    Matrix<9,9> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = ((F[2])*(F[2]));
    Scalar const a3 = (1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a2;
    Scalar const a4 = ((F[3])*(F[3]));
    Scalar const a5 = ((F[4])*(F[4]));
    Scalar const a6 = ((F[5])*(F[5]));
    Scalar const a7 = (1.0/2.0)*a4 + (1.0/2.0)*a5 + (1.0/2.0)*a6;
    Scalar const a8 = ((F[6])*(F[6]));
    Scalar const a9 = ((F[7])*(F[7]));
    Scalar const a10 = ((F[8])*(F[8]));
    Scalar const a11 = (1.0/2.0)*a10 + (1.0/2.0)*a8 + (1.0/2.0)*a9;
    Scalar const a12 = a11 + a3 + a7 - 3.0/2.0;
    Scalar const a13 = a3 - 1.0/2.0;
    Scalar const a14 = a7 - 1.0/2.0;
    Scalar const a15 = a11 - 1.0/2.0;
    Scalar const a16 = F[0]*F[3];
    Scalar const a17 = F[1]*F[4];
    Scalar const a18 = F[2]*F[5];
    Scalar const a19 = (1.0/2.0)*a16 + (1.0/2.0)*a17 + (1.0/2.0)*a18;
    Scalar const a20 = F[0]*F[6];
    Scalar const a21 = F[1]*F[7];
    Scalar const a22 = F[2]*F[8];
    Scalar const a23 = (1.0/2.0)*a20 + (1.0/2.0)*a21 + (1.0/2.0)*a22;
    Scalar const a24 = F[3]*F[6];
    Scalar const a25 = F[4]*F[7];
    Scalar const a26 = F[5]*F[8];
    Scalar const a27 = (1.0/2.0)*a24 + (1.0/2.0)*a25 + (1.0/2.0)*a26;
    Scalar const a28 = a12*lambda;
    Scalar const a29 = 2*a13;
    Scalar const a30 = 2*a19;
    Scalar const a31 = 2*a23;
    Scalar const a32 = 2*a14;
    Scalar const a33 = 2*a27;
    Scalar const a34 = 2*a15;
    Scalar const a35 = a1 + a4;
    Scalar const a36 = a2 - 1;
    Scalar const a37 = a36 + a8;
    Scalar const a38 = F[0]*F[1];
    Scalar const a39 = F[3]*F[4];
    Scalar const a40 = F[6]*F[7];
    Scalar const a41 = a38*lambda + mu*(2*a38 + a39 + a40);
    Scalar const a42 = F[0]*F[2];
    Scalar const a43 = F[3]*F[5];
    Scalar const a44 = F[6]*F[8];
    Scalar const a45 = a42*lambda + mu*(2*a42 + a43 + a44);
    Scalar const a46 = a16*lambda + mu*(2*a16 + a17 + a18);
    Scalar const a47 = lambda*F[0];
    Scalar const a48 = mu*F[3];
    Scalar const a49 = a47*F[4] + a48*F[1];
    Scalar const a50 = a47*F[5] + a48*F[2];
    Scalar const a51 = a20*lambda + mu*(2*a20 + a21 + a22);
    Scalar const a52 = mu*F[6];
    Scalar const a53 = a47*F[7] + a52*F[1];
    Scalar const a54 = a47*F[8] + a52*F[2];
    Scalar const a55 = a0 + a5;
    Scalar const a56 = F[1]*F[2];
    Scalar const a57 = F[4]*F[5];
    Scalar const a58 = F[7]*F[8];
    Scalar const a59 = a56*lambda + mu*(2*a56 + a57 + a58);
    Scalar const a60 = lambda*F[1];
    Scalar const a61 = mu*F[4];
    Scalar const a62 = a60*F[3] + a61*F[0];
    Scalar const a63 = a17*lambda + mu*(a16 + 2*a17 + a18);
    Scalar const a64 = a60*F[5] + a61*F[2];
    Scalar const a65 = mu*F[7];
    Scalar const a66 = a60*F[6] + a65*F[0];
    Scalar const a67 = a21*lambda + mu*(a20 + 2*a21 + a22);
    Scalar const a68 = a60*F[8] + a65*F[2];
    Scalar const a69 = a6 - 1;
    Scalar const a70 = a0 + a10;
    Scalar const a71 = lambda*F[2];
    Scalar const a72 = mu*F[5];
    Scalar const a73 = a71*F[3] + a72*F[0];
    Scalar const a74 = a71*F[4] + a72*F[1];
    Scalar const a75 = a18*lambda + mu*(a16 + a17 + 2*a18);
    Scalar const a76 = mu*F[8];
    Scalar const a77 = a71*F[6] + a76*F[0];
    Scalar const a78 = a71*F[7] + a76*F[1];
    Scalar const a79 = a22*lambda + mu*(a20 + a21 + 2*a22);
    Scalar const a80 = a39*lambda + mu*(a38 + 2*a39 + a40);
    Scalar const a81 = a43*lambda + mu*(a42 + 2*a43 + a44);
    Scalar const a82 = a24*lambda + mu*(2*a24 + a25 + a26);
    Scalar const a83 = lambda*F[3];
    Scalar const a84 = a52*F[4] + a83*F[7];
    Scalar const a85 = a52*F[5] + a83*F[8];
    Scalar const a86 = a57*lambda + mu*(a56 + 2*a57 + a58);
    Scalar const a87 = lambda*F[4];
    Scalar const a88 = a48*F[7] + a87*F[6];
    Scalar const a89 = a25*lambda + mu*(a24 + 2*a25 + a26);
    Scalar const a90 = a65*F[5] + a87*F[8];
    Scalar const a91 = a10 + a5;
    Scalar const a92 = lambda*F[5];
    Scalar const a93 = a48*F[8] + a92*F[6];
    Scalar const a94 = a61*F[8] + a92*F[7];
    Scalar const a95 = a26*lambda + mu*(a24 + a25 + 2*a26);
    Scalar const a96 = a40*lambda + mu*(a38 + a39 + 2*a40);
    Scalar const a97 = a44*lambda + mu*(a42 + a43 + 2*a44);
    Scalar const a98 = a58*lambda + mu*(a56 + a57 + 2*a58);
    psi = (1.0/2.0)*((a12)*(a12))*lambda + mu*(((a13)*(a13)) + ((a14)*(a14)) + ((a15)*(a15)) + 2*((a19)*(a19)) + 2*((a23)*(a23)) + 2*((a27)*(a27)));
    vecG[0] = a28*F[0] + mu*(a29*F[0] + a30*F[3] + a31*F[6]);
    vecG[1] = a28*F[1] + mu*(a29*F[1] + a30*F[4] + a31*F[7]);
    vecG[2] = a28*F[2] + mu*(a29*F[2] + a30*F[5] + a31*F[8]);
    vecG[3] = a28*F[3] + mu*(a30*F[0] + a32*F[3] + a33*F[6]);
    vecG[4] = a28*F[4] + mu*(a30*F[1] + a32*F[4] + a33*F[7]);
    vecG[5] = a28*F[5] + mu*(a30*F[2] + a32*F[5] + a33*F[8]);
    vecG[6] = a28*F[6] + mu*(a31*F[0] + a33*F[3] + a34*F[6]);
    vecG[7] = a28*F[7] + mu*(a31*F[1] + a33*F[4] + a34*F[7]);
    vecG[8] = a28*F[8] + mu*(a31*F[2] + a33*F[5] + a34*F[8]);
    vecH[0] = a0*lambda + a28 + mu*(3*a0 + a35 + a37);
    vecH[1] = a41;
    vecH[2] = a45;
    vecH[3] = a46;
    vecH[4] = a49;
    vecH[5] = a50;
    vecH[6] = a51;
    vecH[7] = a53;
    vecH[8] = a54;
    vecH[9] = a41;
    vecH[10] = a1*lambda + a28 + mu*(3*a1 + a36 + a55 + a9);
    vecH[11] = a59;
    vecH[12] = a62;
    vecH[13] = a63;
    vecH[14] = a64;
    vecH[15] = a66;
    vecH[16] = a67;
    vecH[17] = a68;
    vecH[18] = a45;
    vecH[19] = a59;
    vecH[20] = a2*lambda + a28 + mu*(a1 + 3*a2 + a69 + a70);
    vecH[21] = a73;
    vecH[22] = a74;
    vecH[23] = a75;
    vecH[24] = a77;
    vecH[25] = a78;
    vecH[26] = a79;
    vecH[27] = a46;
    vecH[28] = a62;
    vecH[29] = a73;
    vecH[30] = a28 + a4*lambda + mu*(3*a4 + a55 + a69 + a8);
    vecH[31] = a80;
    vecH[32] = a81;
    vecH[33] = a82;
    vecH[34] = a84;
    vecH[35] = a85;
    vecH[36] = a49;
    vecH[37] = a63;
    vecH[38] = a74;
    vecH[39] = a80;
    vecH[40] = a28 + a5*lambda + mu*(a35 + 3*a5 + a69 + a9);
    vecH[41] = a86;
    vecH[42] = a88;
    vecH[43] = a89;
    vecH[44] = a90;
    vecH[45] = a50;
    vecH[46] = a64;
    vecH[47] = a75;
    vecH[48] = a81;
    vecH[49] = a86;
    vecH[50] = a28 + a6*lambda + mu*(a36 + a4 + 3*a6 + a91);
    vecH[51] = a93;
    vecH[52] = a94;
    vecH[53] = a95;
    vecH[54] = a51;
    vecH[55] = a66;
    vecH[56] = a77;
    vecH[57] = a82;
    vecH[58] = a88;
    vecH[59] = a93;
    vecH[60] = a28 + a8*lambda + mu*(a4 + a70 + 3*a8 + a9 - 1);
    vecH[61] = a96;
    vecH[62] = a97;
    vecH[63] = a53;
    vecH[64] = a67;
    vecH[65] = a78;
    vecH[66] = a84;
    vecH[67] = a89;
    vecH[68] = a94;
    vecH[69] = a96;
    vecH[70] = a28 + a9*lambda + mu*(a1 + a8 + 3*a9 + a91 - 1);
    vecH[71] = a98;
    vecH[72] = a54;
    vecH[73] = a68;
    vecH[74] = a79;
    vecH[75] = a85;
    vecH[76] = a90;
    vecH[77] = a95;
    vecH[78] = a97;
    vecH[79] = a98;
    vecH[80] = a10*lambda + a28 + mu*(3*a10 + a37 + a6 + a9);
    return {psi, G, H};
}

template <class Derived>
std::tuple<Vector<9>, Matrix<9,9>>
SaintVenantKirchhoffEnergy<3>::gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
{
    Vector<9> G;
    Matrix<9,9> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = ((F[2])*(F[2]));
    Scalar const a3 = (1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a2;
    Scalar const a4 = ((F[3])*(F[3]));
    Scalar const a5 = ((F[4])*(F[4]));
    Scalar const a6 = ((F[5])*(F[5]));
    Scalar const a7 = (1.0/2.0)*a4 + (1.0/2.0)*a5 + (1.0/2.0)*a6;
    Scalar const a8 = ((F[6])*(F[6]));
    Scalar const a9 = ((F[7])*(F[7]));
    Scalar const a10 = ((F[8])*(F[8]));
    Scalar const a11 = (1.0/2.0)*a10 + (1.0/2.0)*a8 + (1.0/2.0)*a9;
    Scalar const a12 = lambda*(a11 + a3 + a7 - 3.0/2.0);
    Scalar const a13 = 2*a3 - 1;
    Scalar const a14 = F[0]*F[3];
    Scalar const a15 = F[1]*F[4];
    Scalar const a16 = F[2]*F[5];
    Scalar const a17 = a14 + a15 + a16;
    Scalar const a18 = F[0]*F[6];
    Scalar const a19 = F[1]*F[7];
    Scalar const a20 = F[2]*F[8];
    Scalar const a21 = a18 + a19 + a20;
    Scalar const a22 = 2*a7 - 1;
    Scalar const a23 = F[3]*F[6];
    Scalar const a24 = F[4]*F[7];
    Scalar const a25 = F[5]*F[8];
    Scalar const a26 = a23 + a24 + a25;
    Scalar const a27 = 2*a11 - 1;
    Scalar const a28 = a1 + a4;
    Scalar const a29 = a2 - 1;
    Scalar const a30 = a29 + a8;
    Scalar const a31 = F[0]*F[1];
    Scalar const a32 = F[3]*F[4];
    Scalar const a33 = F[6]*F[7];
    Scalar const a34 = a31*lambda + mu*(2*a31 + a32 + a33);
    Scalar const a35 = F[0]*F[2];
    Scalar const a36 = F[3]*F[5];
    Scalar const a37 = F[6]*F[8];
    Scalar const a38 = a35*lambda + mu*(2*a35 + a36 + a37);
    Scalar const a39 = a14*lambda + mu*(2*a14 + a15 + a16);
    Scalar const a40 = lambda*F[0];
    Scalar const a41 = mu*F[3];
    Scalar const a42 = a40*F[4] + a41*F[1];
    Scalar const a43 = a40*F[5] + a41*F[2];
    Scalar const a44 = a18*lambda + mu*(2*a18 + a19 + a20);
    Scalar const a45 = mu*F[6];
    Scalar const a46 = a40*F[7] + a45*F[1];
    Scalar const a47 = a40*F[8] + a45*F[2];
    Scalar const a48 = a0 + a5;
    Scalar const a49 = F[1]*F[2];
    Scalar const a50 = F[4]*F[5];
    Scalar const a51 = F[7]*F[8];
    Scalar const a52 = a49*lambda + mu*(2*a49 + a50 + a51);
    Scalar const a53 = lambda*F[1];
    Scalar const a54 = mu*F[4];
    Scalar const a55 = a53*F[3] + a54*F[0];
    Scalar const a56 = a15*lambda + mu*(a14 + 2*a15 + a16);
    Scalar const a57 = a53*F[5] + a54*F[2];
    Scalar const a58 = mu*F[7];
    Scalar const a59 = a53*F[6] + a58*F[0];
    Scalar const a60 = a19*lambda + mu*(a18 + 2*a19 + a20);
    Scalar const a61 = a53*F[8] + a58*F[2];
    Scalar const a62 = a6 - 1;
    Scalar const a63 = a0 + a10;
    Scalar const a64 = lambda*F[2];
    Scalar const a65 = mu*F[5];
    Scalar const a66 = a64*F[3] + a65*F[0];
    Scalar const a67 = a64*F[4] + a65*F[1];
    Scalar const a68 = a16*lambda + mu*(a14 + a15 + 2*a16);
    Scalar const a69 = mu*F[8];
    Scalar const a70 = a64*F[6] + a69*F[0];
    Scalar const a71 = a64*F[7] + a69*F[1];
    Scalar const a72 = a20*lambda + mu*(a18 + a19 + 2*a20);
    Scalar const a73 = a32*lambda + mu*(a31 + 2*a32 + a33);
    Scalar const a74 = a36*lambda + mu*(a35 + 2*a36 + a37);
    Scalar const a75 = a23*lambda + mu*(2*a23 + a24 + a25);
    Scalar const a76 = lambda*F[3];
    Scalar const a77 = a45*F[4] + a76*F[7];
    Scalar const a78 = a45*F[5] + a76*F[8];
    Scalar const a79 = a50*lambda + mu*(a49 + 2*a50 + a51);
    Scalar const a80 = lambda*F[4];
    Scalar const a81 = a41*F[7] + a80*F[6];
    Scalar const a82 = a24*lambda + mu*(a23 + 2*a24 + a25);
    Scalar const a83 = a58*F[5] + a80*F[8];
    Scalar const a84 = a10 + a5;
    Scalar const a85 = lambda*F[5];
    Scalar const a86 = a41*F[8] + a85*F[6];
    Scalar const a87 = a54*F[8] + a85*F[7];
    Scalar const a88 = a25*lambda + mu*(a23 + a24 + 2*a25);
    Scalar const a89 = a33*lambda + mu*(a31 + a32 + 2*a33);
    Scalar const a90 = a37*lambda + mu*(a35 + a36 + 2*a37);
    Scalar const a91 = a51*lambda + mu*(a49 + a50 + 2*a51);
    vecG[0] = a12*F[0] + mu*(a13*F[0] + a17*F[3] + a21*F[6]);
    vecG[1] = a12*F[1] + mu*(a13*F[1] + a17*F[4] + a21*F[7]);
    vecG[2] = a12*F[2] + mu*(a13*F[2] + a17*F[5] + a21*F[8]);
    vecG[3] = a12*F[3] + mu*(a17*F[0] + a22*F[3] + a26*F[6]);
    vecG[4] = a12*F[4] + mu*(a17*F[1] + a22*F[4] + a26*F[7]);
    vecG[5] = a12*F[5] + mu*(a17*F[2] + a22*F[5] + a26*F[8]);
    vecG[6] = a12*F[6] + mu*(a21*F[0] + a26*F[3] + a27*F[6]);
    vecG[7] = a12*F[7] + mu*(a21*F[1] + a26*F[4] + a27*F[7]);
    vecG[8] = a12*F[8] + mu*(a21*F[2] + a26*F[5] + a27*F[8]);
    vecH[0] = a0*lambda + a12 + mu*(3*a0 + a28 + a30);
    vecH[1] = a34;
    vecH[2] = a38;
    vecH[3] = a39;
    vecH[4] = a42;
    vecH[5] = a43;
    vecH[6] = a44;
    vecH[7] = a46;
    vecH[8] = a47;
    vecH[9] = a34;
    vecH[10] = a1*lambda + a12 + mu*(3*a1 + a29 + a48 + a9);
    vecH[11] = a52;
    vecH[12] = a55;
    vecH[13] = a56;
    vecH[14] = a57;
    vecH[15] = a59;
    vecH[16] = a60;
    vecH[17] = a61;
    vecH[18] = a38;
    vecH[19] = a52;
    vecH[20] = a12 + a2*lambda + mu*(a1 + 3*a2 + a62 + a63);
    vecH[21] = a66;
    vecH[22] = a67;
    vecH[23] = a68;
    vecH[24] = a70;
    vecH[25] = a71;
    vecH[26] = a72;
    vecH[27] = a39;
    vecH[28] = a55;
    vecH[29] = a66;
    vecH[30] = a12 + a4*lambda + mu*(3*a4 + a48 + a62 + a8);
    vecH[31] = a73;
    vecH[32] = a74;
    vecH[33] = a75;
    vecH[34] = a77;
    vecH[35] = a78;
    vecH[36] = a42;
    vecH[37] = a56;
    vecH[38] = a67;
    vecH[39] = a73;
    vecH[40] = a12 + a5*lambda + mu*(a28 + 3*a5 + a62 + a9);
    vecH[41] = a79;
    vecH[42] = a81;
    vecH[43] = a82;
    vecH[44] = a83;
    vecH[45] = a43;
    vecH[46] = a57;
    vecH[47] = a68;
    vecH[48] = a74;
    vecH[49] = a79;
    vecH[50] = a12 + a6*lambda + mu*(a29 + a4 + 3*a6 + a84);
    vecH[51] = a86;
    vecH[52] = a87;
    vecH[53] = a88;
    vecH[54] = a44;
    vecH[55] = a59;
    vecH[56] = a70;
    vecH[57] = a75;
    vecH[58] = a81;
    vecH[59] = a86;
    vecH[60] = a12 + a8*lambda + mu*(a4 + a63 + 3*a8 + a9 - 1);
    vecH[61] = a89;
    vecH[62] = a90;
    vecH[63] = a46;
    vecH[64] = a60;
    vecH[65] = a71;
    vecH[66] = a77;
    vecH[67] = a82;
    vecH[68] = a87;
    vecH[69] = a89;
    vecH[70] = a12 + a9*lambda + mu*(a1 + a8 + a84 + 3*a9 - 1);
    vecH[71] = a91;
    vecH[72] = a47;
    vecH[73] = a61;
    vecH[74] = a72;
    vecH[75] = a78;
    vecH[76] = a83;
    vecH[77] = a88;
    vecH[78] = a90;
    vecH[79] = a91;
    vecH[80] = a10*lambda + a12 + mu*(3*a10 + a30 + a6 + a9);
    return {G, H};
}

} // physics
} // pba

#endif // PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
