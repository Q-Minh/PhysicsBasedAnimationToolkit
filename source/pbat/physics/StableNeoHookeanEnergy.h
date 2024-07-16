
#ifndef PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
#define PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H

#include <pbat/Aliases.h>

#include <cmath>
#include <tuple>

namespace pbat {
namespace physics {

template <int Dims>
struct StableNeoHookeanEnergy;

template <>
struct StableNeoHookeanEnergy<1>
{
    public:
        static auto constexpr kDims = 1;
    
        template <class TDerived>
        Scalar
        eval(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Vector<1>
        grad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Matrix<1,1>
        hessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<1>>
        evalWithGrad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<1>, Matrix<1,1>>
        evalWithGradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Vector<1>, Matrix<1,1>>
        gradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;
};

template <class TDerived>
Scalar
StableNeoHookeanEnergy<1>::eval(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    psi = (1.0/2.0)*lambda*((F[0] - 1 - mu/lambda)*(F[0] - 1 - mu/lambda)) + (1.0/2.0)*mu*(((F[0])*(F[0])) - 1);
    return psi;
}

template <class TDerived>
Vector<1>
StableNeoHookeanEnergy<1>::grad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Vector<1> G;
    auto vecG = G.reshaped();
    vecG[0] = (1.0/2.0)*lambda*(2*F[0] - 2 - 2*mu/lambda) + mu*F[0];
    return G;
}

template <class TDerived>
Matrix<1,1>
StableNeoHookeanEnergy<1>::hessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Matrix<1,1> H;
    auto vecH = H.reshaped();
    vecH[0] = lambda + mu;
    return H;
}

template <class TDerived>
std::tuple<Scalar, Vector<1>>
StableNeoHookeanEnergy<1>::evalWithGrad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<1> G;
    auto vecG = G.reshaped();
    Scalar const a0 = mu/lambda;
    Scalar const a1 = (1.0/2.0)*lambda;
    psi = a1*((-a0 + F[0] - 1)*(-a0 + F[0] - 1)) + (1.0/2.0)*mu*(((F[0])*(F[0])) - 1);
    vecG[0] = a1*(-2*a0 + 2*F[0] - 2) + mu*F[0];
    return {psi, G};
}

template <class TDerived>
std::tuple<Scalar, Vector<1>, Matrix<1,1>>
StableNeoHookeanEnergy<1>::evalWithGradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<1> G;
    Matrix<1,1> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = mu/lambda;
    Scalar const a1 = (1.0/2.0)*lambda;
    psi = a1*((-a0 + F[0] - 1)*(-a0 + F[0] - 1)) + (1.0/2.0)*mu*(((F[0])*(F[0])) - 1);
    vecG[0] = a1*(-2*a0 + 2*F[0] - 2) + mu*F[0];
    vecH[0] = lambda + mu;
    return {psi, G, H};
}

template <class TDerived>
std::tuple<Vector<1>, Matrix<1,1>>
StableNeoHookeanEnergy<1>::gradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F, 
    [[maybe_unused]] Scalar mu, 
    [[maybe_unused]] Scalar lambda) const
{
    Vector<1> G;
    Matrix<1,1> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    vecG[0] = (1.0/2.0)*lambda*(2*F[0] - 2 - 2*mu/lambda) + mu*F[0];
    vecH[0] = lambda + mu;
    return {G, H};
}

template <>
struct StableNeoHookeanEnergy<2>
{
    public:
        static auto constexpr kDims = 2;
    
        template <class TDerived>
        Scalar
        eval(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Vector<4>
        grad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Matrix<4,4>
        hessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<4>>
        evalWithGrad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<4>, Matrix<4,4>>
        evalWithGradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Vector<4>, Matrix<4,4>>
        gradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;
};

template <class TDerived>
Scalar
StableNeoHookeanEnergy<2>::eval(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    psi = (1.0/2.0)*lambda*((F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda)*(F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda)) + (1.0/2.0)*mu*(((F[0])*(F[0])) + ((F[1])*(F[1])) + ((F[2])*(F[2])) + ((F[3])*(F[3])) - 2);
    return psi;
}

template <class TDerived>
Vector<4>
StableNeoHookeanEnergy<2>::grad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Vector<4> G;
    auto vecG = G.reshaped();
    Scalar const a0 = lambda*(F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda);
    vecG[0] = a0*F[3] + mu*F[0];
    vecG[1] = -a0*F[2] + mu*F[1];
    vecG[2] = -a0*F[1] + mu*F[2];
    vecG[3] = a0*F[0] + mu*F[3];
    return G;
}

template <class TDerived>
Matrix<4,4>
StableNeoHookeanEnergy<2>::hessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Matrix<4,4> H;
    auto vecH = H.reshaped();
    Scalar const a0 = lambda*F[3];
    Scalar const a1 = -a0*F[2];
    Scalar const a2 = -a0*F[1];
    Scalar const a3 = lambda*(F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda);
    Scalar const a4 = a3 + lambda*F[0]*F[3];
    Scalar const a5 = -a3 + lambda*F[1]*F[2];
    Scalar const a6 = lambda*F[0];
    Scalar const a7 = -a6*F[2];
    Scalar const a8 = -a6*F[1];
    vecH[0] = lambda*((F[3])*(F[3])) + mu;
    vecH[1] = a1;
    vecH[2] = a2;
    vecH[3] = a4;
    vecH[4] = a1;
    vecH[5] = lambda*((F[2])*(F[2])) + mu;
    vecH[6] = a5;
    vecH[7] = a7;
    vecH[8] = a2;
    vecH[9] = a5;
    vecH[10] = lambda*((F[1])*(F[1])) + mu;
    vecH[11] = a8;
    vecH[12] = a4;
    vecH[13] = a7;
    vecH[14] = a8;
    vecH[15] = lambda*((F[0])*(F[0])) + mu;
    return H;
}

template <class TDerived>
std::tuple<Scalar, Vector<4>>
StableNeoHookeanEnergy<2>::evalWithGrad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<4> G;
    auto vecG = G.reshaped();
    Scalar const a0 = F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda;
    Scalar const a1 = a0*lambda;
    psi = (1.0/2.0)*((a0)*(a0))*lambda + (1.0/2.0)*mu*(((F[0])*(F[0])) + ((F[1])*(F[1])) + ((F[2])*(F[2])) + ((F[3])*(F[3])) - 2);
    vecG[0] = a1*F[3] + mu*F[0];
    vecG[1] = -a1*F[2] + mu*F[1];
    vecG[2] = -a1*F[1] + mu*F[2];
    vecG[3] = a1*F[0] + mu*F[3];
    return {psi, G};
}

template <class TDerived>
std::tuple<Scalar, Vector<4>, Matrix<4,4>>
StableNeoHookeanEnergy<2>::evalWithGradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<4> G;
    Matrix<4,4> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = ((F[0])*(F[0]));
    Scalar const a1 = ((F[1])*(F[1]));
    Scalar const a2 = ((F[2])*(F[2]));
    Scalar const a3 = ((F[3])*(F[3]));
    Scalar const a4 = F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda;
    Scalar const a5 = a4*lambda;
    Scalar const a6 = lambda*F[3];
    Scalar const a7 = -a6*F[2];
    Scalar const a8 = -a6*F[1];
    Scalar const a9 = a5 + lambda*F[0]*F[3];
    Scalar const a10 = -a5 + lambda*F[1]*F[2];
    Scalar const a11 = lambda*F[0];
    Scalar const a12 = -a11*F[2];
    Scalar const a13 = -a11*F[1];
    psi = (1.0/2.0)*((a4)*(a4))*lambda + (1.0/2.0)*mu*(a0 + a1 + a2 + a3 - 2);
    vecG[0] = a5*F[3] + mu*F[0];
    vecG[1] = -a5*F[2] + mu*F[1];
    vecG[2] = -a5*F[1] + mu*F[2];
    vecG[3] = a5*F[0] + mu*F[3];
    vecH[0] = a3*lambda + mu;
    vecH[1] = a7;
    vecH[2] = a8;
    vecH[3] = a9;
    vecH[4] = a7;
    vecH[5] = a2*lambda + mu;
    vecH[6] = a10;
    vecH[7] = a12;
    vecH[8] = a8;
    vecH[9] = a10;
    vecH[10] = a1*lambda + mu;
    vecH[11] = a13;
    vecH[12] = a9;
    vecH[13] = a12;
    vecH[14] = a13;
    vecH[15] = a0*lambda + mu;
    return {psi, G, H};
}

template <class TDerived>
std::tuple<Vector<4>, Matrix<4,4>>
StableNeoHookeanEnergy<2>::gradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F, 
    [[maybe_unused]] Scalar mu, 
    [[maybe_unused]] Scalar lambda) const
{
    Vector<4> G;
    Matrix<4,4> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = lambda*(F[0]*F[3] - F[1]*F[2] - 1 - mu/lambda);
    Scalar const a1 = lambda*F[3];
    Scalar const a2 = -a1*F[2];
    Scalar const a3 = -a1*F[1];
    Scalar const a4 = a0 + lambda*F[0]*F[3];
    Scalar const a5 = -a0 + lambda*F[1]*F[2];
    Scalar const a6 = lambda*F[0];
    Scalar const a7 = -a6*F[2];
    Scalar const a8 = -a6*F[1];
    vecG[0] = a0*F[3] + mu*F[0];
    vecG[1] = -a0*F[2] + mu*F[1];
    vecG[2] = -a0*F[1] + mu*F[2];
    vecG[3] = a0*F[0] + mu*F[3];
    vecH[0] = lambda*((F[3])*(F[3])) + mu;
    vecH[1] = a2;
    vecH[2] = a3;
    vecH[3] = a4;
    vecH[4] = a2;
    vecH[5] = lambda*((F[2])*(F[2])) + mu;
    vecH[6] = a5;
    vecH[7] = a7;
    vecH[8] = a3;
    vecH[9] = a5;
    vecH[10] = lambda*((F[1])*(F[1])) + mu;
    vecH[11] = a8;
    vecH[12] = a4;
    vecH[13] = a7;
    vecH[14] = a8;
    vecH[15] = lambda*((F[0])*(F[0])) + mu;
    return {G, H};
}

template <>
struct StableNeoHookeanEnergy<3>
{
    public:
        static auto constexpr kDims = 3;
    
        template <class TDerived>
        Scalar
        eval(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Vector<9>
        grad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        Matrix<9,9>
        hessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<9>>
        evalWithGrad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Scalar, Vector<9>, Matrix<9,9>>
        evalWithGradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;

        template <class TDerived>
        std::tuple<Vector<9>, Matrix<9,9>>
        gradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const;
};

template <class TDerived>
Scalar
StableNeoHookeanEnergy<3>::eval(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    psi = (1.0/2.0)*lambda*((F[0]*F[4]*F[8] - F[0]*F[5]*F[7] - F[1]*F[3]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - F[2]*F[4]*F[6] - 1 - mu/lambda)*(F[0]*F[4]*F[8] - F[0]*F[5]*F[7] - F[1]*F[3]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - F[2]*F[4]*F[6] - 1 - mu/lambda)) + (1.0/2.0)*mu*(((F[0])*(F[0])) + ((F[1])*(F[1])) + ((F[2])*(F[2])) + ((F[3])*(F[3])) + ((F[4])*(F[4])) + ((F[5])*(F[5])) + ((F[6])*(F[6])) + ((F[7])*(F[7])) + ((F[8])*(F[8])) - 3);
    return psi;
}

template <class TDerived>
Vector<9>
StableNeoHookeanEnergy<3>::grad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Vector<9> G;
    auto vecG = G.reshaped();
    Scalar const a0 = F[5]*F[7];
    Scalar const a1 = F[3]*F[8];
    Scalar const a2 = F[4]*F[6];
    Scalar const a3 = (1.0/2.0)*lambda*(-a0*F[0] - a1*F[1] - a2*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda);
    Scalar const a4 = 2*F[8];
    Scalar const a5 = 2*F[2];
    Scalar const a6 = 2*F[0];
    Scalar const a7 = 2*F[1];
    vecG[0] = a3*(-2*a0 + 2*F[4]*F[8]) + mu*F[0];
    vecG[1] = a3*(-2*a1 + 2*F[5]*F[6]) + mu*F[1];
    vecG[2] = a3*(-2*a2 + 2*F[3]*F[7]) + mu*F[2];
    vecG[3] = a3*(-a4*F[1] + 2*F[2]*F[7]) + mu*F[3];
    vecG[4] = a3*(a4*F[0] - a5*F[6]) + mu*F[4];
    vecG[5] = a3*(-a6*F[7] + 2*F[1]*F[6]) + mu*F[5];
    vecG[6] = a3*(-a5*F[4] + a7*F[5]) + mu*F[6];
    vecG[7] = a3*(-a6*F[5] + 2*F[2]*F[3]) + mu*F[7];
    vecG[8] = a3*(a6*F[4] - a7*F[3]) + mu*F[8];
    return G;
}

template <class TDerived>
Matrix<9,9>
StableNeoHookeanEnergy<3>::hessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Matrix<9,9> H;
    auto vecH = H.reshaped();
    Scalar const a0 = F[4]*F[8];
    Scalar const a1 = F[5]*F[7];
    Scalar const a2 = a0 - a1;
    Scalar const a3 = (1.0/2.0)*lambda;
    Scalar const a4 = a3*(2*a0 - 2*a1);
    Scalar const a5 = F[3]*F[8];
    Scalar const a6 = -a5 + F[5]*F[6];
    Scalar const a7 = F[3]*F[7];
    Scalar const a8 = F[4]*F[6];
    Scalar const a9 = a7 - a8;
    Scalar const a10 = F[1]*F[8];
    Scalar const a11 = -a10 + F[2]*F[7];
    Scalar const a12 = F[0]*F[8];
    Scalar const a13 = F[2]*F[6];
    Scalar const a14 = a12 - a13;
    Scalar const a15 = lambda*(-a1*F[0] - a5*F[1] - a8*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda);
    Scalar const a16 = a15*F[8];
    Scalar const a17 = F[0]*F[7];
    Scalar const a18 = -a17 + F[1]*F[6];
    Scalar const a19 = a15*F[7];
    Scalar const a20 = -a19;
    Scalar const a21 = F[1]*F[5];
    Scalar const a22 = F[2]*F[4];
    Scalar const a23 = a21 - a22;
    Scalar const a24 = F[0]*F[5];
    Scalar const a25 = -a24 + F[2]*F[3];
    Scalar const a26 = a15*F[5];
    Scalar const a27 = -a26;
    Scalar const a28 = F[0]*F[4];
    Scalar const a29 = F[1]*F[3];
    Scalar const a30 = a28 - a29;
    Scalar const a31 = a15*F[4];
    Scalar const a32 = a3*(-2*a5 + 2*F[5]*F[6]);
    Scalar const a33 = -a16;
    Scalar const a34 = a15*F[6];
    Scalar const a35 = a15*F[3];
    Scalar const a36 = -a35;
    Scalar const a37 = a3*(2*a7 - 2*a8);
    Scalar const a38 = -a34;
    Scalar const a39 = -a31;
    Scalar const a40 = a3*(-2*a10 + 2*F[2]*F[7]);
    Scalar const a41 = a15*F[2];
    Scalar const a42 = a15*F[1];
    Scalar const a43 = -a42;
    Scalar const a44 = a3*(2*a12 - 2*a13);
    Scalar const a45 = -a41;
    Scalar const a46 = a15*F[0];
    Scalar const a47 = a3*(-2*a17 + 2*F[1]*F[6]);
    Scalar const a48 = -a46;
    Scalar const a49 = a3*(2*a21 - 2*a22);
    Scalar const a50 = a3*(-2*a24 + 2*F[2]*F[3]);
    Scalar const a51 = a3*(2*a28 - 2*a29);
    vecH[0] = a2*a4 + mu;
    vecH[1] = a4*a6;
    vecH[2] = a4*a9;
    vecH[3] = a11*a4;
    vecH[4] = a14*a4 + a16;
    vecH[5] = a18*a4 + a20;
    vecH[6] = a23*a4;
    vecH[7] = a25*a4 + a27;
    vecH[8] = a30*a4 + a31;
    vecH[9] = a2*a32;
    vecH[10] = a32*a6 + mu;
    vecH[11] = a32*a9;
    vecH[12] = a11*a32 + a33;
    vecH[13] = a14*a32;
    vecH[14] = a18*a32 + a34;
    vecH[15] = a23*a32 + a26;
    vecH[16] = a25*a32;
    vecH[17] = a30*a32 + a36;
    vecH[18] = a2*a37;
    vecH[19] = a37*a6;
    vecH[20] = a37*a9 + mu;
    vecH[21] = a11*a37 + a19;
    vecH[22] = a14*a37 + a38;
    vecH[23] = a18*a37;
    vecH[24] = a23*a37 + a39;
    vecH[25] = a25*a37 + a35;
    vecH[26] = a30*a37;
    vecH[27] = a2*a40;
    vecH[28] = a33 + a40*a6;
    vecH[29] = a19 + a40*a9;
    vecH[30] = a11*a40 + mu;
    vecH[31] = a14*a40;
    vecH[32] = a18*a40;
    vecH[33] = a23*a40;
    vecH[34] = a25*a40 + a41;
    vecH[35] = a30*a40 + a43;
    vecH[36] = a16 + a2*a44;
    vecH[37] = a44*a6;
    vecH[38] = a38 + a44*a9;
    vecH[39] = a11*a44;
    vecH[40] = a14*a44 + mu;
    vecH[41] = a18*a44;
    vecH[42] = a23*a44 + a45;
    vecH[43] = a25*a44;
    vecH[44] = a30*a44 + a46;
    vecH[45] = a2*a47 + a20;
    vecH[46] = a34 + a47*a6;
    vecH[47] = a47*a9;
    vecH[48] = a11*a47;
    vecH[49] = a14*a47;
    vecH[50] = a18*a47 + mu;
    vecH[51] = a23*a47 + a42;
    vecH[52] = a25*a47 + a48;
    vecH[53] = a30*a47;
    vecH[54] = a2*a49;
    vecH[55] = a26 + a49*a6;
    vecH[56] = a39 + a49*a9;
    vecH[57] = a11*a49;
    vecH[58] = a14*a49 + a45;
    vecH[59] = a18*a49 + a42;
    vecH[60] = a23*a49 + mu;
    vecH[61] = a25*a49;
    vecH[62] = a30*a49;
    vecH[63] = a2*a50 + a27;
    vecH[64] = a50*a6;
    vecH[65] = a35 + a50*a9;
    vecH[66] = a11*a50 + a41;
    vecH[67] = a14*a50;
    vecH[68] = a18*a50 + a48;
    vecH[69] = a23*a50;
    vecH[70] = a25*a50 + mu;
    vecH[71] = a30*a50;
    vecH[72] = a2*a51 + a31;
    vecH[73] = a36 + a51*a6;
    vecH[74] = a51*a9;
    vecH[75] = a11*a51 + a43;
    vecH[76] = a14*a51 + a46;
    vecH[77] = a18*a51;
    vecH[78] = a23*a51;
    vecH[79] = a25*a51;
    vecH[80] = a30*a51 + mu;
    return H;
}

template <class TDerived>
std::tuple<Scalar, Vector<9>>
StableNeoHookeanEnergy<3>::evalWithGrad(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    auto vecG = G.reshaped();
    Scalar const a0 = F[5]*F[7];
    Scalar const a1 = F[3]*F[8];
    Scalar const a2 = F[4]*F[6];
    Scalar const a3 = -a0*F[0] - a1*F[1] - a2*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda;
    Scalar const a4 = (1.0/2.0)*lambda;
    Scalar const a5 = a3*a4;
    Scalar const a6 = 2*F[8];
    Scalar const a7 = 2*F[2];
    Scalar const a8 = 2*F[0];
    Scalar const a9 = 2*F[1];
    psi = ((a3)*(a3))*a4 + (1.0/2.0)*mu*(((F[0])*(F[0])) + ((F[1])*(F[1])) + ((F[2])*(F[2])) + ((F[3])*(F[3])) + ((F[4])*(F[4])) + ((F[5])*(F[5])) + ((F[6])*(F[6])) + ((F[7])*(F[7])) + ((F[8])*(F[8])) - 3);
    vecG[0] = a5*(-2*a0 + 2*F[4]*F[8]) + mu*F[0];
    vecG[1] = a5*(-2*a1 + 2*F[5]*F[6]) + mu*F[1];
    vecG[2] = a5*(-2*a2 + 2*F[3]*F[7]) + mu*F[2];
    vecG[3] = a5*(-a6*F[1] + 2*F[2]*F[7]) + mu*F[3];
    vecG[4] = a5*(a6*F[0] - a7*F[6]) + mu*F[4];
    vecG[5] = a5*(-a8*F[7] + 2*F[1]*F[6]) + mu*F[5];
    vecG[6] = a5*(-a7*F[4] + a9*F[5]) + mu*F[6];
    vecG[7] = a5*(-a8*F[5] + 2*F[2]*F[3]) + mu*F[7];
    vecG[8] = a5*(a8*F[4] - a9*F[3]) + mu*F[8];
    return {psi, G};
}

template <class TDerived>
std::tuple<Scalar, Vector<9>, Matrix<9,9>>
StableNeoHookeanEnergy<3>::evalWithGradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F,
    [[maybe_unused]] Scalar mu,
    [[maybe_unused]] Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    Matrix<9,9> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = F[5]*F[7];
    Scalar const a1 = F[3]*F[8];
    Scalar const a2 = F[4]*F[6];
    Scalar const a3 = -a0*F[0] - a1*F[1] - a2*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda;
    Scalar const a4 = (1.0/2.0)*lambda;
    Scalar const a5 = F[4]*F[8];
    Scalar const a6 = -2*a0 + 2*a5;
    Scalar const a7 = a3*a4;
    Scalar const a8 = -2*a1 + 2*F[5]*F[6];
    Scalar const a9 = F[3]*F[7];
    Scalar const a10 = -2*a2 + 2*a9;
    Scalar const a11 = F[1]*F[8];
    Scalar const a12 = -2*a11 + 2*F[2]*F[7];
    Scalar const a13 = F[0]*F[8];
    Scalar const a14 = F[2]*F[6];
    Scalar const a15 = 2*a13 - 2*a14;
    Scalar const a16 = F[0]*F[7];
    Scalar const a17 = -2*a16 + 2*F[1]*F[6];
    Scalar const a18 = F[1]*F[5];
    Scalar const a19 = F[2]*F[4];
    Scalar const a20 = 2*a18 - 2*a19;
    Scalar const a21 = F[0]*F[5];
    Scalar const a22 = -2*a21 + 2*F[2]*F[3];
    Scalar const a23 = F[0]*F[4];
    Scalar const a24 = F[1]*F[3];
    Scalar const a25 = 2*a23 - 2*a24;
    Scalar const a26 = a4*(-a0 + a5);
    Scalar const a27 = a3*lambda;
    Scalar const a28 = a27*F[8];
    Scalar const a29 = a27*F[7];
    Scalar const a30 = -a29;
    Scalar const a31 = a27*F[5];
    Scalar const a32 = -a31;
    Scalar const a33 = a27*F[4];
    Scalar const a34 = a4*(-a1 + F[5]*F[6]);
    Scalar const a35 = -a28;
    Scalar const a36 = a27*F[6];
    Scalar const a37 = a27*F[3];
    Scalar const a38 = -a37;
    Scalar const a39 = a4*(-a2 + a9);
    Scalar const a40 = -a36;
    Scalar const a41 = -a33;
    Scalar const a42 = a4*(-a11 + F[2]*F[7]);
    Scalar const a43 = a27*F[2];
    Scalar const a44 = a27*F[1];
    Scalar const a45 = -a44;
    Scalar const a46 = a4*(a13 - a14);
    Scalar const a47 = -a43;
    Scalar const a48 = a27*F[0];
    Scalar const a49 = a4*(-a16 + F[1]*F[6]);
    Scalar const a50 = -a48;
    Scalar const a51 = a4*(a18 - a19);
    Scalar const a52 = a4*(-a21 + F[2]*F[3]);
    Scalar const a53 = a4*(a23 - a24);
    psi = ((a3)*(a3))*a4 + (1.0/2.0)*mu*(((F[0])*(F[0])) + ((F[1])*(F[1])) + ((F[2])*(F[2])) + ((F[3])*(F[3])) + ((F[4])*(F[4])) + ((F[5])*(F[5])) + ((F[6])*(F[6])) + ((F[7])*(F[7])) + ((F[8])*(F[8])) - 3);
    vecG[0] = a6*a7 + mu*F[0];
    vecG[1] = a7*a8 + mu*F[1];
    vecG[2] = a10*a7 + mu*F[2];
    vecG[3] = a12*a7 + mu*F[3];
    vecG[4] = a15*a7 + mu*F[4];
    vecG[5] = a17*a7 + mu*F[5];
    vecG[6] = a20*a7 + mu*F[6];
    vecG[7] = a22*a7 + mu*F[7];
    vecG[8] = a25*a7 + mu*F[8];
    vecH[0] = a26*a6 + mu;
    vecH[1] = a26*a8;
    vecH[2] = a10*a26;
    vecH[3] = a12*a26;
    vecH[4] = a15*a26 + a28;
    vecH[5] = a17*a26 + a30;
    vecH[6] = a20*a26;
    vecH[7] = a22*a26 + a32;
    vecH[8] = a25*a26 + a33;
    vecH[9] = a34*a6;
    vecH[10] = a34*a8 + mu;
    vecH[11] = a10*a34;
    vecH[12] = a12*a34 + a35;
    vecH[13] = a15*a34;
    vecH[14] = a17*a34 + a36;
    vecH[15] = a20*a34 + a31;
    vecH[16] = a22*a34;
    vecH[17] = a25*a34 + a38;
    vecH[18] = a39*a6;
    vecH[19] = a39*a8;
    vecH[20] = a10*a39 + mu;
    vecH[21] = a12*a39 + a29;
    vecH[22] = a15*a39 + a40;
    vecH[23] = a17*a39;
    vecH[24] = a20*a39 + a41;
    vecH[25] = a22*a39 + a37;
    vecH[26] = a25*a39;
    vecH[27] = a42*a6;
    vecH[28] = a35 + a42*a8;
    vecH[29] = a10*a42 + a29;
    vecH[30] = a12*a42 + mu;
    vecH[31] = a15*a42;
    vecH[32] = a17*a42;
    vecH[33] = a20*a42;
    vecH[34] = a22*a42 + a43;
    vecH[35] = a25*a42 + a45;
    vecH[36] = a28 + a46*a6;
    vecH[37] = a46*a8;
    vecH[38] = a10*a46 + a40;
    vecH[39] = a12*a46;
    vecH[40] = a15*a46 + mu;
    vecH[41] = a17*a46;
    vecH[42] = a20*a46 + a47;
    vecH[43] = a22*a46;
    vecH[44] = a25*a46 + a48;
    vecH[45] = a30 + a49*a6;
    vecH[46] = a36 + a49*a8;
    vecH[47] = a10*a49;
    vecH[48] = a12*a49;
    vecH[49] = a15*a49;
    vecH[50] = a17*a49 + mu;
    vecH[51] = a20*a49 + a44;
    vecH[52] = a22*a49 + a50;
    vecH[53] = a25*a49;
    vecH[54] = a51*a6;
    vecH[55] = a31 + a51*a8;
    vecH[56] = a10*a51 + a41;
    vecH[57] = a12*a51;
    vecH[58] = a15*a51 + a47;
    vecH[59] = a17*a51 + a44;
    vecH[60] = a20*a51 + mu;
    vecH[61] = a22*a51;
    vecH[62] = a25*a51;
    vecH[63] = a32 + a52*a6;
    vecH[64] = a52*a8;
    vecH[65] = a10*a52 + a37;
    vecH[66] = a12*a52 + a43;
    vecH[67] = a15*a52;
    vecH[68] = a17*a52 + a50;
    vecH[69] = a20*a52;
    vecH[70] = a22*a52 + mu;
    vecH[71] = a25*a52;
    vecH[72] = a33 + a53*a6;
    vecH[73] = a38 + a53*a8;
    vecH[74] = a10*a53;
    vecH[75] = a12*a53 + a45;
    vecH[76] = a15*a53 + a48;
    vecH[77] = a17*a53;
    vecH[78] = a20*a53;
    vecH[79] = a22*a53;
    vecH[80] = a25*a53 + mu;
    return {psi, G, H};
}

template <class TDerived>
std::tuple<Vector<9>, Matrix<9,9>>
StableNeoHookeanEnergy<3>::gradAndHessian(
    [[maybe_unused]] Eigen::DenseBase<TDerived> const& F, 
    [[maybe_unused]] Scalar mu, 
    [[maybe_unused]] Scalar lambda) const
{
    Vector<9> G;
    Matrix<9,9> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
    Scalar const a0 = F[4]*F[8];
    Scalar const a1 = F[5]*F[7];
    Scalar const a2 = 2*a0 - 2*a1;
    Scalar const a3 = F[3]*F[8];
    Scalar const a4 = F[4]*F[6];
    Scalar const a5 = lambda*(-a1*F[0] - a3*F[1] - a4*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda);
    Scalar const a6 = (1.0/2.0)*a5;
    Scalar const a7 = -2*a3 + 2*F[5]*F[6];
    Scalar const a8 = F[3]*F[7];
    Scalar const a9 = -2*a4 + 2*a8;
    Scalar const a10 = F[1]*F[8];
    Scalar const a11 = -2*a10 + 2*F[2]*F[7];
    Scalar const a12 = F[0]*F[8];
    Scalar const a13 = F[2]*F[6];
    Scalar const a14 = 2*a12 - 2*a13;
    Scalar const a15 = F[0]*F[7];
    Scalar const a16 = -2*a15 + 2*F[1]*F[6];
    Scalar const a17 = F[1]*F[5];
    Scalar const a18 = F[2]*F[4];
    Scalar const a19 = 2*a17 - 2*a18;
    Scalar const a20 = F[0]*F[5];
    Scalar const a21 = -2*a20 + 2*F[2]*F[3];
    Scalar const a22 = F[0]*F[4];
    Scalar const a23 = F[1]*F[3];
    Scalar const a24 = 2*a22 - 2*a23;
    Scalar const a25 = (1.0/2.0)*lambda;
    Scalar const a26 = a25*(a0 - a1);
    Scalar const a27 = a5*F[8];
    Scalar const a28 = a5*F[7];
    Scalar const a29 = -a28;
    Scalar const a30 = a5*F[5];
    Scalar const a31 = -a30;
    Scalar const a32 = a5*F[4];
    Scalar const a33 = a25*(-a3 + F[5]*F[6]);
    Scalar const a34 = -a27;
    Scalar const a35 = a5*F[6];
    Scalar const a36 = a5*F[3];
    Scalar const a37 = -a36;
    Scalar const a38 = a25*(-a4 + a8);
    Scalar const a39 = -a35;
    Scalar const a40 = -a32;
    Scalar const a41 = a25*(-a10 + F[2]*F[7]);
    Scalar const a42 = a5*F[2];
    Scalar const a43 = a5*F[1];
    Scalar const a44 = -a43;
    Scalar const a45 = a25*(a12 - a13);
    Scalar const a46 = -a42;
    Scalar const a47 = a5*F[0];
    Scalar const a48 = a25*(-a15 + F[1]*F[6]);
    Scalar const a49 = -a47;
    Scalar const a50 = a25*(a17 - a18);
    Scalar const a51 = a25*(-a20 + F[2]*F[3]);
    Scalar const a52 = a25*(a22 - a23);
    vecG[0] = a2*a6 + mu*F[0];
    vecG[1] = a6*a7 + mu*F[1];
    vecG[2] = a6*a9 + mu*F[2];
    vecG[3] = a11*a6 + mu*F[3];
    vecG[4] = a14*a6 + mu*F[4];
    vecG[5] = a16*a6 + mu*F[5];
    vecG[6] = a19*a6 + mu*F[6];
    vecG[7] = a21*a6 + mu*F[7];
    vecG[8] = a24*a6 + mu*F[8];
    vecH[0] = a2*a26 + mu;
    vecH[1] = a26*a7;
    vecH[2] = a26*a9;
    vecH[3] = a11*a26;
    vecH[4] = a14*a26 + a27;
    vecH[5] = a16*a26 + a29;
    vecH[6] = a19*a26;
    vecH[7] = a21*a26 + a31;
    vecH[8] = a24*a26 + a32;
    vecH[9] = a2*a33;
    vecH[10] = a33*a7 + mu;
    vecH[11] = a33*a9;
    vecH[12] = a11*a33 + a34;
    vecH[13] = a14*a33;
    vecH[14] = a16*a33 + a35;
    vecH[15] = a19*a33 + a30;
    vecH[16] = a21*a33;
    vecH[17] = a24*a33 + a37;
    vecH[18] = a2*a38;
    vecH[19] = a38*a7;
    vecH[20] = a38*a9 + mu;
    vecH[21] = a11*a38 + a28;
    vecH[22] = a14*a38 + a39;
    vecH[23] = a16*a38;
    vecH[24] = a19*a38 + a40;
    vecH[25] = a21*a38 + a36;
    vecH[26] = a24*a38;
    vecH[27] = a2*a41;
    vecH[28] = a34 + a41*a7;
    vecH[29] = a28 + a41*a9;
    vecH[30] = a11*a41 + mu;
    vecH[31] = a14*a41;
    vecH[32] = a16*a41;
    vecH[33] = a19*a41;
    vecH[34] = a21*a41 + a42;
    vecH[35] = a24*a41 + a44;
    vecH[36] = a2*a45 + a27;
    vecH[37] = a45*a7;
    vecH[38] = a39 + a45*a9;
    vecH[39] = a11*a45;
    vecH[40] = a14*a45 + mu;
    vecH[41] = a16*a45;
    vecH[42] = a19*a45 + a46;
    vecH[43] = a21*a45;
    vecH[44] = a24*a45 + a47;
    vecH[45] = a2*a48 + a29;
    vecH[46] = a35 + a48*a7;
    vecH[47] = a48*a9;
    vecH[48] = a11*a48;
    vecH[49] = a14*a48;
    vecH[50] = a16*a48 + mu;
    vecH[51] = a19*a48 + a43;
    vecH[52] = a21*a48 + a49;
    vecH[53] = a24*a48;
    vecH[54] = a2*a50;
    vecH[55] = a30 + a50*a7;
    vecH[56] = a40 + a50*a9;
    vecH[57] = a11*a50;
    vecH[58] = a14*a50 + a46;
    vecH[59] = a16*a50 + a43;
    vecH[60] = a19*a50 + mu;
    vecH[61] = a21*a50;
    vecH[62] = a24*a50;
    vecH[63] = a2*a51 + a31;
    vecH[64] = a51*a7;
    vecH[65] = a36 + a51*a9;
    vecH[66] = a11*a51 + a42;
    vecH[67] = a14*a51;
    vecH[68] = a16*a51 + a49;
    vecH[69] = a19*a51;
    vecH[70] = a21*a51 + mu;
    vecH[71] = a24*a51;
    vecH[72] = a2*a52 + a32;
    vecH[73] = a37 + a52*a7;
    vecH[74] = a52*a9;
    vecH[75] = a11*a52 + a44;
    vecH[76] = a14*a52 + a47;
    vecH[77] = a16*a52;
    vecH[78] = a19*a52;
    vecH[79] = a21*a52;
    vecH[80] = a24*a52 + mu;
    return {G, H};
}

} // namespace physics
} // namespace pbat

#endif // PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
