
#ifndef PBA_CORE_PHYSICS_STABLENEOHOOKEANENERGY_H
#define PBA_CORE_PHYSICS_STABLENEOHOOKEANENERGY_H

#include "pba/aliases.h"

#include <cmath>
#include <tuple>

namespace pba {
namespace physics {

class StableNeoHookeanEnergy
{
    public:
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
    private:
};

template <class Derived>
Scalar
StableNeoHookeanEnergy::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    psi = (1.0/2.0)*lambda*F[0]*F[4]*F[8] - F[0]*F[5]*F[7] - F[1]*F[3]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - F[2]*F[4]*F[6] - 1 - mu/lambda*F[0]*F[4]*F[8] - F[0]*F[5]*F[7] - F[1]*F[3]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - F[2]*F[4]*F[6] - 1 - mu/lambda + (1.0/2.0)*mu*(F[0]*F[0] + F[1]*F[1] + F[2]*F[2] + F[3]*F[3] + F[4]*F[4] + F[5]*F[5] + F[6]*F[6] + F[7]*F[7] + F[8]*F[8] - 3);
    return psi;
}

template <class Derived>
Vector<9>
StableNeoHookeanEnergy::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Vector<9> G;
    Scalar const a0 = F[5]*F[7];
    Scalar const a1 = F[3]*F[8];
    Scalar const a2 = F[4]*F[6];
    Scalar const a3 = (1.0/2.0)*lambda*(-a0*F[0] - a1*F[1] - a2*F[2] + F[0]*F[4]*F[8] + F[1]*F[5]*F[6] + F[2]*F[3]*F[7] - 1 - mu/lambda);
    Scalar const a4 = 2*F[8];
    Scalar const a5 = 2*F[2];
    Scalar const a6 = 2*F[0];
    Scalar const a7 = 2*F[1];
    G[0] = a3*(-2*a0 + 2*F[4]*F[8]) + mu*F[0];
    G[1] = a3*(-2*a1 + 2*F[5]*F[6]) + mu*F[1];
    G[2] = a3*(-2*a2 + 2*F[3]*F[7]) + mu*F[2];
    G[3] = a3*(-a4*F[1] + 2*F[2]*F[7]) + mu*F[3];
    G[4] = a3*(a4*F[0] - a5*F[6]) + mu*F[4];
    G[5] = a3*(-a6*F[7] + 2*F[1]*F[6]) + mu*F[5];
    G[6] = a3*(-a5*F[4] + a7*F[5]) + mu*F[6];
    G[7] = a3*(-a6*F[5] + 2*F[2]*F[3]) + mu*F[7];
    G[8] = a3*(a6*F[4] - a7*F[3]) + mu*F[8];
    return G;
}

template <class Derived>
Matrix<9,9>
StableNeoHookeanEnergy::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Matrix<9,9> H;
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
    H[0] = a2*a4 + mu;
    H[1] = a4*a6;
    H[2] = a4*a9;
    H[3] = a11*a4;
    H[4] = a14*a4 + a16;
    H[5] = a18*a4 + a20;
    H[6] = a23*a4;
    H[7] = a25*a4 + a27;
    H[8] = a30*a4 + a31;
    H[9] = a2*a32;
    H[10] = a32*a6 + mu;
    H[11] = a32*a9;
    H[12] = a11*a32 + a33;
    H[13] = a14*a32;
    H[14] = a18*a32 + a34;
    H[15] = a23*a32 + a26;
    H[16] = a25*a32;
    H[17] = a30*a32 + a36;
    H[18] = a2*a37;
    H[19] = a37*a6;
    H[20] = a37*a9 + mu;
    H[21] = a11*a37 + a19;
    H[22] = a14*a37 + a38;
    H[23] = a18*a37;
    H[24] = a23*a37 + a39;
    H[25] = a25*a37 + a35;
    H[26] = a30*a37;
    H[27] = a2*a40;
    H[28] = a33 + a40*a6;
    H[29] = a19 + a40*a9;
    H[30] = a11*a40 + mu;
    H[31] = a14*a40;
    H[32] = a18*a40;
    H[33] = a23*a40;
    H[34] = a25*a40 + a41;
    H[35] = a30*a40 + a43;
    H[36] = a16 + a2*a44;
    H[37] = a44*a6;
    H[38] = a38 + a44*a9;
    H[39] = a11*a44;
    H[40] = a14*a44 + mu;
    H[41] = a18*a44;
    H[42] = a23*a44 + a45;
    H[43] = a25*a44;
    H[44] = a30*a44 + a46;
    H[45] = a2*a47 + a20;
    H[46] = a34 + a47*a6;
    H[47] = a47*a9;
    H[48] = a11*a47;
    H[49] = a14*a47;
    H[50] = a18*a47 + mu;
    H[51] = a23*a47 + a42;
    H[52] = a25*a47 + a48;
    H[53] = a30*a47;
    H[54] = a2*a49;
    H[55] = a26 + a49*a6;
    H[56] = a39 + a49*a9;
    H[57] = a11*a49;
    H[58] = a14*a49 + a45;
    H[59] = a18*a49 + a42;
    H[60] = a23*a49 + mu;
    H[61] = a25*a49;
    H[62] = a30*a49;
    H[63] = a2*a50 + a27;
    H[64] = a50*a6;
    H[65] = a35 + a50*a9;
    H[66] = a11*a50 + a41;
    H[67] = a14*a50;
    H[68] = a18*a50 + a48;
    H[69] = a23*a50;
    H[70] = a25*a50 + mu;
    H[71] = a30*a50;
    H[72] = a2*a51 + a31;
    H[73] = a36 + a51*a6;
    H[74] = a51*a9;
    H[75] = a11*a51 + a43;
    H[76] = a14*a51 + a46;
    H[77] = a18*a51;
    H[78] = a23*a51;
    H[79] = a25*a51;
    H[80] = a30*a51 + mu;
    return H;
}

template <class Derived>
std::tuple<Scalar, Vector<9>>
StableNeoHookeanEnergy::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
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
    psi = a3*a3*a4 + (1.0/2.0)*mu*(F[0]*F[0] + F[1]*F[1] + F[2]*F[2] + F[3]*F[3] + F[4]*F[4] + F[5]*F[5] + F[6]*F[6] + F[7]*F[7] + F[8]*F[8] - 3);
    G[0] = a5*(-2*a0 + 2*F[4]*F[8]) + mu*F[0];
    G[1] = a5*(-2*a1 + 2*F[5]*F[6]) + mu*F[1];
    G[2] = a5*(-2*a2 + 2*F[3]*F[7]) + mu*F[2];
    G[3] = a5*(-a6*F[1] + 2*F[2]*F[7]) + mu*F[3];
    G[4] = a5*(a6*F[0] - a7*F[6]) + mu*F[4];
    G[5] = a5*(-a8*F[7] + 2*F[1]*F[6]) + mu*F[5];
    G[6] = a5*(-a7*F[4] + a9*F[5]) + mu*F[6];
    G[7] = a5*(-a8*F[5] + 2*F[2]*F[3]) + mu*F[7];
    G[8] = a5*(a8*F[4] - a9*F[3]) + mu*F[8];
    return {psi, G};
}

template <class Derived>
std::tuple<Scalar, Vector<9>, Matrix<9,9>>
StableNeoHookeanEnergy::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    Matrix<9,9> H;
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
    psi = a3*a3*a4 + (1.0/2.0)*mu*(F[0]*F[0] + F[1]*F[1] + F[2]*F[2] + F[3]*F[3] + F[4]*F[4] + F[5]*F[5] + F[6]*F[6] + F[7]*F[7] + F[8]*F[8] - 3);
    G[0] = a6*a7 + mu*F[0];
    G[1] = a7*a8 + mu*F[1];
    G[2] = a10*a7 + mu*F[2];
    G[3] = a12*a7 + mu*F[3];
    G[4] = a15*a7 + mu*F[4];
    G[5] = a17*a7 + mu*F[5];
    G[6] = a20*a7 + mu*F[6];
    G[7] = a22*a7 + mu*F[7];
    G[8] = a25*a7 + mu*F[8];
    H[0] = a26*a6 + mu;
    H[1] = a26*a8;
    H[2] = a10*a26;
    H[3] = a12*a26;
    H[4] = a15*a26 + a28;
    H[5] = a17*a26 + a30;
    H[6] = a20*a26;
    H[7] = a22*a26 + a32;
    H[8] = a25*a26 + a33;
    H[9] = a34*a6;
    H[10] = a34*a8 + mu;
    H[11] = a10*a34;
    H[12] = a12*a34 + a35;
    H[13] = a15*a34;
    H[14] = a17*a34 + a36;
    H[15] = a20*a34 + a31;
    H[16] = a22*a34;
    H[17] = a25*a34 + a38;
    H[18] = a39*a6;
    H[19] = a39*a8;
    H[20] = a10*a39 + mu;
    H[21] = a12*a39 + a29;
    H[22] = a15*a39 + a40;
    H[23] = a17*a39;
    H[24] = a20*a39 + a41;
    H[25] = a22*a39 + a37;
    H[26] = a25*a39;
    H[27] = a42*a6;
    H[28] = a35 + a42*a8;
    H[29] = a10*a42 + a29;
    H[30] = a12*a42 + mu;
    H[31] = a15*a42;
    H[32] = a17*a42;
    H[33] = a20*a42;
    H[34] = a22*a42 + a43;
    H[35] = a25*a42 + a45;
    H[36] = a28 + a46*a6;
    H[37] = a46*a8;
    H[38] = a10*a46 + a40;
    H[39] = a12*a46;
    H[40] = a15*a46 + mu;
    H[41] = a17*a46;
    H[42] = a20*a46 + a47;
    H[43] = a22*a46;
    H[44] = a25*a46 + a48;
    H[45] = a30 + a49*a6;
    H[46] = a36 + a49*a8;
    H[47] = a10*a49;
    H[48] = a12*a49;
    H[49] = a15*a49;
    H[50] = a17*a49 + mu;
    H[51] = a20*a49 + a44;
    H[52] = a22*a49 + a50;
    H[53] = a25*a49;
    H[54] = a51*a6;
    H[55] = a31 + a51*a8;
    H[56] = a10*a51 + a41;
    H[57] = a12*a51;
    H[58] = a15*a51 + a47;
    H[59] = a17*a51 + a44;
    H[60] = a20*a51 + mu;
    H[61] = a22*a51;
    H[62] = a25*a51;
    H[63] = a32 + a52*a6;
    H[64] = a52*a8;
    H[65] = a10*a52 + a37;
    H[66] = a12*a52 + a43;
    H[67] = a15*a52;
    H[68] = a17*a52 + a50;
    H[69] = a20*a52;
    H[70] = a22*a52 + mu;
    H[71] = a25*a52;
    H[72] = a33 + a53*a6;
    H[73] = a38 + a53*a8;
    H[74] = a10*a53;
    H[75] = a12*a53 + a45;
    H[76] = a15*a53 + a48;
    H[77] = a17*a53;
    H[78] = a20*a53;
    H[79] = a22*a53;
    H[80] = a25*a53 + mu;
    return {psi, G, H};
}

} // physics
} // pba

#endif // PBA_CORE_PHYSICS_STABLENEOHOOKEANENERGY_H
