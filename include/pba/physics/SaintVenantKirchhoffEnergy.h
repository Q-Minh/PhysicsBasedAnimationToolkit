
#ifndef PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
#define PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H

#include "pba/aliases.h"

#include <cmath>
#include <tuple>

namespace pba {
namespace physics {

struct SaintVenantKirchhoffEnergy
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

        template <class Derived>
        std::tuple<Vector<9>, Matrix<9,9>>
        gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;
};

template <class Derived>
Scalar
SaintVenantKirchhoffEnergy::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Scalar const a0 = (1.0/2.0)*F[0]*F[0] + (1.0/2.0)*F[1]*F[1] + (1.0/2.0)*F[2]*F[2];
    Scalar const a1 = (1.0/2.0)*F[3]*F[3] + (1.0/2.0)*F[4]*F[4] + (1.0/2.0)*F[5]*F[5];
    Scalar const a2 = (1.0/2.0)*F[6]*F[6] + (1.0/2.0)*F[7]*F[7] + (1.0/2.0)*F[8]*F[8];
    Scalar const a3 = (1.0/2.0)*F[0];
    Scalar const a4 = (1.0/2.0)*F[1];
    Scalar const a5 = (1.0/2.0)*F[2];
    psi = (1.0/2.0)*lambda*a0 + a1 + a2 - 3.0/2.0*a0 + a1 + a2 - 3.0/2.0 + mu*(a0 - 1.0/2.0*a0 - 1.0/2.0 + a1 - 1.0/2.0*a1 - 1.0/2.0 + a2 - 1.0/2.0*a2 - 1.0/2.0 + 2*a3*F[3] + a4*F[4] + a5*F[5]*a3*F[3] + a4*F[4] + a5*F[5] + 2*a3*F[6] + a4*F[7] + a5*F[8]*a3*F[6] + a4*F[7] + a5*F[8] + 2*(1.0/2.0)*F[3]*F[6] + (1.0/2.0)*F[4]*F[7] + (1.0/2.0)*F[5]*F[8]*(1.0/2.0)*F[3]*F[6] + (1.0/2.0)*F[4]*F[7] + (1.0/2.0)*F[5]*F[8]);
    return psi;
}

template <class Derived>
Vector<9>
SaintVenantKirchhoffEnergy::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Vector<9> G;
    Scalar const a0 = (1.0/2.0)*F[0]*F[0] + (1.0/2.0)*F[1]*F[1] + (1.0/2.0)*F[2]*F[2];
    Scalar const a1 = (1.0/2.0)*F[3]*F[3] + (1.0/2.0)*F[4]*F[4] + (1.0/2.0)*F[5]*F[5];
    Scalar const a2 = (1.0/2.0)*F[6]*F[6] + (1.0/2.0)*F[7]*F[7] + (1.0/2.0)*F[8]*F[8];
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
    G[0] = a3*F[0] + mu*(a4*F[0] + a8*F[3] + a9*F[6]);
    G[1] = a3*F[1] + mu*(a4*F[1] + a8*F[4] + a9*F[7]);
    G[2] = a3*F[2] + mu*(a4*F[2] + a8*F[5] + a9*F[8]);
    G[3] = a3*F[3] + mu*(a10*F[3] + a11*F[6] + a8*F[0]);
    G[4] = a3*F[4] + mu*(a10*F[4] + a11*F[7] + a8*F[1]);
    G[5] = a3*F[5] + mu*(a10*F[5] + a11*F[8] + a8*F[2]);
    G[6] = a3*F[6] + mu*(a11*F[3] + a12*F[6] + a9*F[0]);
    G[7] = a3*F[7] + mu*(a11*F[4] + a12*F[7] + a9*F[1]);
    G[8] = a3*F[8] + mu*(a11*F[5] + a12*F[8] + a9*F[2]);
    return G;
}

template <class Derived>
Matrix<9,9>
SaintVenantKirchhoffEnergy::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Matrix<9,9> H;
    Scalar const a0 = F[0]*F[0];
    Scalar const a1 = F[1]*F[1];
    Scalar const a2 = F[3]*F[3];
    Scalar const a3 = a1 + a2;
    Scalar const a4 = F[6]*F[6];
    Scalar const a5 = F[2]*F[2];
    Scalar const a6 = a5 - 1;
    Scalar const a7 = a4 + a6;
    Scalar const a8 = F[4]*F[4];
    Scalar const a9 = F[5]*F[5];
    Scalar const a10 = F[7]*F[7];
    Scalar const a11 = F[8]*F[8];
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
    H[0] = a0*lambda + a12 + mu*(3*a0 + a3 + a7);
    H[1] = a16;
    H[2] = a20;
    H[3] = a24;
    H[4] = a27;
    H[5] = a28;
    H[6] = a32;
    H[7] = a34;
    H[8] = a35;
    H[9] = a16;
    H[10] = a1*lambda + a12 + mu*(3*a1 + a10 + a36 + a6);
    H[11] = a40;
    H[12] = a43;
    H[13] = a44;
    H[14] = a45;
    H[15] = a47;
    H[16] = a48;
    H[17] = a49;
    H[18] = a20;
    H[19] = a40;
    H[20] = a12 + a5*lambda + mu*(a1 + 3*a5 + a50 + a51);
    H[21] = a54;
    H[22] = a55;
    H[23] = a56;
    H[24] = a58;
    H[25] = a59;
    H[26] = a60;
    H[27] = a24;
    H[28] = a43;
    H[29] = a54;
    H[30] = a12 + a2*lambda + mu*(3*a2 + a36 + a4 + a50);
    H[31] = a61;
    H[32] = a62;
    H[33] = a66;
    H[34] = a68;
    H[35] = a69;
    H[36] = a27;
    H[37] = a44;
    H[38] = a55;
    H[39] = a61;
    H[40] = a12 + a8*lambda + mu*(a10 + a3 + a50 + 3*a8);
    H[41] = a70;
    H[42] = a72;
    H[43] = a73;
    H[44] = a74;
    H[45] = a28;
    H[46] = a45;
    H[47] = a56;
    H[48] = a62;
    H[49] = a70;
    H[50] = a12 + a9*lambda + mu*(a2 + a6 + a75 + 3*a9);
    H[51] = a77;
    H[52] = a78;
    H[53] = a79;
    H[54] = a32;
    H[55] = a47;
    H[56] = a58;
    H[57] = a66;
    H[58] = a72;
    H[59] = a77;
    H[60] = a12 + a4*lambda + mu*(a10 + a2 + 3*a4 + a51 - 1);
    H[61] = a80;
    H[62] = a81;
    H[63] = a34;
    H[64] = a48;
    H[65] = a59;
    H[66] = a68;
    H[67] = a73;
    H[68] = a78;
    H[69] = a80;
    H[70] = a10*lambda + a12 + mu*(a1 + 3*a10 + a4 + a75 - 1);
    H[71] = a82;
    H[72] = a35;
    H[73] = a49;
    H[74] = a60;
    H[75] = a69;
    H[76] = a74;
    H[77] = a79;
    H[78] = a81;
    H[79] = a82;
    H[80] = a11*lambda + a12 + mu*(a10 + 3*a11 + a7 + a9);
    return H;
}

template <class Derived>
std::tuple<Scalar, Vector<9>>
SaintVenantKirchhoffEnergy::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    Scalar const a0 = (1.0/2.0)*F[0]*F[0] + (1.0/2.0)*F[1]*F[1] + (1.0/2.0)*F[2]*F[2];
    Scalar const a1 = (1.0/2.0)*F[3]*F[3] + (1.0/2.0)*F[4]*F[4] + (1.0/2.0)*F[5]*F[5];
    Scalar const a2 = (1.0/2.0)*F[6]*F[6] + (1.0/2.0)*F[7]*F[7] + (1.0/2.0)*F[8]*F[8];
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
    psi = (1.0/2.0)*a3*a3*lambda + mu*(2*a10*a10 + 2*a11*a11 + 2*a12*a12 + a4*a4 + a5*a5 + a6*a6);
    G[0] = a13*F[0] + mu*(a14*F[0] + a15*F[3] + a16*F[6]);
    G[1] = a13*F[1] + mu*(a14*F[1] + a15*F[4] + a16*F[7]);
    G[2] = a13*F[2] + mu*(a14*F[2] + a15*F[5] + a16*F[8]);
    G[3] = a13*F[3] + mu*(a15*F[0] + a17*F[3] + a18*F[6]);
    G[4] = a13*F[4] + mu*(a15*F[1] + a17*F[4] + a18*F[7]);
    G[5] = a13*F[5] + mu*(a15*F[2] + a17*F[5] + a18*F[8]);
    G[6] = a13*F[6] + mu*(a16*F[0] + a18*F[3] + a19*F[6]);
    G[7] = a13*F[7] + mu*(a16*F[1] + a18*F[4] + a19*F[7]);
    G[8] = a13*F[8] + mu*(a16*F[2] + a18*F[5] + a19*F[8]);
    return {psi, G};
}

template <class Derived>
std::tuple<Scalar, Vector<9>, Matrix<9,9>>
SaintVenantKirchhoffEnergy::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{
    Scalar psi;
    Vector<9> G;
    Matrix<9,9> H;
    Scalar const a0 = F[0]*F[0];
    Scalar const a1 = F[1]*F[1];
    Scalar const a2 = F[2]*F[2];
    Scalar const a3 = (1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a2;
    Scalar const a4 = F[3]*F[3];
    Scalar const a5 = F[4]*F[4];
    Scalar const a6 = F[5]*F[5];
    Scalar const a7 = (1.0/2.0)*a4 + (1.0/2.0)*a5 + (1.0/2.0)*a6;
    Scalar const a8 = F[6]*F[6];
    Scalar const a9 = F[7]*F[7];
    Scalar const a10 = F[8]*F[8];
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
    psi = (1.0/2.0)*a12*a12*lambda + mu*(a13*a13 + a14*a14 + a15*a15 + 2*a19*a19 + 2*a23*a23 + 2*a27*a27);
    G[0] = a28*F[0] + mu*(a29*F[0] + a30*F[3] + a31*F[6]);
    G[1] = a28*F[1] + mu*(a29*F[1] + a30*F[4] + a31*F[7]);
    G[2] = a28*F[2] + mu*(a29*F[2] + a30*F[5] + a31*F[8]);
    G[3] = a28*F[3] + mu*(a30*F[0] + a32*F[3] + a33*F[6]);
    G[4] = a28*F[4] + mu*(a30*F[1] + a32*F[4] + a33*F[7]);
    G[5] = a28*F[5] + mu*(a30*F[2] + a32*F[5] + a33*F[8]);
    G[6] = a28*F[6] + mu*(a31*F[0] + a33*F[3] + a34*F[6]);
    G[7] = a28*F[7] + mu*(a31*F[1] + a33*F[4] + a34*F[7]);
    G[8] = a28*F[8] + mu*(a31*F[2] + a33*F[5] + a34*F[8]);
    H[0] = a0*lambda + a28 + mu*(3*a0 + a35 + a37);
    H[1] = a41;
    H[2] = a45;
    H[3] = a46;
    H[4] = a49;
    H[5] = a50;
    H[6] = a51;
    H[7] = a53;
    H[8] = a54;
    H[9] = a41;
    H[10] = a1*lambda + a28 + mu*(3*a1 + a36 + a55 + a9);
    H[11] = a59;
    H[12] = a62;
    H[13] = a63;
    H[14] = a64;
    H[15] = a66;
    H[16] = a67;
    H[17] = a68;
    H[18] = a45;
    H[19] = a59;
    H[20] = a2*lambda + a28 + mu*(a1 + 3*a2 + a69 + a70);
    H[21] = a73;
    H[22] = a74;
    H[23] = a75;
    H[24] = a77;
    H[25] = a78;
    H[26] = a79;
    H[27] = a46;
    H[28] = a62;
    H[29] = a73;
    H[30] = a28 + a4*lambda + mu*(3*a4 + a55 + a69 + a8);
    H[31] = a80;
    H[32] = a81;
    H[33] = a82;
    H[34] = a84;
    H[35] = a85;
    H[36] = a49;
    H[37] = a63;
    H[38] = a74;
    H[39] = a80;
    H[40] = a28 + a5*lambda + mu*(a35 + 3*a5 + a69 + a9);
    H[41] = a86;
    H[42] = a88;
    H[43] = a89;
    H[44] = a90;
    H[45] = a50;
    H[46] = a64;
    H[47] = a75;
    H[48] = a81;
    H[49] = a86;
    H[50] = a28 + a6*lambda + mu*(a36 + a4 + 3*a6 + a91);
    H[51] = a93;
    H[52] = a94;
    H[53] = a95;
    H[54] = a51;
    H[55] = a66;
    H[56] = a77;
    H[57] = a82;
    H[58] = a88;
    H[59] = a93;
    H[60] = a28 + a8*lambda + mu*(a4 + a70 + 3*a8 + a9 - 1);
    H[61] = a96;
    H[62] = a97;
    H[63] = a53;
    H[64] = a67;
    H[65] = a78;
    H[66] = a84;
    H[67] = a89;
    H[68] = a94;
    H[69] = a96;
    H[70] = a28 + a9*lambda + mu*(a1 + a8 + 3*a9 + a91 - 1);
    H[71] = a98;
    H[72] = a54;
    H[73] = a68;
    H[74] = a79;
    H[75] = a85;
    H[76] = a90;
    H[77] = a95;
    H[78] = a97;
    H[79] = a98;
    H[80] = a10*lambda + a28 + mu*(3*a10 + a37 + a6 + a9);
    return {psi, G, H};
}

template <class Derived>
std::tuple<Vector<9>, Matrix<9,9>>
SaintVenantKirchhoffEnergy::gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
{
    Vector<9> G;
    Matrix<9,9> H;
    Scalar const a0 = F[0]*F[0];
    Scalar const a1 = F[1]*F[1];
    Scalar const a2 = F[2]*F[2];
    Scalar const a3 = (1.0/2.0)*a0 + (1.0/2.0)*a1 + (1.0/2.0)*a2;
    Scalar const a4 = F[3]*F[3];
    Scalar const a5 = F[4]*F[4];
    Scalar const a6 = F[5]*F[5];
    Scalar const a7 = (1.0/2.0)*a4 + (1.0/2.0)*a5 + (1.0/2.0)*a6;
    Scalar const a8 = F[6]*F[6];
    Scalar const a9 = F[7]*F[7];
    Scalar const a10 = F[8]*F[8];
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
    G[0] = a12*F[0] + mu*(a13*F[0] + a17*F[3] + a21*F[6]);
    G[1] = a12*F[1] + mu*(a13*F[1] + a17*F[4] + a21*F[7]);
    G[2] = a12*F[2] + mu*(a13*F[2] + a17*F[5] + a21*F[8]);
    G[3] = a12*F[3] + mu*(a17*F[0] + a22*F[3] + a26*F[6]);
    G[4] = a12*F[4] + mu*(a17*F[1] + a22*F[4] + a26*F[7]);
    G[5] = a12*F[5] + mu*(a17*F[2] + a22*F[5] + a26*F[8]);
    G[6] = a12*F[6] + mu*(a21*F[0] + a26*F[3] + a27*F[6]);
    G[7] = a12*F[7] + mu*(a21*F[1] + a26*F[4] + a27*F[7]);
    G[8] = a12*F[8] + mu*(a21*F[2] + a26*F[5] + a27*F[8]);
    H[0] = a0*lambda + a12 + mu*(3*a0 + a28 + a30);
    H[1] = a34;
    H[2] = a38;
    H[3] = a39;
    H[4] = a42;
    H[5] = a43;
    H[6] = a44;
    H[7] = a46;
    H[8] = a47;
    H[9] = a34;
    H[10] = a1*lambda + a12 + mu*(3*a1 + a29 + a48 + a9);
    H[11] = a52;
    H[12] = a55;
    H[13] = a56;
    H[14] = a57;
    H[15] = a59;
    H[16] = a60;
    H[17] = a61;
    H[18] = a38;
    H[19] = a52;
    H[20] = a12 + a2*lambda + mu*(a1 + 3*a2 + a62 + a63);
    H[21] = a66;
    H[22] = a67;
    H[23] = a68;
    H[24] = a70;
    H[25] = a71;
    H[26] = a72;
    H[27] = a39;
    H[28] = a55;
    H[29] = a66;
    H[30] = a12 + a4*lambda + mu*(3*a4 + a48 + a62 + a8);
    H[31] = a73;
    H[32] = a74;
    H[33] = a75;
    H[34] = a77;
    H[35] = a78;
    H[36] = a42;
    H[37] = a56;
    H[38] = a67;
    H[39] = a73;
    H[40] = a12 + a5*lambda + mu*(a28 + 3*a5 + a62 + a9);
    H[41] = a79;
    H[42] = a81;
    H[43] = a82;
    H[44] = a83;
    H[45] = a43;
    H[46] = a57;
    H[47] = a68;
    H[48] = a74;
    H[49] = a79;
    H[50] = a12 + a6*lambda + mu*(a29 + a4 + 3*a6 + a84);
    H[51] = a86;
    H[52] = a87;
    H[53] = a88;
    H[54] = a44;
    H[55] = a59;
    H[56] = a70;
    H[57] = a75;
    H[58] = a81;
    H[59] = a86;
    H[60] = a12 + a8*lambda + mu*(a4 + a63 + 3*a8 + a9 - 1);
    H[61] = a89;
    H[62] = a90;
    H[63] = a46;
    H[64] = a60;
    H[65] = a71;
    H[66] = a77;
    H[67] = a82;
    H[68] = a87;
    H[69] = a89;
    H[70] = a12 + a9*lambda + mu*(a1 + a8 + a84 + 3*a9 - 1);
    H[71] = a91;
    H[72] = a47;
    H[73] = a61;
    H[74] = a72;
    H[75] = a78;
    H[76] = a83;
    H[77] = a88;
    H[78] = a90;
    H[79] = a91;
    H[80] = a10*lambda + a12 + mu*(3*a10 + a30 + a6 + a9);
    return {G, H};
}

} // physics
} // pba

#endif // PBA_CORE_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
