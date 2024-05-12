
#ifndef PBA_CORE_FEM_HEXAHEDRON_H    
#define PBA_CORE_FEM_HEXAHEDRON_H

#include "pba/aliases.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Hexahedron;

template <>
struct Hexahedron<1>
{
    using AffineBase = Hexahedron<1>;
    
    static int constexpr Order = 1;
    static int constexpr Dims  = 3;
    static int constexpr Nodes = 8;
    static int constexpr Vertices = 8;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,1}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<Scalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = X[1] - 1;
        auto const a2 = X[2] - 1;
        auto const a3 = a1*a2;
        auto const a4 = a2*X[1];
        auto const a5 = a1*X[2];
        auto const a6 = X[1]*X[2];
        Nm[0] = -a0*a3;
        Nm[1] = a3*X[0];
        Nm[2] = a0*a4;
        Nm[3] = -a4*X[0];
        Nm[4] = a0*a5;
        Nm[5] = -a5*X[0];
        Nm[6] = -a0*a6;
        Nm[7] = a6*X[0];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        auto const a0 = X[2] - 1;
        auto const a1 = X[1] - 1;
        auto const a2 = -a1;
        auto const a3 = -a0;
        auto const a4 = X[1]*X[2];
        auto const a5 = X[0] - 1;
        auto const a6 = -a5;
        auto const a7 = X[0]*X[2];
        auto const a8 = X[0]*X[1];
        GNp[0] = a0*a2;
        GNp[1] = a0*a1;
        GNp[2] = a0*X[1];
        GNp[3] = a3*X[1];
        GNp[4] = a1*X[2];
        GNp[5] = a2*X[2];
        GNp[6] = -a4;
        GNp[7] = a4;
        GNp[8] = a0*a6;
        GNp[9] = a0*X[0];
        GNp[10] = a0*a5;
        GNp[11] = a3*X[0];
        GNp[12] = a5*X[2];
        GNp[13] = -a7;
        GNp[14] = a6*X[2];
        GNp[15] = a7;
        GNp[16] = a1*a6;
        GNp[17] = a1*X[0];
        GNp[18] = a5*X[1];
        GNp[19] = -a8;
        GNp[20] = a1*a5;
        GNp[21] = a2*X[0];
        GNp[22] = a6*X[1];
        GNp[23] = a8;
        return GNm;
    }
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, Dims> Jacobian(
        [[maybe_unused]] Vector<Dims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == Nodes);
        auto constexpr DimsOut = Derived::RowsAtCompileTime;
        Matrix<DimsOut, Dims> const J = x * GradN(X);
        return J;
    }
};    

template <>
struct Hexahedron<2>
{
    using AffineBase = Hexahedron<1>;
    
    static int constexpr Order = 2;
    static int constexpr Dims  = 3;
    static int constexpr Nodes = 27;
    static int constexpr Vertices = 8;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,0,1,0,1,1,0,2,1,0,0,2,0,1,2,0,2,2,0,0,0,1,1,0,1,2,0,1,0,1,1,1,1,1,2,1,1,0,2,1,1,2,1,2,2,1,0,0,2,1,0,2,2,0,2,0,1,2,1,1,2,2,1,2,0,2,2,1,2,2,2,2,2}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<Scalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
        auto const a0 = 2*X[0] - 1;
        auto const a1 = 2*X[1] - 1;
        auto const a2 = 2*X[2] - 1;
        auto const a3 = a0*a1*a2;
        auto const a4 = X[0] - 1;
        auto const a5 = X[1] - 1;
        auto const a6 = X[2] - 1;
        auto const a7 = a4*a5*a6;
        auto const a8 = a2*a7;
        auto const a9 = 4*a1;
        auto const a10 = a3*a6;
        auto const a11 = a5*X[0];
        auto const a12 = a0*X[1];
        auto const a13 = 4*a12;
        auto const a14 = X[0]*X[1];
        auto const a15 = 16*a14;
        auto const a16 = a2*a6;
        auto const a17 = a11*a13;
        auto const a18 = a10*X[1];
        auto const a19 = a4*a9;
        auto const a20 = a14*a19;
        auto const a21 = a7*X[2];
        auto const a22 = a0*a9;
        auto const a23 = 16*a21;
        auto const a24 = a6*X[2];
        auto const a25 = a11*a24;
        auto const a26 = a3*X[2];
        auto const a27 = a26*a4;
        auto const a28 = a2*X[2];
        auto const a29 = a11*a28;
        Nm[0] = a3*a7;
        Nm[1] = -a8*a9*X[0];
        Nm[2] = a10*a11;
        Nm[3] = -a13*a8;
        Nm[4] = a15*a8;
        Nm[5] = -a16*a17;
        Nm[6] = a18*a4;
        Nm[7] = -a16*a20;
        Nm[8] = a18*X[0];
        Nm[9] = -a21*a22;
        Nm[10] = a1*a23*X[0];
        Nm[11] = -a22*a25;
        Nm[12] = a12*a23;
        Nm[13] = -64*a14*a21;
        Nm[14] = 16*a12*a25;
        Nm[15] = -a12*a19*a24;
        Nm[16] = a1*a15*a24*a4;
        Nm[17] = -a14*a22*a24;
        Nm[18] = a27*a5;
        Nm[19] = -a19*a29;
        Nm[20] = a11*a26;
        Nm[21] = -a13*a28*a4*a5;
        Nm[22] = 16*a29*a4*X[1];
        Nm[23] = -a17*a28;
        Nm[24] = a27*X[1];
        Nm[25] = -a20*a28;
        Nm[26] = a14*a26;
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        auto const a0 = 4*X[1] - 2;
        auto const a1 = X[0] - 1;
        auto const a2 = 2*X[2];
        auto const a3 = a2 - 1;
        auto const a4 = X[1] - 1;
        auto const a5 = X[2] - 1;
        auto const a6 = a4*a5;
        auto const a7 = a3*a6;
        auto const a8 = a1*a7;
        auto const a9 = 2*X[0];
        auto const a10 = 2*X[1];
        auto const a11 = a10 - 1;
        auto const a12 = a11*(a9 - 1);
        auto const a13 = a12*a7;
        auto const a14 = 4 - 8*X[1];
        auto const a15 = a7*X[0];
        auto const a16 = a6*X[1];
        auto const a17 = 8 - 16*X[2];
        auto const a18 = a1*a17;
        auto const a19 = 4 - 8*X[0];
        auto const a20 = a19*X[1];
        auto const a21 = a20*a7;
        auto const a22 = 32*X[2] - 16;
        auto const a23 = a16*a22;
        auto const a24 = a17*X[0];
        auto const a25 = a5*X[1];
        auto const a26 = a25*a3;
        auto const a27 = a1*a26;
        auto const a28 = a12*a3;
        auto const a29 = a25*a28;
        auto const a30 = a26*X[0];
        auto const a31 = 8 - 16*X[1];
        auto const a32 = a6*X[2];
        auto const a33 = a1*a32;
        auto const a34 = a11*a19;
        auto const a35 = a32*a34;
        auto const a36 = 32*X[1];
        auto const a37 = a36 - 16;
        auto const a38 = a32*X[0];
        auto const a39 = 32*X[0];
        auto const a40 = a39 - 32;
        auto const a41 = a16*X[2];
        auto const a42 = a39 - 16;
        auto const a43 = a41*a42;
        auto const a44 = a25*X[2];
        auto const a45 = a44*X[0];
        auto const a46 = 64 - 64*X[0];
        auto const a47 = a1*a44;
        auto const a48 = a20*X[2];
        auto const a49 = a11*a5;
        auto const a50 = a48*a49;
        auto const a51 = a4*X[2];
        auto const a52 = a3*a51;
        auto const a53 = a1*a52;
        auto const a54 = a28*a51;
        auto const a55 = a52*X[0];
        auto const a56 = a51*X[1];
        auto const a57 = a3*a48;
        auto const a58 = a4*a57;
        auto const a59 = a22*a56;
        auto const a60 = X[1]*X[2];
        auto const a61 = a3*a60;
        auto const a62 = a1*a61;
        auto const a63 = a28*a60;
        auto const a64 = a61*X[0];
        auto const a65 = 4*X[0] - 2;
        auto const a66 = a1*a5;
        auto const a67 = a28*a66;
        auto const a68 = a6*X[0];
        auto const a69 = a3*a66;
        auto const a70 = a14*X[0];
        auto const a71 = a69*a70;
        auto const a72 = a5*X[0];
        auto const a73 = a28*a72;
        auto const a74 = a1*a22;
        auto const a75 = a25*X[0];
        auto const a76 = a20*a3;
        auto const a77 = 8 - 16*X[0];
        auto const a78 = a66*X[2];
        auto const a79 = a34*a78;
        auto const a80 = a37*X[0];
        auto const a81 = a78*a80;
        auto const a82 = a34*a72*X[2];
        auto const a83 = a1*X[2];
        auto const a84 = a28*a83;
        auto const a85 = a51*X[0];
        auto const a86 = a3*a70;
        auto const a87 = a83*a86;
        auto const a88 = a28*X[0];
        auto const a89 = a88*X[2];
        auto const a90 = a60*X[0];
        auto const a91 = a12*a6;
        auto const a92 = a1*a4;
        auto const a93 = a28*a92;
        auto const a94 = a1*a31;
        auto const a95 = a86*a92;
        auto const a96 = a4*a88;
        auto const a97 = a16*a77;
        auto const a98 = a76*a92;
        auto const a99 = a4*X[0];
        auto const a100 = a74*a99*X[1];
        auto const a101 = a16*X[0];
        auto const a102 = a76*a99;
        auto const a103 = a10*a12;
        auto const a104 = a1*X[1];
        auto const a105 = a104*a28;
        auto const a106 = a104*a86;
        auto const a107 = a12*a9;
        auto const a108 = a88*X[1];
        auto const a109 = a1*a34;
        auto const a110 = a1*a80;
        auto const a111 = a1*a37;
        auto const a112 = a1*a42;
        auto const a113 = a56*X[0];
        auto const a114 = a11*a48;
        auto const a115 = a20*a49;
        GNp[0] = a0*a8 + a13;
        GNp[1] = a14*a15 + a14*a8;
        GNp[2] = a0*a15 + a13;
        GNp[3] = a16*a18 + a21;
        GNp[4] = a1*a23 + a23*X[0];
        GNp[5] = a16*a24 + a21;
        GNp[6] = a0*a27 + a29;
        GNp[7] = a14*a27 + a14*a30;
        GNp[8] = a0*a30 + a29;
        GNp[9] = a31*a33 + a35;
        GNp[10] = a33*a37 + a37*a38;
        GNp[11] = a31*a38 + a35;
        GNp[12] = a40*a41 + a43;
        GNp[13] = a41*a46 + a45*(64 - 64*X[1]);
        GNp[14] = a43 + a45*(a36 - 32);
        GNp[15] = a31*a47 + a50;
        GNp[16] = a37*a45 + a37*a47;
        GNp[17] = a31*a45 + a50;
        GNp[18] = a0*a53 + a54;
        GNp[19] = a14*a53 + a14*a55;
        GNp[20] = a0*a55 + a54;
        GNp[21] = a18*a56 + a58;
        GNp[22] = a1*a59 + a59*X[0];
        GNp[23] = a24*a56 + a58;
        GNp[24] = a0*a62 + a63;
        GNp[25] = a14*a62 + a14*a64;
        GNp[26] = a0*a64 + a63;
        GNp[27] = a65*a8 + a67;
        GNp[28] = a18*a68 + a71;
        GNp[29] = a15*a65 + a73;
        GNp[30] = a19*a8 + a20*a69;
        GNp[31] = a68*a74 + a74*a75;
        GNp[32] = a15*a19 + a72*a76;
        GNp[33] = a27*a65 + a67;
        GNp[34] = a18*a75 + a71;
        GNp[35] = a30*a65 + a73;
        GNp[36] = a33*a77 + a79;
        GNp[37] = a38*a40 + a81;
        GNp[38] = a38*a77 + a82;
        GNp[39] = a33*a42 + a42*a47;
        GNp[40] = a38*a46 + a45*a46;
        GNp[41] = a38*a42 + a42*a45;
        GNp[42] = a47*a77 + a79;
        GNp[43] = a40*a45 + a81;
        GNp[44] = a45*a77 + a82;
        GNp[45] = a53*a65 + a84;
        GNp[46] = a18*a85 + a87;
        GNp[47] = a55*a65 + a89;
        GNp[48] = a1*a57 + a19*a53;
        GNp[49] = a74*a85 + a74*a90;
        GNp[50] = a19*a55 + a57*X[0];
        GNp[51] = a62*a65 + a84;
        GNp[52] = a18*a90 + a87;
        GNp[53] = a64*a65 + a89;
        GNp[54] = 2*a1*a91 + a93;
        GNp[55] = a68*a94 + a95;
        GNp[56] = a9*a91 + a96;
        GNp[57] = a1*a97 + a98;
        GNp[58] = a100 + a101*a40;
        GNp[59] = a102 + a97*X[0];
        GNp[60] = a103*a66 + a105;
        GNp[61] = a106 + a75*a94;
        GNp[62] = a107*a25 + a108;
        GNp[63] = a109*a51 + a109*a6;
        GNp[64] = a110*a51 + a111*a68;
        GNp[65] = a34*a68 + a34*a85;
        GNp[66] = a112*a16 + a112*a56;
        GNp[67] = a101*a46 + a113*a46;
        GNp[68] = a101*a42 + a113*a42;
        GNp[69] = a1*a114 + a1*a115;
        GNp[70] = a110*a60 + a111*a75;
        GNp[71] = a114*X[0] + a115*X[0];
        GNp[72] = a12*a2*a92 + a93;
        GNp[73] = a85*a94 + a95;
        GNp[74] = a107*a51 + a96;
        GNp[75] = a1*a56*a77 + a98;
        GNp[76] = a100 + a113*a40;
        GNp[77] = a102 + a113*a77;
        GNp[78] = a103*a83 + a105;
        GNp[79] = a106 + a90*a94;
        GNp[80] = a107*a60 + a108;
        return GNm;
    }
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, Dims> Jacobian(
        [[maybe_unused]] Vector<Dims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == Nodes);
        auto constexpr DimsOut = Derived::RowsAtCompileTime;
        Matrix<DimsOut, Dims> const J = x * GradN(X);
        return J;
    }
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_HEXAHEDRON_H
