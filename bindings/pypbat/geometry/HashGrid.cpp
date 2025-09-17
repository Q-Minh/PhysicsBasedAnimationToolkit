#include "HashGrid.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/HashGrid.h>
#include <string>
#include <utility>
#include <vector>

namespace pbat::py::geometry {

void BindHashGrid(nanobind::module_& m)
{
    namespace nb = nanobind;
    pbat::common::ForValues<2, 3>([&m]<auto kDims>() {
        std::string const className = []() {
            if constexpr (kDims == 2)
                return "HashGrid2D";
            if constexpr (kDims == 3)
                return "HashGrid3D";
        }();

        using HashGridType = pbat::geometry::HashGrid<kDims>;
        nb::class_<HashGridType>(m, className.data())
            .def(nb::init<>())
            .def(
                nb::init<typename HashGridType::ScalarType, typename HashGridType::IndexType>(),
                nb::arg("cell_size"),
                nb::arg("n_buckets"),
                "Construct a HashGrid with a specific cell size and number of buckets.\n\n"
                "Args:\n"
                "   cell_size (float): Uniform grid cell extents\n"
                "   n_buckets (int): Number of buckets to allocate for the hash table.\n")
            .def(
                "configure",
                &HashGridType::Configure,
                nb::arg("cell_size"),
                nb::arg("n_buckets"),
                "Construct a HashGrid with a specific cell size and number of buckets.\n\n"
                "Args:\n"
                "   cell_size (float): Uniform grid cell extents\n"
                "   n_buckets (int): Number of buckets to allocate for the hash table.\n")
            .def(
                "reserve",
                &HashGridType::Reserve,
                nb::arg("n_primitives"),
                "Reserve space for a specific number of primitives in the hash grid.\n\n"
                "Args:\n"
                "   n_primitives (int): Number of primitives to reserve space for.\n")
            .def(
                "construct",
                [](HashGridType& self, nb::DRef<MatrixX const> L, nb::DRef<MatrixX const> U) {
                    auto fHash = pbat::geometry::HashByXorOfPrimeMultiples<Index>();
                    self.Construct(
                        L.topRows<HashGridType::kDims>(),
                        U.topRows<HashGridType::kDims>(),
                        fHash);
                },
                nb::arg("L"),
                nb::arg("U"),
                "Construct a HashGrid from lower and upper bounds of input axis-aligned bounding "
                "boxes "
                "(aabbs).\n\n"
                "Args:\n"
                "   L (numpy.ndarray): `|# dims| x |# aabbs|` lower bounds of the aabbs.\n"
                "   U (numpy.ndarray): `|# dims| x |# aabbs|` upper bounds of the aabbs.\n")
            .def(
                "construct",
                [](HashGridType& self, nb::DRef<MatrixX const> X) {
                    auto fHash = pbat::geometry::HashByXorOfPrimeMultiples<Index>();
                    self.Construct(X.topRows<HashGridType::kDims>(), fHash);
                },
                nb::arg("X"),
                "Construct a HashGrid from points.\n\n"
                "Args:\n"
                "   X (numpy.ndarray): `|# dims| x |# points|` points.\n")
            .def(
                "broad_phase",
                [](HashGridType& self,
                   nb::DRef<MatrixX const> X,
                   std::size_t nExpectedPrimitivesPerCell) {
                    std::vector<std::pair<Index, Index>> broadPhasePairs{};
                    auto constexpr kDims      = HashGridType::kDims;
                    auto const nCellsPerVisit = kDims == 3 ? 27 : 9;
                    auto const nQueries       = static_cast<std::size_t>(X.cols());
                    broadPhasePairs.reserve(nExpectedPrimitivesPerCell * nCellsPerVisit * nQueries);
                    auto fHash = pbat::geometry::HashByXorOfPrimeMultiples<Index>();
                    self.BroadPhase(
                        X.topRows<HashGridType::kDims>(),
                        [&](Index q, Index p) { broadPhasePairs.push_back({q, p}); },
                        fHash);
                    return broadPhasePairs;
                },
                nb::arg("X"),
                nb::arg("n_expected_primitives_per_cell") = 1,
                "Find all primitives whose cell overlaps with points `X`.\n\n"
                "Args:\n"
                "   X (numpy.ndarray): `|# dims| x |# query points|` matrix of query points.\n"
                "   n_expected_primitives_per_cell (int): Expected number of primitives per cell. "
                "Defaults to 1.\n"
                "Returns:\n"
                "   List[Tuple[int, int]]: List of broad phase query-point to primitive pairs (q, "
                "p).\n");
    });
}

} // namespace pbat::py::geometry