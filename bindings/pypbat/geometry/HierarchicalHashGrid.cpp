#include "HierarchicalHashGrid.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/HierarchicalHashGrid.h>
#include <string>
#include <utility>
#include <vector>

namespace pbat::py::geometry {

void BindHierarchicalHashGrid(nanobind::module_& m)
{
    namespace nb = nanobind;
    pbat::common::ForTypes<
        pbat::geometry::HierarchicalHashGrid<2>,
        pbat::geometry::HierarchicalHashGrid<3>>([&m]<class HashGridType>() {
        std::string const className =
            std::string("HierarchicalHashGrid") + std::to_string(HashGridType::kDims);
        nb::class_<HashGridType>(m, className.data())
            .def(nb::init<>())
            .def(
                nb::init<typename HashGridType::IndexType, typename HashGridType::IndexType>(),
                nb::arg("n_primitives"),
                nb::arg("n_buckets") = 0,
                "Construct a HierarchicalHashGrid with memory reserved for a specific number of "
                "primitives. Optionally, the number of buckets to allocate can be directly "
                "specified, overriding the default number of buckets allocated for n_primitives "
                "primitives.\n\n"
                "Args:\n"
                "   n_primitives (int): Number of primitives to reserve space for.\n"
                "   n_buckets (int): Number of buckets to reserve space for.\n")
            .def(
                "configure",
                &HashGridType::Configure,
                nb::arg("n_primitives"),
                nb::arg("n_buckets") = 0,
                "Reserve memory for a specific number of primitives. Optionally, the number of "
                "buckets to allocate can be directly specified, overriding the default number of "
                "buckets allocated for n_primitives primitives.\n\n"
                "Args:\n"
                "   n_primitives (int): Number of primitives to reserve space for.\n"
                "   n_buckets (int): Number of buckets to reserve space for.\n")
            .def(
                "construct",
                [](HashGridType& self, nb::DRef<MatrixX const> L, nb::DRef<MatrixX const> U) {
                    self.Construct(
                        L.topRows<HashGridType::kDims>(),
                        U.topRows<HashGridType::kDims>());
                },
                nb::arg("L"),
                nb::arg("U"),
                "Construct a HierarchicalHashGrid from lower and upper bounds of input "
                "axis-aligned bounding boxes (aabbs).\n\n"
                "Args:\n"
                "   L (numpy.ndarray): `|# dims| x |# aabbs|` lower bounds of the aabbs.\n"
                "   U (numpy.ndarray): `|# dims| x |# aabbs|` upper bounds of the aabbs.\n")
            .def(
                "broad_phase",
                [](HashGridType& self,
                   nb::DRef<MatrixX const> X,
                   std::size_t nExpectedPrimitivesPerCell) {
                    std::vector<std::pair<Index, Index>> broadPhasePairs{};
                    auto constexpr kDims      = HashGridType::kDims;
                    auto const nCellsPerVisit = kDims == 3 ? 8 : 4;
                    auto const nQueries       = static_cast<std::size_t>(X.cols());
                    broadPhasePairs.reserve(nExpectedPrimitivesPerCell * nCellsPerVisit * nQueries);
                    self.BroadPhase(X.topRows<HashGridType::kDims>(), [&](Index q, Index p) {
                        broadPhasePairs.push_back({q, p});
                    });
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
                "p).\n")
            .def(
                "broad_phase",
                [](HashGridType& self,
                   nb::DRef<MatrixX const> L,
                   nb::DRef<MatrixX const> U,
                   std::size_t nExpectedPrimitivesPerCell) {
                    std::vector<std::pair<Index, Index>> broadPhasePairs{};
                    auto constexpr kDims      = HashGridType::kDims;
                    auto const nCellsPerVisit = kDims == 3 ? 8 : 4;
                    auto const nQueries       = static_cast<std::size_t>(L.cols());
                    broadPhasePairs.reserve(nExpectedPrimitivesPerCell * nCellsPerVisit * nQueries);
                    self.BroadPhase(
                        L.topRows<HashGridType::kDims>(),
                        U.topRows<HashGridType::kDims>(),
                        [&](Index q, Index p) { broadPhasePairs.push_back({q, p}); });
                    return broadPhasePairs;
                },
                nb::arg("L"),
                nb::arg("U"),
                nb::arg("n_expected_primitives_per_cell") = 1,
                "Find all primitives whose cell overlaps with the input axis-aligned bounding "
                "boxes (aabbs).\n\n"
                "Args:\n"
                "   L (numpy.ndarray): `|# dims| x |# aabbs|` lower bounds of the aabbs.\n"
                "   U (numpy.ndarray): `|# dims| x |# aabbs|` upper bounds of the aabbs.\n"
                "   n_expected_primitives_per_cell (int): Expected number of primitives per cell. "
                "Defaults to 1.\n"
                "Returns:\n"
                "   List[Tuple[int, int]]: List of broad phase query-aabb to primitive aabb pairs "
                "(q, p).\n")
            .def(
                "broad_phase",
                [](HashGridType& self,
                   nb::DRef<MatrixX const> L,
                   nb::DRef<MatrixX const> U,
                   nb::DRef<MatrixX const> X,
                   std::size_t nExpectedPrimitivesPerCell) {
                    std::vector<std::pair<Index, Index>> broadPhasePairs{};
                    auto constexpr kDims      = HashGridType::kDims;
                    auto const nCellsPerVisit = kDims == 3 ? 8 : 4;
                    auto const nQueries       = static_cast<std::size_t>(X.cols());
                    broadPhasePairs.reserve(nExpectedPrimitivesPerCell * nCellsPerVisit * nQueries);
                    self.BroadPhase(
                        L.topRows<HashGridType::kDims>(),
                        U.topRows<HashGridType::kDims>(),
                        X.topRows<HashGridType::kDims>(),
                        [&](Index q, Index p) { broadPhasePairs.push_back({q, p}); });
                    return broadPhasePairs;
                },
                nb::arg("L"),
                nb::arg("U"),
                nb::arg("X"),
                nb::arg("n_expected_primitives_per_cell") = 1,
                "Find all primitives whose cell overlaps with points `X`.\n\n"
                "Args:\n"
                "   L (numpy.ndarray): `|# dims| x |# aabbs|` lower bounds of the primitive aabbs "
                "used to construct the HierarchicalHashGrid.\n"
                "   U (numpy.ndarray): `|# dims| x |# aabbs|` upper bounds of the primitive aabbs "
                "used to construct the HierarchicalHashGrid.\n"
                "   X (numpy.ndarray): `|# dims| x |# query points|` matrix of query points.\n"
                "   n_expected_primitives_per_cell (int): Expected number of primitives per cell. "
                "Defaults to 1.\n"
                "Returns:\n"
                "   List[Tuple[int, int]]: List of broad phase query-point to primitive pairs (q, "
                "p).\n")
            .def(
                "broad_phase",
                [](HashGridType& self,
                   nb::DRef<MatrixX const> LP,
                   nb::DRef<MatrixX const> UP,
                   nb::DRef<MatrixX const> LQ,
                   nb::DRef<MatrixX const> UQ,
                   std::size_t nExpectedPrimitivesPerCell) {
                    std::vector<std::pair<Index, Index>> broadPhasePairs{};
                    auto constexpr kDims      = HashGridType::kDims;
                    auto const nCellsPerVisit = kDims == 3 ? 8 : 4;
                    auto const nQueries       = static_cast<std::size_t>(LQ.cols());
                    broadPhasePairs.reserve(nExpectedPrimitivesPerCell * nCellsPerVisit * nQueries);
                    self.BroadPhase(
                        LP.topRows<HashGridType::kDims>(),
                        UP.topRows<HashGridType::kDims>(),
                        LQ.topRows<HashGridType::kDims>(),
                        UQ.topRows<HashGridType::kDims>(),
                        [&](Index q, Index p) { broadPhasePairs.push_back({q, p}); });
                    return broadPhasePairs;
                },
                nb::arg("LP"),
                nb::arg("UP"),
                nb::arg("LQ"),
                nb::arg("UQ"),
                nb::arg("n_expected_primitives_per_cell") = 1,
                "Find all primitives whose cell overlaps with the input axis-aligned bounding "
                "boxes (aabbs).\n\n"
                "Args:\n"
                "   LP (numpy.ndarray): `|# dims| x |# aabbs|` lower bounds of the primitive "
                "aabbs.\n"
                "   UP (numpy.ndarray): `|# dims| x |# aabbs|` upper bounds of the primitive "
                "aabbs.\n"
                "   LQ (numpy.ndarray): `|# dims| x |# query aabbs|` lower bounds of the query "
                "aabbs.\n"
                "   UQ (numpy.ndarray): `|# dims| x |# query aabbs|` upper bounds of the query "
                "aabbs.\n"
                "   n_expected_primitives_per_cell (int): Expected number of primitives per cell. "
                "Defaults to 1.\n"
                "Returns:\n"
                "   List[Tuple[int, int]]: List of broad phase query-aabb to primitive aabb pairs "
                "(q, p).\n");
    });
}

} // namespace pbat::py::geometry