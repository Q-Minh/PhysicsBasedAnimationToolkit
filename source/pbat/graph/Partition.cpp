#include "Partition.h"

#include "pbat/common/Eigen.h"

#ifdef PBAT_USE_METIS
    #include <metis.h>
#endif // PBAT_USE_METIS

#include <exception>
#include <string>

namespace pbat {
namespace graph {

#ifdef PBAT_USE_METIS
std::vector<Index> Partition(
    std::vector<Index> const& ptr,
    std::vector<Index> const& adj,
    std::vector<Index> const& wadj,
    std::size_t nPartitions)
{
    // Use Metis to partition the graph
    auto xadj       = common::ToEigen(ptr).cast<idx_t>().eval();
    auto adjncy     = common::ToEigen(adj).cast<idx_t>().eval();
    auto adjwgt     = common::ToEigen(wadj).cast<idx_t>().eval();
    idx_t nVertices = static_cast<idx_t>(ptr.size() - 1);
    idx_t nBalancingConstraints(1);
    idx_t nparts = static_cast<idx_t>(nPartitions);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE]     = METIS_PTYPE_KWAY;
    options[METIS_OPTION_NUMBERING] = 0;
    idx_t objval(-1);
    std::vector<idx_t> part(static_cast<std::size_t>(nVertices), idx_t(-1));
    int const ec = METIS_PartGraphKway(
        &nVertices,
        &nBalancingConstraints,
        xadj.data(),
        adjncy.data(),
        NULL,
        NULL,
        adjwgt.data(),
        &nparts,
        NULL,
        NULL,
        options,
        &objval,
        part.data());
    if (ec != METIS_OK)
    {
        std::string const what =
            (ec == METIS_ERROR_INPUT)  ? "Invalid input to METIS" :
            (ec == METIS_ERROR_MEMORY) ? "METIS could not allocate required memory" :
                                         "Unknown error from METIS";
        throw std::invalid_argument(what);
    }
    auto pbatPart = common::ToEigen(part).cast<Index>();
    return std::vector<Index>{pbatPart.begin(), pbatPart.end()};
}
#endif // PBAT_USE_METIS

} // namespace graph
} // namespace pbat