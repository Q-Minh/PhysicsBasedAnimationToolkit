#include "HessianProduct.h"

namespace pbat::sim::algorithm::newton {

void Hessian::AllocateContacts(std::size_t nContacts)
{
    HCij.reserve(nContacts * kDims * kDims * kContactInds * kContactInds);
}

void Hessian::SetSparsityPattern(CSCMatrix const& HS)
{
    HNC = HS;
    HC.resize(HS.rows(), HS.cols());
    diag.resize(HS.cols());
    for (auto j = 0; j < HNC.outerSize(); ++j)
    {
        for (CSCMatrix::InnerIterator it(HNC, j); it; ++it)
        {
            auto const i = it.row();
            if (i == j)
                diag(i) = it.index();
        }
    }
}

void Hessian::ConstructContactHessian()
{
    HC.setFromTriplets(HCij.begin(), HCij.end());
}

HessianOperator::HessianOperator(Hessian* data) : mData(data) {}

} // namespace pbat::sim::algorithm::newton