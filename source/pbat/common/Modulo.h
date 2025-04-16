#ifndef PBAT_COMMON_MODULO_H
#define PBAT_COMMON_MODULO_H

namespace pbat::common {

auto Modulo(auto a, auto b)
{
    return (a % b + b) % b;
}

} // namespace pbat::common

#endif // PBAT_COMMON_MODULO_H
