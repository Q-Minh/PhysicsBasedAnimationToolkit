/**
 * @file Modulo.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Modulo function
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_MODULO_H
#define PBAT_COMMON_MODULO_H

namespace pbat::common {

auto Modulo(auto a, auto b)
{
    return (a % b + b) % b;
}

} // namespace pbat::common

#endif // PBAT_COMMON_MODULO_H
