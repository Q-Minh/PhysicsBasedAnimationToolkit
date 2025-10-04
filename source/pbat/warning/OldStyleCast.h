#ifndef PBAT_WARNING_OLDSTYLECAST_H
#define PBAT_WARNING_OLDSTYLECAST_H

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic ignored "-Wold-style-cast"
#elif defined(_MSC_VER)
    // Not sure what the MSVC equivalent is
#endif

#endif // PBAT_WARNING_OLDSTYLECAST_H
