#ifndef PBAT_WARNING_FLOATCONVERSION_H
#define PBAT_WARNING_FLOATCONVERSION_H

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wfloat-conversion"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic ignored "-Wfloat-conversion"
#elif defined(_MSC_VER)
    // I think MSVC is less granular on this and has C4244 which is generally about converting
    // between small and large types
#endif

#endif // PBAT_WARNING_FLOATCONVERSION_H
