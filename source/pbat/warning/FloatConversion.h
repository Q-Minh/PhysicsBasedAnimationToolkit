#ifndef PBAT_WARNING_FLOATCONVERSION_H
#define PBAT_WARNING_FLOATCONVERSION_H

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wfloat-conversion"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic ignored "-Wfloat-conversion"
#elif defined(_MSC_VER)
    // C4244 is generally possible loss of data, i.e. also considers float to int
    #pragma warning(disable : 4244)
    // C4305 'conversion': truncation from 'type1' to 'type2'
    #pragma warning(disable : 4305)
#endif

#endif // PBAT_WARNING_FLOATCONVERSION_H
