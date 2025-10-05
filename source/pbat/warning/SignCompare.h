#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)
    #pragma warning(disable : 4388)
#endif
