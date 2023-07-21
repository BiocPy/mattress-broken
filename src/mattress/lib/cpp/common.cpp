#include "tatami/tatami.hpp"

inline int extract_nrow(uintptr_t ptr) {
    return reinterpret_cast<tatami::NumericMatrix*>(ptr)->nrow();
}

inline int extract_ncol(uintptr_t ptr) {
    return reinterpret_cast<tatami::NumericMatrix*>(ptr)->ncol();
}

inline bool extract_sparse(uintptr_t ptr) {
    return reinterpret_cast<tatami::NumericMatrix*>(ptr)->sparse();
}

inline void extract_row(uintptr_t ptr, int r, double* output) {
    reinterpret_cast<tatami::NumericMatrix*>(ptr)->dense_row()->fetch_copy(r, output);
}

inline void extract_column(uintptr_t& ptr, int c, double* output) {
    reinterpret_cast<tatami::NumericMatrix*>(ptr)->dense_column()->fetch_copy(c, output);
}
