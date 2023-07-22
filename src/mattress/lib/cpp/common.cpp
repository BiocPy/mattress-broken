#include "Mattress.h"

inline int extract_nrow(uintptr_t mat) {
    return reinterpret_cast<Mattress*>(mat)->ptr->nrow();
}

inline int extract_ncol(uintptr_t mat) {
    return reinterpret_cast<Mattress*>(mat)->ptr->ncol();
}

inline bool extract_sparse(uintptr_t mat) {
    return reinterpret_cast<Mattress*>(mat)->ptr->sparse();
}

inline void extract_row(uintptr_t mat, int r, double* output) {
    reinterpret_cast<Mattress*>(mat)->row()->fetch_copy(r, output);
}

inline void extract_column(uintptr_t mat, int c, double* output) {
    reinterpret_cast<Mattress*>(mat)->column()->fetch_copy(c, output);
}
