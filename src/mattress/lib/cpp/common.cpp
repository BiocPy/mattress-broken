#include "Mattress.h"

inline int extract_nrow(const Mattress* mat) {
    return mat->ptr->nrow();
}

inline int extract_ncol(const Mattress* mat) {
    return mat->ptr->ncol();
}

inline bool extract_sparse(const Mattress* mat) {
    return mat->ptr->sparse();
}

inline void extract_row(Mattress* mat, int r, double* output) {
    mat->row()->fetch_copy(r, output);
}

inline void extract_column(Mattress* mat, int c, double* output) {
    mat->column()->fetch_copy(c, output);
}

inline void free_mat(Mattress* mat) {
    delete mat;
}
