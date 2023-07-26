#include "Mattress.h"

extern "C" {

int py_extract_nrow(Mattress* mat) {
    return mat->ptr->nrow();
}

int py_extract_ncol(Mattress* mat) {
    return mat->ptr->ncol();
}

int py_extract_sparse(Mattress* mat) {
    return mat->ptr->sparse();
}

void py_extract_row(Mattress* mat, int r, void* output) {
    if (!mat->byrow) {
        mat->byrow = mat->ptr->dense_row();
    }
    mat->byrow->fetch_copy(r, reinterpret_cast<double*>(output));
}

void py_extract_column(Mattress* mat, int c, void* output) {
    if (!mat->bycol) {
        mat->bycol = mat->ptr->dense_column();
    }
    mat->bycol->fetch_copy(c, reinterpret_cast<double*>(output));
}

void py_free_mat(Mattress* mat) {
    delete mat;
}

}
