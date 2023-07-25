#include "Mattress.h"

extern "C" {

int py_extract_nrow(void* raw) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    return mat->ptr->nrow();
}

int py_extract_ncol(void* raw) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    return mat->ptr->ncol();
}

bool py_extract_sparse(void* raw) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    return mat->ptr->sparse();
}

void py_extract_row(void* raw, int r, void* output) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    if (!mat->byrow) {
        mat->byrow = mat->ptr->dense_row();
    }
    mat->byrow->fetch_copy(r, reinterpret_cast<double*>(output));
}

void py_extract_column(void* raw, int c, void* output) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    if (!mat->bycol) {
        mat->bycol = mat->ptr->dense_column();
    }
    mat->bycol->fetch_copy(c, reinterpret_cast<double*>(output));
}

void py_free_mat(void* raw) {
    Mattress* mat = reinterpret_cast<Mattress*>(raw);
    delete mat;
}

}
