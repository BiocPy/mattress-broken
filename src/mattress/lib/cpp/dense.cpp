#include "Mattress.h"

template<typename T>
inline uintptr_t initialize_dense_matrix(int nr, int nc, const T* ptr, bool byrow) { 
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    if (byrow) {
        return reinterpret_cast<uintptr_t>(new Mattress(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view)));
    } else {
        return reinterpret_cast<uintptr_t>(new Mattress(new tatami::DenseColumnMatrix<double, int, decltype(view)>(nr, nc, view)));
    }
}
