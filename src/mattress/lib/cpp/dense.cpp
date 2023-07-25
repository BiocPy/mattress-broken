#include "Mattress.h"

template<typename T>
inline Mattress* initialize_dense_matrix(int nr, int nc, const T* ptr, bool byrow) { 
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    if (byrow) {
        return new Mattress(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view));
    } else {
        return new Mattress(new tatami::DenseColumnMatrix<double, int, decltype(view)>(nr, nc, view));
    }
}
