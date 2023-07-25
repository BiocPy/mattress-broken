#include "Mattress.h"

template<typename T>
Mattress* initialize_dense_matrix(int nr, int nc, const T* ptr, bool byrow) { 
    std::cout << ptr[0] << "\t" << ptr[nr * nc - 1] << std::endl;
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    if (byrow) {
        return new Mattress(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view));
    } else {
        return new Mattress(new tatami::DenseColumnMatrix<double, int, decltype(view)>(nr, nc, view));
    }
}

extern "C" {

void* py_initialize_dense_matrix_double(int nr, int nc, void* ptr, char byrow) {
    auto ptr2 = initialize_dense_matrix(nr, nc, reinterpret_cast<const double*>(ptr), byrow);
    std::cout << nr << "\t" << nc << "\t" << (size_t)ptr << "\t" << (size_t)ptr2 << std::endl;
    return reinterpret_cast<void*>(ptr2);
}

}

