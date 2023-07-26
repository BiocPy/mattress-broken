#include "Mattress.h"
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

template<typename T>
Mattress* initialize_dense_matrix(int nr, int nc, const T* ptr) { 
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    return new Mattress(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view));
}

extern "C" {

Mattress* py_initialize_dense_matrix(int nr, int nc, const char* type, void* ptr) {
    if (std::strcmp(type, "float64") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<double*>(ptr));

    } else if (std::strcmp(type, "float32") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<float*>(ptr));

    } else if (std::strcmp(type, "int64") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<int64_t*>(ptr));

    } else if (std::strcmp(type, "int32") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<int32_t*>(ptr));

    } else if (std::strcmp(type, "int16") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<int16_t*>(ptr));

    } else if (std::strcmp(type, "int8") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<int8_t*>(ptr));

    } else if (std::strcmp(type, "uint64") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<uint64_t*>(ptr));

    } else if (std::strcmp(type, "uint32") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<uint32_t*>(ptr));

    } else if (std::strcmp(type, "uint16") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<uint16_t*>(ptr));

    } else if (std::strcmp(type, "uint8") == 0) {
        return initialize_dense_matrix(nr, nc, reinterpret_cast<uint8_t*>(ptr));
    }

    throw std::runtime_error("unrecognized array type '" + std::string(type) + "' for dense matrix initialization");
    return NULL;
}

}

