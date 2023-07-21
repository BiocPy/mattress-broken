#include "tatami/tatami.hpp"
#include <iostream>

// Interface methods to tatami objects.

template<typename T>
inline std::shared_ptr<tatami::NumericMatrix> initialize_dense_matrix_core(int nr, int nc, const T* ptr, bool byrow) { 
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    if (byrow) {
        return std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view));
    } else {
        return std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseColumnMatrix<double, int, decltype(view)>(nr, nc, view));
    }
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const double* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const float* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const int64_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const uint64_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const int32_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const uint32_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const int16_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const uint16_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const int8_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline std::shared_ptr<tatami::NumericMatirx> initialize_dense_matrix(int nr, int nc, const uint8_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}
