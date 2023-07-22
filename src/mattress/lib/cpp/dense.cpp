#include "Mattress.h"

template<typename T>
inline uintptr_t initialize_dense_matrix_core(int nr, int nc, const T* ptr, bool byrow) { 
    tatami::ArrayView<T> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    if (byrow) {
        return reinterpret_cast<uintptr_t>(new Mattress(new tatami::DenseRowMatrix<double, int, decltype(view)>(nr, nc, view)));
    } else {
        return reinterpret_cast<uintptr_t>(new Mattress(new tatami::DenseColumnMatrix<double, int, decltype(view)>(nr, nc, view)));
    }
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const double* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const float* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const signed long long* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const int64_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const uint64_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const int32_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const uint32_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const int16_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const uint16_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const int8_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}

inline uintptr_t initialize_dense_matrix(int nr, int nc, const uint8_t* ptr, bool byrow) { 
    return initialize_dense_matrix_core(nr, nc, ptr, byrow);
}
