#ifndef MATTRESS_COMMON_H
#define MATTRESS_COMMON_H

#include "tatami/tatami.hpp"

struct Mattress {
    Mattress(tatami::NumericMatrix* p) : ptr(p) {}
    Mattress(std::shared_ptr<tatami::NumericMatrix> p) : ptr(std::move(p)) {}
    std::shared_ptr<tatami::NumericMatrix> ptr;

public:
    std::unique_ptr<tatami::FullDenseExtractor<double, int> >& row() {
        if (!byrow) {
            byrow = ptr->dense_row();
        }
        return byrow;
    }

    std::unique_ptr<tatami::FullDenseExtractor<double, int> >& column() {
        if (!bycol) {
            bycol = ptr->dense_column();
        }
        return bycol;
    }

private:
    std::unique_ptr<tatami::FullDenseExtractor<double, int> > byrow, bycol;
};

#endif
