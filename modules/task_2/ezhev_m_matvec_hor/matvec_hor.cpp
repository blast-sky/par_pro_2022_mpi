// Copyright 2023 Ezhev Mikhail

#include "../../../modules/task_2/ezhev_m_matvec_hor/matvec_hor.h"

#include <algorithm>
#include <ctime>
#include <vector>

std::vector<int> RandVector(int n) {
    std::mt19937 gen;
    std::vector<int> vec(n);
    std::default_random_engine rand_val;
    gen.seed(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; i++) {
        vec[i] = gen() % 10;
    }
    return vec;
}

std::vector<int> MultiplySequential(const std::vector<int>& mat, int row,
                                    int col, const std::vector<int>& vector) {
    std::vector<int> result(row, 0);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result[i] += mat[i * col + j] * vector[j];
        }
    }

    return result;
}

std::vector<int> MultiplyParallel(const std::vector<int>& matr, int row,
                                  int col, const std::vector<int>& vect) {
    int size;
    int rank;
    int ost;
    int delta;
    int flag = 1;
    int S = vect.size();
    if ((S != col) || (row <= 0) || (col <= 0)) throw "incorrect dimensions";

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ost = row % size;
    if (row < size) {
        size = 1;
        ost = 0;
        flag = 0;
    }
    delta = row / size;

    if (rank == 0) {
        int num;
        for (int i = col * (delta + ost); i <= col * (row - delta);
             i += col * delta) {
            num = (i / col - ost) / delta;
            MPI_Send(&matr[i], col * delta, MPI_INT, num, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Status status;
    std::vector<int> mat(col * delta, 0);
    std::vector<int> tmp(delta);
    std::vector<int> res(row);

    if (rank == 0) {
        tmp.resize(delta + ost);
        for (int i = 0; i < col * (delta + ost); i += col) {
            for (int j = 0; j < col; j++) tmp[i / col] += matr[i + j] * vect[j];
        }
    } else {
        if (flag == 1) {
            int n = col * delta;
            MPI_Recv(&mat[0], n, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < col * delta; i += col) {
                for (int j = 0; j < col; j++)
                    tmp[i / col] += mat[i + j] * vect[j];
            }
        }
    }

    if (flag == 1) {
        MPI_Gather(&tmp[0], delta, MPI_INT, &res[ost], delta, MPI_INT, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i = 0; i < delta + ost; i++) res[i] = tmp[i];
        }
        return res;
    } else {
        if (rank == 0) {
            return tmp;
        } else {
            std::vector<int> err = {-111};
            return err;
        }
    }
}
