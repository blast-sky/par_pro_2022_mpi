// Copyright 2023 Okulov Stepan
#include <gtest/gtest.h>
#include "./gaussjordan.h"
#include <gtest-mpi-listener.hpp>


void assertNear(const Vector& expected, const Vector& res, int rank) {
    if (rank == 0) {
        for (int i = 0; i < expected.size(); i++) {
            ASSERT_NEAR(expected[i], res[i], 0.001);
        }
    }
}

TEST(Gauss_Jordan, Matrix_1x1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A = {
        {2},
    };

    Vector b = {
        4
    };

    Vector expected = {
        2
    };

    Vector res = getGauseJordanPar(A, b);

    assertNear(expected, res, rank);
}

TEST(Gauss_Jordan, Matrix_2x2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A = {
        {1, 2},
        {3, 4}
    };

    Vector b = {
        4, 10
    };

    Vector expected = {
        2, 1
    };

    Vector res = getGauseJordanPar(A, b);

    assertNear(expected, res, rank);
}

TEST(Gauss_Jordan, Matrix_3x3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A = {
        {1, 2, 3},
        {3, 4, -5},
        {-1, 1, -1}
    };

    Vector b = {
        2, -6, -3
    };

    Vector expected = {
        1, -1, 1
    };

    Vector res = getGauseJordanPar(A, b);

    assertNear(expected, res, rank);
}

TEST(Gauss_Jordan, Matrix_4x4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A = {
        {1, 2, 3, 4},
        {3, 4, -5, 1},
        {-1, 1, -1, -10},
        {2, -4, 6, -8}
    };

    Vector b = {
        10, 3, -11, -4
    };

    Vector expected = {
        1, 1, 1, 1
    };

    Vector res = getGauseJordanPar(A, b);

    assertNear(expected, res, rank);
}

TEST(Gauss_Jordan, Matrix_5x5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A = {
        {1, 2, 3, 4, 5},
        {3, 4, -5, 1, 6},
        {-1, 1, -1, -10, 7},
        {2, -4, 6, -8, 8},
        {20, 30, 40, 50, 60}
    };

    Vector b = {
        0, -9, -25, -20, 20
    };

    Vector expected = {
        1, 1, 1, 1, -2
    };

    Vector res = getGauseJordanPar(A, b);

    assertNear(expected, res, rank);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
