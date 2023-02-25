// Copyright 2023 Ezhev Mikhail

#include <gtest/gtest.h>
#include "./matvec_hor.h"
#include <gtest-mpi-listener.hpp>

TEST(MatOnVecHor_MPI, Test1_2x2) {
    int m = 2, n = 2;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Mat = RandVector(m * n);
    std::vector<int> Vec = RandVector(n);

    std::vector<int> Res = MultiplyParallel(Mat, m, n, Vec);

    //Check
    if (rank == 0) EXPECT_EQ(Res, MultiplySequential(Mat, m, n, Vec));
}

TEST(MatOnVecHor_MPI, Test1_2x3) {
    int m = 2, n = 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Mat = RandVector(m * n);
    std::vector<int> Vec = RandVector(n);

    std::vector<int> Res = MultiplyParallel(Mat, m, n, Vec);

    //Check
    if (rank == 0) EXPECT_EQ(Res, MultiplySequential(Mat, m, n, Vec));
}

TEST(MatOnVecHor_MPI, Test1_3x3) {
    int m = 3, n = 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Mat = RandVector(m * n);
    std::vector<int> Vec = RandVector(n);

    std::vector<int> Res = MultiplyParallel(Mat, m, n, Vec);

    //Check
    if (rank == 0) EXPECT_EQ(Res, MultiplySequential(Mat, m, n, Vec));
}

TEST(MatOnVecHor_MPI, Test1_25x25) {
    int m = 25, n = 25;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Mat = RandVector(m * n);
    std::vector<int> Vec = RandVector(n);

    std::vector<int> Res = MultiplyParallel(Mat, m, n, Vec);

    //Check
    if (rank == 0) EXPECT_EQ(Res, MultiplySequential(Mat, m, n, Vec));
}

TEST(MatOnVecHor_MPI, Test1_95x95) {
    int m = 95, n = 95;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> Mat = RandVector(m * n);
    std::vector<int> Vec = RandVector(n);

    std::vector<int> Res = MultiplyParallel(Mat, m, n, Vec);

    //Check
    if (rank == 0) EXPECT_EQ(Res, MultiplySequential(Mat, m, n, Vec));
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
