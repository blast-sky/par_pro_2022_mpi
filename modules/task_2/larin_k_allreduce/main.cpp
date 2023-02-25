// Copyright 2022 Larin Konstantin
#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>

#include "allreduce.h"

int64_t pw(int64_t a, int64_t pw) {
  int64_t res = 1;
  while (pw) {
    if (pw & 1) {
      res *= a;
    }

    a *= a;
    pw >>= 1;
  }

  return res;
}

TEST(allreduce, array_sum) {
  int rank_cnt;
  MPI_Comm_size(MPI_COMM_WORLD, &rank_cnt);

  constexpr size_t size = 1000;
  int* arr = new int[size];
  int* result = new int[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = i;
  }

  allreduce(arr, result, int(size), MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(result[i], i * rank_cnt);
  }

  delete[] arr;
  delete[] result;
}

TEST(allreduce, array_mul) {
  int rank_cnt;
  MPI_Comm_size(MPI_COMM_WORLD, &rank_cnt);

  constexpr size_t size = 64;
  int64_t* arr = new int64_t[size];
  int64_t* result = new int64_t[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = i;
  }

  allreduce(arr, result, int(size), MPI_INT64_T, MPI_PROD, MPI_COMM_WORLD);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(result[i], pw(i, rank_cnt));
  }

  delete[] arr;
  delete[] result;
}

TEST(allreduce, min) {
  int rank_cnt;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &rank_cnt);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr size_t size = 64;
  int64_t* arr = new int64_t[size];
  int64_t* result = new int64_t[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = i + rank;
  }

  allreduce(arr, result, int(size), MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(result[i], i);
  }

  delete[] arr;
  delete[] result;
}

TEST(allreduce, max) {
  int rank_cnt;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &rank_cnt);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr size_t size = 64;
  int64_t* arr = new int64_t[size];
  int64_t* result = new int64_t[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = i + rank;
  }

  allreduce(arr, result, int(size), MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(result[i], i + rank_cnt - 1);
  }

  delete[] arr;
  delete[] result;
}

TEST(allreduce, null_send) {
  constexpr size_t size = 64;

  int64_t* result = new int64_t[size];

  int res = allreduce(nullptr, result, int(size), MPI_INT64_T, MPI_MAX,
                      MPI_COMM_WORLD);

  ASSERT_EQ(res, MPI_ERR_BUFFER);

  delete[] result;
}

TEST(allreduce, null_recv) {
  constexpr size_t size = 64;

  int64_t* result = new int64_t[size];

  int res = allreduce(result, nullptr, int(size), MPI_INT64_T, MPI_MAX,
                      MPI_COMM_WORLD);

  ASSERT_EQ(res, MPI_ERR_BUFFER);

  delete[] result;
}

TEST(allreduce, negative_cnt) {
  constexpr size_t size = 64;

  int64_t* result = new int64_t[size];

  int res = allreduce(result, result, -1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

  ASSERT_EQ(res, MPI_ERR_COUNT);

  delete[] result;
}

TEST(allreduce, comm_self) {
  constexpr size_t size = 1000;
  int* arr = new int[size];
  int* result = new int[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = i;
  }

  allreduce(arr, result, int(size), MPI_INT32_T, MPI_SUM, MPI_COMM_SELF);

  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(result[i], i);
  }

  delete[] arr;
  delete[] result;
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

  auto result = RUN_ALL_TESTS();

  MPI_Finalize();
  return result;
}