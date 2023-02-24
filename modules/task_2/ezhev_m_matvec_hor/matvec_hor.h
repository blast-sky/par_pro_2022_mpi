// Copyright 2023 Ezhev Mikhail

#ifndef MODULES_TASK_2_EZHEV_M_MATVEC_HOR_MATVEC_HOR_H_
#define MODULES_TASK_2_EZHEV_M_MATVEC_HOR_MATVEC_HOR_H_

#include <mpi.h>

#include <iostream>
#include <random>
#include <vector>

std::vector<int> RandVector(int n);

std::vector<int> MultiplySequential(const std::vector<int>& mat, int row,
                                    int col, const std::vector<int>& vec);
std::vector<int> MultiplyParallel(const std::vector<int>& mat, int row, int col,
                                  const std::vector<int>& vec);

#endif  // MODULES_TASK_2_EZHEV_M_MATVEC_HOR_MATVEC_HOR_H_
