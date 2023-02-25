// Copyright 2023 Okulov Stepan
#ifndef MODULES_TASK_2_OKULOV_S_GAUSSJORDAN_GAUSSJORDAN_H_
#define MODULES_TASK_2_OKULOV_S_GAUSSJORDAN_GAUSSJORDAN_H_

#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

Vector getGauseJordanPar(const Matrix& A, const Vector& b);

#endif  // MODULES_TASK_2_OKULOV_S_GAUSSJORDAN_GAUSSJORDAN_H_
