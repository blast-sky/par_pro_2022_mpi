// Copyright 2023 Okulov Stepan
#include <mpi.h>
#include <vector>
#include <algorithm>
#include "../../../modules/task_2/okulov_s_gaussjordan/gaussjordan.h"


Matrix expandMatrixByB(const Matrix& A, const Vector& b) {
    Matrix result = A;
    for (int i = 0; i < result.size(); i++) {
        Vector& row = result[i];
        row.push_back(b[i]);
    }
    return result;
}

void subtructRowFromMatrix(Matrix* A_, const Vector& row, int rowIndex, int begin, int end) {
    Matrix& A = *A_;

    for (int i = begin; i < end; i++) {
        if (i == rowIndex) continue;
        Vector& currentRow = A[i];

        const double ratio = currentRow[rowIndex] / row[rowIndex];
        for (int j = 0; j < currentRow.size(); j++) {
            currentRow[j] -= row[j] * ratio;
        }
    }
}

void divRow(Vector* row_, double ratio) {
    Vector& row = *row_;
    std::transform(row.begin(), row.end(), row.begin(), [&](double e) { return e / ratio; });
}

Vector getGauseJordanPar(const Matrix& A, const Vector& b) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix expandedMatrix = expandMatrixByB(A, b);

    int rowPerProc = A.size() / size + ((A.size() % size) != 0);

    int currentProcBeginRow = rowPerProc * rank;
    int currentProcEndRow = std::min(rowPerProc * (rank + 1), static_cast<int>(A.size()));

    for (int row = 0; row < expandedMatrix.size(); row++) {
        Vector dividedRow(A.size() + 1);

        if (currentProcBeginRow <= row && row < currentProcEndRow) {
            const double mainElement = expandedMatrix[row][row];
            divRow(&expandedMatrix[row], mainElement);
            dividedRow = expandedMatrix[row];
        }

        int currentProcSource = row / rowPerProc;
        MPI_Bcast(dividedRow.data(), dividedRow.size(), MPI_DOUBLE, currentProcSource, MPI_COMM_WORLD);

        subtructRowFromMatrix(&expandedMatrix, dividedRow, row, currentProcBeginRow, currentProcEndRow);
    }

    Vector resultB;
    for (int i = currentProcBeginRow; i < currentProcEndRow; i++) {
        resultB.push_back(expandedMatrix[i].back());
    }
    resultB.resize(rowPerProc);

    Vector result;
    if (rank == 0) result.resize(rowPerProc * size);

    MPI_Gather(resultB.data(), rowPerProc, MPI_DOUBLE, result.data(), rowPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    result.resize(b.size());

    return result;
}
