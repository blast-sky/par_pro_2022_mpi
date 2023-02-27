// Copyright 2023 Shuin Evgeniy
#ifndef MODULES_TASK_2_SHUIN_E_ALLREDUCINGER_ALLREDUCINGER_H_
#define MODULES_TASK_2_SHUIN_E_ALLREDUCINGER_ALLREDUCINGER_H_

#include <mpi.h>

int Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, MPI_Comm comm);

#endif  // MODULES_TASK_2_SHUIN_E_ALLREDUCINGER_ALLREDUCINGER_H_
