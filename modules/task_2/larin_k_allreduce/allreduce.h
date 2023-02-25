// Copyright 2022 Larin Konstantin
#ifndef ALLREDUCE_H
#define ALLREDUCE_H

int allreduce(const void* sendbuf, void* recvbuf, int count,
              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif  // ALLREDUCE_H
