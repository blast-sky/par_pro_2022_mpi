// Copyright 2023 Titov Egor
#ifndef MODULES_TASK_2_TITOV_E_STAR_TOPOLOGY_STAR_TOPOLOGY_H_
#define MODULES_TASK_2_TITOV_E_STAR_TOPOLOGY_STAR_TOPOLOGY_H_

#include <mpi.h>

MPI_Comm createStarComm(MPI_Comm old);
bool isStarTopology(MPI_Comm new_comm);

#endif  // MODULES_TASK_2_TITOV_E_STAR_TOPOLOGY_STAR_TOPOLOGY_H_
