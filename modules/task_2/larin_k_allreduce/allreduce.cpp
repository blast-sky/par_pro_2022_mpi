// Copyright 2022 Larin Konstantin
#include <mpi.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

static void recv_any(void* recvbuf, int count, MPI_Datatype datatype,
                     MPI_Comm comm) {
  MPI_Recv(recvbuf, count, datatype, MPI_ANY_SOURCE, MPI_ANY_TAG, comm,
           nullptr);
}

template <typename T>
static std::function<T(const T&, const T&)> get_reduce_function(MPI_Op op) {
  if (op == MPI_MAX) {
    return [](const T& l, const T& r) { return std::max(l, r); };
  } else if (op == MPI_MIN) {
    return [](const T& l, const T& r) { return std::min(l, r); };
  } else if (op == MPI_SUM) {
    return std::plus<T>{};
  } else if (op == MPI_PROD) {
    return std::multiplies<T>{};
  } else {
    throw 1;
    return std::plus<T>{};
  }
}

template <typename T>
static void processing(T* dst, T* src, size_t size, MPI_Op op) {
  auto operation = get_reduce_function<T>(op);

  for (size_t i = 0; i < size; i++) {
    dst[i] = operation(dst[i], src[i]);
  }
}

static void call_processing(void* dst, void* src, size_t size, MPI_Op op,
                            MPI_Datatype type) {
  if (type == MPI_INT8_T) {
    processing(static_cast<int8_t*>(dst), static_cast<int8_t*>(src), size, op);

  } else if (type == MPI_INT16_T) {
    processing(static_cast<int16_t*>(dst), static_cast<int16_t*>(src), size,
               op);

  } else if (type == MPI_INT32_T) {
    processing(static_cast<int32_t*>(dst), static_cast<int32_t*>(src), size,
               op);

  } else if (type == MPI_INT64_T) {
    processing(static_cast<int64_t*>(dst), static_cast<int64_t*>(src), size,
               op);

  } else if (type == MPI_FLOAT) {
    processing(static_cast<float*>(dst), static_cast<float*>(src), size, op);

  } else if (type == MPI_DOUBLE) {
    processing(static_cast<double*>(dst), static_cast<double*>(src), size, op);
  }
}

static int init_consts(int& rank_cnt, int& rank, int& typesize,
                       MPI_Datatype datatype, MPI_Comm comm) {
  int res;

  res = MPI_Comm_size(comm, &rank_cnt);
  if (res != MPI_SUCCESS) {
    return res;
  }

  res = MPI_Comm_rank(comm, &rank);
  if (res != MPI_SUCCESS) {
    return res;
  }

  res = MPI_Type_size(datatype, &typesize);
  if (res != MPI_SUCCESS) {
    return res;
  }

  rank++;

  return MPI_SUCCESS;
}

static int check_args(const void* sendbuf, void* recvbuf, int count,
                      MPI_Comm comm) {
  if (count < 0) {
    return MPI_ERR_COUNT;
  }
  if (comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF) {
    return MPI_ERR_COMM;
  }
  if (sendbuf == nullptr || recvbuf == nullptr) {
    return MPI_ERR_BUFFER;
  }
  return MPI_SUCCESS;
}

int allreduce(const void* sendbuf, void* recvbuf, int count,
              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank_cnt;
  int rank;
  int typesize;

  int res = init_consts(rank_cnt, rank, typesize, datatype, comm);
  if (res != MPI_SUCCESS) {
    return res;
  }
  res = check_args(sendbuf, recvbuf, count, comm);
  if (res != MPI_SUCCESS) {
    return res;
  }

  if (count == 0) {
    return MPI_SUCCESS;
  }

  std::unique_ptr<char[]> buffer;
  bool buffer_copied = false;
  if (rank * 2 <= rank_cnt) {
    buffer.reset(new char[size_t(typesize) * count]);

    buffer_copied = true;
    std::copy(static_cast<const char*>(sendbuf),
              static_cast<const char*>(sendbuf) + size_t(count) * typesize,
              static_cast<char*>(recvbuf));
  }

  constexpr int child_cnt = 2;
  for (int i = 0; i < child_cnt; i++) {
    int child = rank * 2 + i;
    if (child <= rank_cnt) {
      recv_any(buffer.get(), count, datatype, comm);
      call_processing(recvbuf, buffer.get(), count, op, datatype);
    }
  }

  if (rank != 1) {
    int parent = rank / 2 - 1;

    if (buffer_copied) {
      MPI_Send(recvbuf, count, datatype, parent, 0, comm);
    } else {
      MPI_Send(sendbuf, count, datatype, parent, 0, comm);
    }

    recv_any(recvbuf, count, datatype, comm);
  }

  for (int i = 0; i < child_cnt; i++) {
    int child = rank * 2 + i;
    if (child <= rank_cnt) {
      MPI_Send(recvbuf, count, datatype, child - 1, 0, comm);
    }
  }

  if (rank == 1 && !buffer_copied) {
    std::copy(static_cast<const char*>(sendbuf),
              static_cast<const char*>(sendbuf) + size_t(count) * typesize,
              static_cast<char*>(recvbuf));
  }

  return MPI_SUCCESS;
}
