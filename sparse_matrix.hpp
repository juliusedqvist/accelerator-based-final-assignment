
#ifndef sparse_matrix_hpp
#define sparse_matrix_hpp

#include <utility>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include <omp.h>
#include <iostream>
#include <algorithm>

#include <vector>

#include "vector.hpp"


#ifndef DISABLE_CUDA
template <typename Number>
__global__ void compute_spmv(
                              const std::size_t N,
                              const std::size_t *row_starts,
                              const unsigned int *column_indices,
                              const Number *values,
                              const Number *x,
                              Number *y)
{
  // TODO implement for GPU
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N)
    return;

  Number sum = 0;
  const std::size_t start = row_starts[row];
  const std::size_t end = row_starts[row + 1];

  for (std::size_t idx = start; idx < end; ++idx)
  {
    const unsigned int col = column_indices[idx];
    sum += values[idx] * x[col];
  }

  y[row] = sum;
}

template <typename Number>
__global__ void compute_spmv_sell(
    const unsigned int n_rows,
    const unsigned int n_slices,
    const unsigned int C,
    const std::size_t  *cs,
    const unsigned int *cl,
    const unsigned int *col,
    const Number       *val,
    const Number       *x,
    Number             *y)
{
  const unsigned int slice = blockIdx.x;
  const unsigned int lane  = threadIdx.x; // row inside slice

  if (slice >= n_slices || lane >= C)
    return;

  const unsigned int row = slice * C + lane;
  if (row >= n_rows)
    return;  // very important for the last partial slice

  Number sum = 0;
  const unsigned int slice_len = cl[slice];
  const std::size_t  base      = cs[slice];

  // Column-major by slice: idx = base + j*C + lane
  for (unsigned int j = 0; j < slice_len; ++j) {
    const std::size_t idx = base + j * C + lane;
    sum += val[idx] * x[col[idx]];
  }

  y[row] = sum;
}
#endif



// Sparse matrix in compressed row storage (crs) format

template <typename Number>
class SparseMatrix
{
public:
  static const int block_size = Vector<Number>::block_size;

  SparseMatrix(const std::vector<unsigned int> &row_lengths,
               const MemorySpace                memory_space,
               const MPI_Comm                   communicator)
    : communicator(communicator),
      memory_space(memory_space)
  {
    n_rows     = row_lengths.size();
    row_starts = new std::size_t[n_rows + 1];

#pragma omp parallel for
    for (unsigned int row = 0; row < n_rows + 1; ++row)
      row_starts[row] = 0;

    for (unsigned int row = 0; row < n_rows; ++row)
      row_starts[row + 1] = row_starts[row] + row_lengths[row];

    const std::size_t n_entries = row_starts[n_rows];


    if (memory_space == MemorySpace::CUDA)
      {
        std::size_t *host_row_starts = row_starts;
        row_starts = 0;
        AssertCuda(cudaMalloc(&row_starts, (n_rows + 1) * sizeof(std::size_t)));
        AssertCuda(cudaMemcpy(row_starts,
                              host_row_starts,
                              (n_rows + 1) * sizeof(std::size_t),
                              cudaMemcpyHostToDevice));
        delete[] host_row_starts;

        AssertCuda(cudaMalloc(&column_indices,
                              n_entries * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));

#ifndef DISABLE_CUDA
        const unsigned int n_blocks =
          (n_entries + block_size - 1) / block_size;
        set_entries<<<n_blocks, block_size>>>(n_entries, 0U, column_indices);
        set_entries<<<n_blocks, block_size>>>(n_entries, Number(0), values);
        AssertCuda(cudaPeekAtLastError());
#endif
      }
    else
      {
        column_indices = new unsigned int[n_entries];
        values         = new Number[n_entries];

#pragma omp parallel for
        for (std::size_t i = 0; i < n_entries; ++i)
          column_indices[i] = 0;

#pragma omp parallel for
        for (std::size_t i = 0; i < n_entries; ++i)
          values[i] = 0;
      }

    n_global_nonzero_entries = mpi_sum(n_entries, communicator);
  }

  ~SparseMatrix()
  {
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        cudaFree(row_starts);
        cudaFree(column_indices);
        cudaFree(values);
#endif
      }
    else
      {
        delete[] row_starts;
        delete[] column_indices;
        delete[] values;
      }
  }

  SparseMatrix(const SparseMatrix &other)
      : communicator(other.communicator),
        memory_space(other.memory_space),
        n_rows(other.n_rows),
        n_global_nonzero_entries(other.n_global_nonzero_entries),
        // also copy SELL metadata
        sell_C(other.sell_C),
        sell_sigma(other.sell_sigma),
        sell_n_slices(other.sell_n_slices),
        sell_total_nnz(other.sell_total_nnz),
        d_sell_cs(nullptr),
        d_sell_cl(nullptr),
        d_sell_col(nullptr),
        d_sell_val(nullptr)
  {
    if (memory_space == MemorySpace::CUDA)
    {
      // === copy CSR device arrays ===
      AssertCuda(cudaMalloc(&row_starts, (n_rows + 1) * sizeof(std::size_t)));
      AssertCuda(cudaMemcpy(row_starts,
                            other.row_starts,
                            (n_rows + 1) * sizeof(std::size_t),
                            cudaMemcpyDeviceToDevice));

      std::size_t n_entries = 0;
      AssertCuda(cudaMemcpy(&n_entries,
                            other.row_starts + n_rows,
                            sizeof(std::size_t),
                            cudaMemcpyDeviceToHost));

      AssertCuda(cudaMalloc(&column_indices, n_entries * sizeof(unsigned int)));
      AssertCuda(cudaMemcpy(column_indices,
                            other.column_indices,
                            n_entries * sizeof(unsigned int),
                            cudaMemcpyDeviceToDevice));

      AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));
      AssertCuda(cudaMemcpy(values,
                            other.values,
                            n_entries * sizeof(Number),
                            cudaMemcpyDeviceToDevice));

      // === copy SELL arrays if present ===
      if (other.d_sell_val != nullptr && sell_total_nnz > 0)
      {
        AssertCuda(cudaMalloc(&d_sell_cs, (sell_n_slices + 1) * sizeof(std::size_t)));
        AssertCuda(cudaMalloc(&d_sell_cl, sell_n_slices * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&d_sell_col, sell_total_nnz * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&d_sell_val, sell_total_nnz * sizeof(Number)));

        AssertCuda(cudaMemcpy(d_sell_cs, other.d_sell_cs,
                              (sell_n_slices + 1) * sizeof(std::size_t),
                              cudaMemcpyDeviceToDevice));
        AssertCuda(cudaMemcpy(d_sell_cl, other.d_sell_cl,
                              sell_n_slices * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
        AssertCuda(cudaMemcpy(d_sell_col, other.d_sell_col,
                              sell_total_nnz * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
        AssertCuda(cudaMemcpy(d_sell_val, other.d_sell_val,
                              sell_total_nnz * sizeof(Number),
                              cudaMemcpyDeviceToDevice));

        std::cout << "[copy ctor] Copied SELL arrays to new CUDA matrix\n";
      }
    }
    else
      {

      }
  }

  // do not allow copying matrix
  SparseMatrix operator=(const SparseMatrix &other) = delete;

  unsigned int m() const
  {
    return n_rows;
  }

  std::size_t n_nonzero_entries() const
  {
    return n_global_nonzero_entries;
  }

  void add_row(unsigned int               row,
               std::vector<unsigned int> &columns_of_row,
               std::vector<Number> &      values_in_row)
  {
    if (columns_of_row.size() != values_in_row.size())
      {
        std::cout << "column_indices and values must have the same size!"
                  << std::endl;
        std::abort();
      }
    for (unsigned int i = 0; i < columns_of_row.size(); ++i)
      {
        column_indices[row_starts[row] + i] = columns_of_row[i];
        values[row_starts[row] + i]         = values_in_row[i];
      }
  }

  void allocate_ghost_data_memory(const std::size_t n_ghost_entries)
  {
    ghost_entries.clear();
    ghost_entries.reserve(n_ghost_entries);
#pragma omp parallel for
    for (unsigned int i = 0; i < n_ghost_entries; ++i)
      {
        ghost_entries[i].index_within_result         = 0;
        ghost_entries[i].index_within_offproc_vector = 0;
        ghost_entries[i].value                       = 0.;
      }
  }

  void add_ghost_entry(const unsigned int local_row,
                       const unsigned int offproc_column,
                       const Number       value)
  {
    GhostEntryCoordinateFormat entry;
    entry.value                       = value;
    entry.index_within_result         = local_row;
    entry.index_within_offproc_vector = offproc_column;
    ghost_entries.push_back(entry);
  }

  // In real codes, the data structure we pass in manually here could be
  // deduced from the global indices that are accessed. In the most general
  // case, it takes some two-phase index lookup via a dictionary to find the
  // owner of particular columns (sometimes called consensus algorithm).
  void set_send_and_receive_information(
    std::vector<std::pair<unsigned int, std::vector<unsigned int>>>
                                                       send_indices,
    std::vector<std::pair<unsigned int, unsigned int>> receive_indices)
  {
    this->send_indices    = send_indices;
    std::size_t send_size = 0;
    for (auto i : send_indices)
      send_size += i.second.size();
    send_data.resize(send_size);
    this->receive_indices    = receive_indices;
    std::size_t receive_size = 0;
    for (auto i : receive_indices)
      receive_size += i.second;
    receive_data.resize(receive_size);

    const unsigned int my_mpi_rank = get_my_mpi_rank(communicator);

    if (receive_size > ghost_entries.size())
      {
        std::cout << "Error, you requested exchange of more entries than what "
                  << "there are ghost entries allocated in the matrix, which "
                  << "does not make sense. Check matrix setup." << std::endl;
        std::abort();
      }
  }


  void apply(const Vector<Number> &src, Vector<Number> &dst) const
  {
    if (m() != src.size_on_this_rank() || m() != dst.size_on_this_rank())
      {
        std::cout << "vector sizes of src " << src.size_on_this_rank()
                  << " and dst " << dst.size_on_this_rank()
                  << " do not match matrix size " << m() << std::endl;
        std::abort();
      }

#ifdef HAVE_MPI
    // start exchanging the off-processor data
    std::vector<MPI_Request> mpi_requests(send_indices.size() +
                                          receive_indices.size());
    for (unsigned int i = 0, count = 0; i < receive_indices.size();
         count += receive_indices[i].second, ++i)
      MPI_Irecv(receive_data.data() + count,
                receive_indices[i].second * sizeof(Number),
                MPI_BYTE,
                receive_indices[i].first,
                /* mpi_tag */ 29,
                communicator,
                &mpi_requests[i]);
    for (unsigned int i = 0, count = 0; i < send_indices.size(); ++i)
      {
#  pragma omp parallel for
        for (unsigned int j = 0; j < send_indices[i].second.size(); ++j)
          send_data[count + j] = src(send_indices[i].second[j]);

        MPI_Isend(send_data.data() + count,
                  send_indices[i].second.size() * sizeof(Number),
                  MPI_BYTE,
                  send_indices[i].first,
                  /* mpi_tag */ 29,
                  communicator,
                  &mpi_requests[i + receive_indices.size()]);
        count += send_indices[i].second.size();
      }
#endif

    // main loop for the sparse matrix-vector product
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        // TODO implement for GPU (with CRS and ELLPACK/SELL-C-sigma)
        if (d_sell_val != nullptr) {
          // TODO launch sell-c-sigma kernel

          dim3 grid(sell_n_slices);
          dim3 block(sell_C);
          compute_spmv_sell<Number><<<grid, block>>>(
              /* n_rows  */ n_rows,
              /* n_slices*/ sell_n_slices,
              /* C       */ sell_C,
              /* cs      */ d_sell_cs,
              /* cl      */ d_sell_cl,
              /* col     */ d_sell_col,
              /* val     */ d_sell_val,
              /* x       */ src.begin(),
              /* y       */ dst.begin());
          AssertCuda(cudaPeekAtLastError());
        } else {
          const unsigned int n_blocks = (n_rows + block_size - 1) / block_size;

          compute_spmv<<<n_blocks, block_size>>>(n_rows,
                                                 row_starts,
                                                 column_indices,
                                                 values,
                                                 src.begin(),
                                                 dst.begin());

          AssertCuda(cudaPeekAtLastError());
        }
#endif
      }
    else
      {
#pragma omp parallel for
        for (unsigned int row = 0; row < n_rows; ++row)
          {
            Number sum = 0;
            for (std::size_t idx = row_starts[row]; idx < row_starts[row + 1];
                 ++idx)
              sum += values[idx] * src(column_indices[idx]);
            dst(row) = sum;
          }
      }

#ifdef HAVE_MPI
    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);

    // work on the off-processor data. do not do it in parallel because we do
    // not know whether two parts would work on the same entry of the result
    // vector
    for (auto &entry : ghost_entries)
      dst(entry.index_within_result) +=
        entry.value * receive_data[entry.index_within_offproc_vector];
#endif
  }

  SparseMatrix copy_to_device()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        std::cout << "Copy between device matrices not implemented"
                  << std::endl;
        exit(EXIT_FAILURE);
        // return dummy
        return SparseMatrix(std::vector<unsigned int>(),
                            MemorySpace::CUDA,
                            communicator);
      }
    else
      {
        std::vector<unsigned int> row_lengths(n_rows);
        for (unsigned int i = 0; i < n_rows; ++i)
          row_lengths[i] = row_starts[i + 1] - row_starts[i];

        SparseMatrix other(row_lengths,
                           MemorySpace::CUDA,
                           communicator);
        AssertCuda(cudaMemcpy(other.column_indices,
                              column_indices,
                              row_starts[n_rows] * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        AssertCuda(cudaMemcpy(other.values,
                              values,
                              row_starts[n_rows] * sizeof(Number),
                              cudaMemcpyHostToDevice));
        if (d_sell_val != nullptr || !h_sell_val.empty())
        {
          other.sell_C = sell_C;
          other.sell_sigma = sell_sigma;
          other.sell_n_slices = sell_n_slices;
          other.sell_total_nnz = sell_total_nnz;

          // Allocate device buffers
          AssertCuda(cudaMalloc(&other.d_sell_cs, (sell_n_slices + 1) * sizeof(std::size_t)));
          AssertCuda(cudaMalloc(&other.d_sell_cl, sell_n_slices * sizeof(unsigned int)));
          AssertCuda(cudaMalloc(&other.d_sell_col, sell_total_nnz * sizeof(unsigned int)));
          AssertCuda(cudaMalloc(&other.d_sell_val, sell_total_nnz * sizeof(Number)));

          // Copy from host vectors
          AssertCuda(cudaMemcpy(other.d_sell_cs, h_sell_cs.data(),
                                (sell_n_slices + 1) * sizeof(std::size_t),
                                cudaMemcpyHostToDevice));
          AssertCuda(cudaMemcpy(other.d_sell_cl, h_sell_cl.data(),
                                sell_n_slices * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
          AssertCuda(cudaMemcpy(other.d_sell_col, h_sell_col.data(),
                                sell_total_nnz * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
          AssertCuda(cudaMemcpy(other.d_sell_val, h_sell_val.data(),
                                sell_total_nnz * sizeof(Number),
                                cudaMemcpyHostToDevice));

          std::cout << "[copy_to_device] Transferred SELL arrays: "
                    << "C=" << sell_C
                    << " slices=" << sell_n_slices
                    << " nnz=" << sell_total_nnz << std::endl;
        }
        return other;
      }
  }
  void convert_to_sell_c_sigma(const unsigned int C, const unsigned int sigma);

  std::size_t memory_consumption() const
  {
    return n_global_nonzero_entries * (sizeof(Number) + sizeof(unsigned int)) +
           (n_rows + 1) * sizeof(decltype(*row_starts)) +
           sizeof(GhostEntryCoordinateFormat) * ghost_entries.capacity();
  }

private:
  MPI_Comm      communicator;
  std::size_t   n_rows;
  std::size_t * row_starts;
  unsigned int *column_indices;
  Number *      values;
  std::size_t   n_global_nonzero_entries;
  MemorySpace   memory_space;
  // sell-c-sigma
  unsigned int sell_C         = 0;
  unsigned int sell_sigma     = 0;
  unsigned int sell_n_slices  = 0;
  std::size_t  sell_total_nnz = 0;   // padded total elements across slices

  // ---------------- SELL-C-sigma device arrays ----------------
  // slice starts (size: sell_n_slices + 1), cumulative offsets into val/col
  std::size_t  *d_sell_cs   = nullptr;     // "cs" = cumulative slice offsets
  // slice lengths (size: sell_n_slices), max row length in each slice
  unsigned int *d_sell_cl   = nullptr;     // "cl" = columns per slice
  // column indices and values in column-major per slice, size: sell_total_nnz
  unsigned int *d_sell_col  = nullptr;
  Number       *d_sell_val  = nullptr;

  std::vector<std::size_t>  h_sell_cs;   // size: sell_n_slices + 1 (cumulative slice offsets)
  std::vector<unsigned int> h_sell_cl;   // size: sell_n_slices     (max row length per slice)
  std::vector<unsigned int> h_sell_col;  // size: sell_total_nnz    (column indices, padded, column-major per slice)
  std::vector<Number> h_sell_val;

  struct GhostEntryCoordinateFormat
  {
    unsigned int index_within_result;
    unsigned int index_within_offproc_vector;
    Number       value;
  };
  std::vector<GhostEntryCoordinateFormat> ghost_entries;

  std::vector<std::pair<unsigned int, std::vector<unsigned int>>> send_indices;
  mutable std::vector<Number>                                     send_data;
  std::vector<std::pair<unsigned int, unsigned int>> receive_indices;
  mutable std::vector<Number>                        receive_data;
};

template <typename Number>
void SparseMatrix<Number>::convert_to_sell_c_sigma(const unsigned int C,
                                                   const unsigned int sigma)
{
  this->sell_C     = C;
  this->sell_sigma = sigma;
  std::cout << "Converting CSR to SELL-C-Sigma\n";

  sell_n_slices = (n_rows + C - 1) / C;

  // Host arrays
  h_sell_cs.resize(sell_n_slices + 1);
  h_sell_cl.resize(sell_n_slices);

  // determine max row length per slice
  for (unsigned int s = 0; s < sell_n_slices; ++s) {
    unsigned int slice_start = s * C;
    unsigned int slice_end = std::min(slice_start + C, (unsigned int)n_rows); // Extra condition if last chunk isnt C rows
    unsigned int max_len = 0;

    for (unsigned int r = slice_start; r < slice_end; ++r)
      max_len = std::max(max_len, (unsigned int)(row_starts[r + 1] - row_starts[r])); // Increments max_len if row is larger

    h_sell_cl[s] = max_len; // cl already complete
  }

  // construct cs (chunk start indices) from cl
  h_sell_cs[0] = 0;
  for (unsigned int s = 0; s < sell_n_slices; ++s)
    h_sell_cs[s + 1] = h_sell_cs[s] + C * h_sell_cl[s];

  // Total number of elements (including padding) stored in sell arrays
  sell_total_nnz = h_sell_cs.back();

  h_sell_col.resize(sell_total_nnz);
  h_sell_val.resize(sell_total_nnz);

  // main func
  for (unsigned int s = 0; s < sell_n_slices; ++s)
  {
    unsigned int slice_start = s * C;
    unsigned int slice_end = std::min(slice_start + C, (unsigned int)n_rows);
    unsigned int slice_len = h_sell_cl[s];

    for (unsigned int j = 0; j < slice_len; ++j)
    {
      for (unsigned int c = 0; c < C; ++c)
      {
        unsigned int row = slice_start + c;
        if (row >= n_rows)
          continue;

        std::size_t row_begin = row_starts[row];
        std::size_t row_end = row_starts[row + 1];
        std::size_t row_len = row_end - row_begin;

        std::size_t dest_index = h_sell_cs[s] + j * C + c; // chunk start index + column index * chunk + row within slice (to get column major)

        if (j < row_len)
        {
          h_sell_val[dest_index] = values[row_begin + j];
          h_sell_col[dest_index] = column_indices[row_begin + j];
        }
        else
        {
          // padding
          h_sell_val[dest_index] = 0;
          h_sell_col[dest_index] = column_indices[row_begin]; // safe index (0@0)
        }
      }
    }
  }

#ifndef DISABLE_CUDA
  // Free old arrays if any
  if (d_sell_cs)
    cudaFree(d_sell_cs);
  if (d_sell_cl)
    cudaFree(d_sell_cl);
  if (d_sell_col)
    cudaFree(d_sell_col);
  if (d_sell_val)
    cudaFree(d_sell_val);

  // Allocate device arrays
  AssertCuda(cudaMalloc(&d_sell_cs, (sell_n_slices + 1) * sizeof(std::size_t)));
  AssertCuda(cudaMalloc(&d_sell_cl, sell_n_slices * sizeof(unsigned int)));
  AssertCuda(cudaMalloc(&d_sell_col, sell_total_nnz * sizeof(unsigned int)));
  AssertCuda(cudaMalloc(&d_sell_val, sell_total_nnz * sizeof(Number)));

  // Copy host → device
  AssertCuda(cudaMemcpy(d_sell_cs, h_sell_cs.data(),
                        (sell_n_slices + 1) * sizeof(std::size_t),
                        cudaMemcpyHostToDevice));

  AssertCuda(cudaMemcpy(d_sell_cl, h_sell_cl.data(),
                        sell_n_slices * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  AssertCuda(cudaMemcpy(d_sell_col, h_sell_col.data(),
                        sell_total_nnz * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  AssertCuda(cudaMemcpy(d_sell_val, h_sell_val.data(),
                        sell_total_nnz * sizeof(Number),
                        cudaMemcpyHostToDevice));
#endif
}

#endif
