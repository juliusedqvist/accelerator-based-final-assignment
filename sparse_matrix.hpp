
#ifndef sparse_matrix_hpp
#define sparse_matrix_hpp

#include <utility>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include <omp.h>

#include <vector>

#include "vector.hpp"


#ifndef DISABLE_CUDA
template <typename Number>
__global__ void compute_spmv(const std::size_t N,
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
      n_global_nonzero_entries(other.n_global_nonzero_entries)
  {
    if (memory_space == MemorySpace::CUDA)
      {
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
        AssertCuda(cudaMalloc(&column_indices,
                              n_entries * sizeof(unsigned int)));
        AssertCuda(cudaMemcpy(column_indices,
                              other.column_indices,
                              n_entries * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

        AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));
        AssertCuda(cudaMemcpy(values,
                              other.values,
                              n_entries * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
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

        const unsigned int n_blocks = (n_rows + block_size - 1) / block_size;

        compute_spmv<<<n_blocks, block_size>>>(n_rows,
                                                row_starts,
                                                column_indices,
                                                values,
                                                src.begin(),
                                                dst.begin());

        AssertCuda(cudaPeekAtLastError());
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
        return other;
      }
  }

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


#endif
