
#ifndef vector_hpp
#define vector_hpp

#include <omp.h>

#include <memory>
#include <utility>

#ifndef HAVE_MPI


////////////////// implementation of some utility functions //////////////////



// define dummy values for some frequently used MPI commands in case we do not
// have MPI
using MPI_Comm           = int;
const int MPI_COMM_SELF  = 0;
const int MPI_COMM_WORLD = 1;

#else
#  include <mpi.h>
#endif


enum class MemorySpace
{
  Host,
  CUDA
};

unsigned int get_n_mpi_ranks(MPI_Comm communicator)
{
  int n_ranks = 1;
#ifdef HAVE_MPI
  MPI_Comm_size(communicator, &n_ranks);
#endif
  return n_ranks;
}

unsigned int get_my_mpi_rank(MPI_Comm communicator)
{
  int my_rank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(communicator, &my_rank);
#endif
  return my_rank;
}

template <typename Number>
Number mpi_sum(const Number local_sum, MPI_Comm communicator)
{
#ifdef HAVE_MPI
  if (std::is_same<Number, double>::value)
    {
      double global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator);
      return global_sum;
    }
  else if (std::is_same<Number, float>::value)
    {
      float global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, communicator);
      return global_sum;
    }
  else if (std::is_same<Number, std::size_t>::value)
    {
      std::size_t global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, communicator);
      return global_sum;
    }
  else
    {
      std::cout << "Unknown number type" << std::endl;
      std::abort();
    }
#else
  return local_sum;
#endif
}


#ifdef DISABLE_CUDA
#define AssertCuda(error_code)
#else
#define AssertCuda(error_code)                                          \
  if (error_code != cudaSuccess)                                        \
    {                                                                   \
      std::cout << "The cuda call in " << __FILE__ << " on line "       \
                << __LINE__ << " resulted in the error '"               \
                << cudaGetErrorString(error_code) << "'" << std::endl;  \
      std::abort();                                                     \
    }



template <typename Number>
__global__ void set_entries(const std::size_t N,
                            Number scalar,
                            Number *destination)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
    destination[idx] = scalar;
}


template <typename Number>
__global__ void vector_update(const std::size_t N,
                              Number scalar1,
                              Number scalar2,
                              const Number *source,
                              Number *destination)
{
  // TODO implement for GPU
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
  {
    destination[idx] = scalar1 * destination[idx] + scalar2 * source[idx];
  }
}


template <unsigned int block_size, typename Number>
__global__ void
do_dot(unsigned int n, Number *vector1, Number *vector2, Number *result)
{
  __shared__ Number sdata[block_size];

  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * blockDim.x + tid;

  if (i < n)
    sdata[tid] = vector1[i] * vector2[i];
  else
    sdata[tid] = 0;
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 0; s /= 2)
    {
      if (tid < s)
        {
          sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
    }

  if (tid == 0)
    atomicAdd(result, sdata[0]);
}

#endif



/////////////////// implementation of actual vector class ///////////////////

template <typename Number>
class Vector
{
public:
  static const int block_size = 256;

  // Create a serial vector of the given size
  Vector(const std::size_t global_size, const MemorySpace memory_space)
    : communicator(MPI_COMM_SELF),
      data(nullptr),
      global_size(global_size),
      locally_owned_range_start(0),
      memory_space(memory_space)
  {
    resize(global_size);
  }

  // Create a parallel vector of given global size with the half-open local
  // range owned by the current processor
  Vector(const std::size_t                         global_size,
         const std::pair<std::size_t, std::size_t> locally_owned_range,
         const MemorySpace                         memory_space,
         const MPI_Comm                            communicator)
    : communicator(communicator),
      data(nullptr),
      global_size(global_size),
      locally_owned_range_start(locally_owned_range.first),
      memory_space(memory_space)
  {
    resize(locally_owned_range.second - locally_owned_range.first);
  }

  Vector(const Vector &other)
    : communicator(other.communicator),
      data(nullptr),
      global_size(other.global_size),
      locally_owned_range_start(other.locally_owned_range_start),
      memory_space(other.memory_space)
  {
    resize_fast(other.local_size);
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaMemcpy(data,
                              other.data,
                              local_size * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = other.data[i];
      }
  }

  ~Vector()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaFree(data));
      }
    else
      delete[] data;
  }

  Vector &operator=(const Vector &other)
  {
    global_size               = other.global_size;
    locally_owned_range_start = other.locally_owned_range_start;
    if (memory_space != other.memory_space)
      {
        std::cout << "Cannot assign from one memory space to another"
                  << std::endl;
        exit(EXIT_FAILURE);
      }

    resize_fast(other.local_size);

    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaMemcpy(data,
                              other.data,
                              local_size * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = other.data[i];
      }
    return *this;
  }

  const Number &operator()(const std::size_t index) const
  {
    return data[index];
  }

  Number &operator()(const std::size_t index)
  {
    return data[index];
  }

  void operator=(const Number scalar)
  {
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        const unsigned int n_blocks =
          (local_size + block_size - 1) / block_size;
        set_entries<<<n_blocks, block_size>>>(local_size, scalar, data);
#endif
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = scalar;
      }
  }

  // computes this += other_scalar * other
  void add(const Number other_scalar, const Vector &other)
  {
    sadd(1., other_scalar, other);
  }

  // computes this = my_scalar * this + other_scalar * other
  void
  sadd(const Number my_scalar, const Number other_scalar, const Vector &other)
  {
    assert_size(other);

    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        // TODO implement for GPU
      const unsigned int n_blocks =
          (local_size + block_size - 1) / block_size;

      vector_update<<<n_blocks, block_size>>>(
        local_size,
        my_scalar,
        other_scalar,
        other.data_dev,
        data_dev);
#endif
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = my_scalar * data[i] + other_scalar * other.data[i];
      }
  }

  Number l2_norm() const
  {
    const Number norm_sqr = norm_square();
    if (std::isfinite(norm_sqr))
      return std::sqrt(norm_sqr);
    else
      {
        std::cout << "Norm not finite, aborting";
        std::abort();
        return 0;
      }
  }

  Number norm_square() const
  {
    return dot(*this);
  }

  Number dot(const Vector &other) const
  {
    assert_size(other);

    Number local_sum = 0;
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        Number *result_device;
        AssertCuda(cudaMalloc(&result_device, sizeof(Number)));
        AssertCuda(cudaMemset(result_device, 0, sizeof(Number)));
        const unsigned int n_blocks =
          (local_size + block_size - 1) / block_size;
        do_dot<block_size, Number><<<n_blocks, block_size>>>(local_size,
                                                             data,
                                                             other.data,
                                                             result_device);
        cudaMemcpy(&local_sum, result_device, sizeof(Number),
                   cudaMemcpyDeviceToHost);
#endif
      }
    else
      {
#pragma omp parallel for reduction(+ : local_sum)
        for (std::size_t i = 0; i < local_size; ++i)
          local_sum += data[i] * other.data[i];
      }

    return mpi_sum(local_sum, communicator);
  }

  Number* begin()
  {
    return data;
  }

  const Number* begin() const
  {
    return data;
  }

  Vector copy_to_device()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        return *this;
      }
    else
      {
        Vector<Number> other(global_size,
                             std::make_pair(locally_owned_range_start,
                                            locally_owned_range_start +
                                              local_size),
                             MemorySpace::CUDA,
                             communicator);
        // TODO implement copy from host to device for GPU
        cudaMalloc(&data_dev, n * sizeof(Number));
        cudaMemcpy(data_dev, data_host, n * sizeof(Number), cudaMemcpyHostToDevice);

        return other;
      }
  }

  Vector copy_to_host()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        Vector<Number> other(global_size,
                             std::make_pair(locally_owned_range_start,
                                            locally_owned_range_start +
                                              local_size),
                             MemorySpace::Host,
                             communicator);
        // TODO implement copy from device to host for GPU
        cudaMemcpy(data_host, data_dev, n_elements * sizeof(Number), cudaMemcpyDeviceToHost);

        return other;
      }
    else
      {
        return *this;
      }
  }

  std::size_t size() const
  {
    return global_size;
  }

  std::size_t size_on_this_rank() const
  {
    return local_size;
  }

private:
  MPI_Comm    communicator;
  Number *    data;
  std::size_t local_size;
  std::size_t global_size;
  std::size_t locally_owned_range_start;
  MemorySpace memory_space;

  void assert_size(const Vector &other) const
  {
    if (local_size != other.local_size)
      {
        std::cout << "The local sizes of the two vectors " << local_size
                  << " vs " << other.local_size << " do not match" << std::endl;
        std::abort();
      }
  }

  void resize(const std::size_t local_size)
  {
    resize_fast(local_size);
    this->operator=(0.);
  }

  void resize_fast(const std::size_t local_size)
  {
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaFree(data));
        AssertCuda(cudaMalloc(&data, local_size * sizeof(Number)));
      }
    else
      {
        delete[] data;
        data             = new Number[local_size];
      }
    this->local_size = local_size;
  }
};

#endif
