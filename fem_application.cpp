

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#define DISABLE_CUDA 1

#include "conjugate_gradient.hpp"
#include "sparse_matrix.hpp"
#include "vector.hpp"



template <typename Number>
SparseMatrix<Number> fill_sparse_matrix(const unsigned int size,
                                        const MemorySpace  memory_space)
{
  std::vector<unsigned int> row_lengths(size * size * size);
#pragma omp parallel for collapse(2)
  for (unsigned int iz = 0; iz < size; ++iz)
    for (unsigned int iy = 0; iy < size; ++iy)
      for (unsigned int ix = 0; ix < size; ++ix)
        {
          const unsigned n_inside =
            ((ix > 0 && ix < size - 1) + (iy > 0 && iy < size - 1) +
             (iz > 0 && iz < size - 1));
          if (n_inside == 3)
            row_lengths[(iz * size + iy) * size + ix] = 27;
          else if (n_inside == 2)
            row_lengths[(iz * size + iy) * size + ix] = 18;
          else if (n_inside == 1)
            row_lengths[(iz * size + iy) * size + ix] = 12;
          else
            row_lengths[(iz * size + iy) * size + ix] = 8;
        }

  Number entries[27] = {-1. / 12., -1. / 6., -1. / 12., -1. / 6., 0., -1. / 6.,
                        -1. / 12,  -1. / 6., -1. / 12., -1. / 6., 0., -1. / 6.,
                        0.,        8. / 3.,  0.,        -1. / 6., 0., -1. / 6.,
                        -1. / 12., -1. / 6., -1. / 12., -1. / 6., 0., -1. / 6.,
                        -1. / 12,  -1. / 6., -1. / 12.};
  const double         scale = (size + 1) * (size + 1);
  SparseMatrix<Number> sparse(row_lengths, memory_space, MPI_COMM_SELF);

#pragma omp parallel
  {
    std::vector<unsigned int> col_indices;
    std::vector<Number>       values;
#pragma omp for collapse(2)
    for (int iz = 0; iz < size; ++iz)
      for (int iy = 0; iy < size; ++iy)
        for (int ix = 0; ix < size; ++ix)
          {
            // go through all entries in the current row, clipping away the
            // parts on the boundary of the cube
            const int row = (iz * size + iy) * size + ix;
            col_indices.clear();
            values.clear();
            for (int iiz = -1, count = 0; iiz <= 1; ++iiz)
              for (int iiy = -1; iiy <= 1; ++iiy)
                for (int iix = -1; iix <= 1; ++iix, ++count)
                  if (iz + iiz >= 0 && iz + iiz < size && iy + iiy >= 0 &&
                      iy + iiy < size && ix + iix >= 0 && ix + iix < size)
                    {
                      col_indices.push_back(row + (iiz * size + iiy) * size +
                                            iix);
                      values.push_back(entries[count] * scale);
                    }
            if (col_indices.size() != row_lengths[row])
              exit(EXIT_FAILURE);
            sparse.add_row(row, col_indices, values);
          }
  }
  return sparse;
}



template <typename Number>
void run_test(const long long N, const long long n_repeat)
{
  if (get_n_mpi_ranks(MPI_COMM_WORLD) > 1)
    {
      std::cout << "Program not written for multiple MPI processes"
                << std::endl;
      exit(EXIT_FAILURE);
    }

  std::cout << "Computing on a " << N << "^3 domain with "
            << (std::is_same<Number, double>::value ? "double" : "float")
            << " numbers" << std::endl;

  MPI_Comm communicator = MPI_COMM_SELF;
  const MemorySpace memory_space = MemorySpace::Host;

  SparseMatrix<Number> matrix = fill_sparse_matrix<Number>(N, memory_space);

  Vector<Number>    src(N * N * N, memory_space);
  Vector<Number>    dst(src), result(src);

  constexpr double PI = 3.14159265358979323846;

#pragma omp parallel for collapse(2)
  for (unsigned int iz = 0; iz < N; ++iz)
    for (unsigned int iy = 0; iy < N; ++iy)
      for (unsigned int ix = 0; ix < N; ++ix)
        src((iz * N + iy) * N + ix) =
          std::sin(PI * static_cast<double>(ix + 1) / (N + 1)) *
          std::sin(PI * static_cast<double>(iy + 1) / (N + 1)) *
          std::sin(PI * static_cast<double>(iz + 1) / (N + 1));

  matrix.apply(src, dst);

  {
    double error = 0;
#pragma omp parallel for
    for (unsigned int iz = 0; iz < N; ++iz)
      {
        double my_error = 0;
        for (unsigned int iy = 0; iy < N; ++iy)
          for (unsigned int ix = 0; ix < N; ++ix)
            {
              double local_error =
                PI * PI * 3 *
                  std::sin(PI * static_cast<double>(ix + 1) / (N + 1)) *
                  std::sin(PI * static_cast<double>(iy + 1) / (N + 1)) *
                  std::sin(PI * static_cast<double>(iz + 1) / (N + 1)) -
                dst(iz * N * N + iy * N + ix);
              my_error +=
                local_error * local_error / ((N + 1) * (N + 1) * (N + 1));
            }
#pragma omp critical
        error += my_error;
      }
    const double global_error = std::sqrt(mpi_sum(error, communicator));
    std::cout << "L2 discretization error with sparse matrix: " << global_error
              << std::endl;
  }

  {
    const auto t1 = std::chrono::steady_clock::now();
    for (unsigned long long rep = 0; rep < n_repeat; ++rep)
      matrix.apply(src, dst);

    const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t1)
        .count();

    std::cout << "Mat-vec of size " << src.size() << ": "
              << time / n_repeat << " seconds or "
              << 1e-9 * n_repeat * (matrix.memory_consumption() +
                                    2 * dst.size() * sizeof(Number)) / time
              << " GB/s " << std::endl;
  }

  {
    // somewhat modify the initial condition to not let CG converge
    // immediately due to hitting an eigenmode
    result(0) = 1.;
    result(1) = 0.8;

    const auto t1 = std::chrono::steady_clock::now();
    const auto info =
      solve_with_conjugate_gradient(500, 1e-12, matrix, dst, result);

    const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t1)
        .count();

    std::cout << "Conjugate gradient solve of size " << src.size() << " in "
              << info.first << " iterations: " << time << " seconds or "
              << std::setw(8) << 1e-6 * dst.size() * info.first / time
              << " MUPD/s/it" << std::endl;
    result.add(-1., src);
    const double l2_norm = result.l2_norm();
    std::cout << "Error conjugate gradient solve: " << l2_norm << std::endl;
  }
}



int main(int argc, char **argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  long long          N           = -1;
  long long          n_repeat    = 100;
  std::string        number      = "double";
  const unsigned int my_mpi_rank = get_my_mpi_rank(MPI_COMM_WORLD);

  if (argc % 2 == 0)
    {
      if (my_mpi_rank == 0)
        std::cout << "Error, expected odd number of common line arguments"
                  << std::endl
                  << "Expected line of the form" << std::endl
                  << "-N 100 -repeat 100 -number double" << std::endl;
      std::abort();
    }

  // parse from the command line
  for (unsigned l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-N")
        N = std::atoll(argv[l + 1]);
      else if (option == "-repeat")
        n_repeat = std::atoll(argv[l + 1]);
      else if (option == "-number")
        number = argv[l + 1];
      else if (my_mpi_rank == 0)
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }

  if (N == -1)
    for (unsigned long long NN = 24; NN < 500; NN += 24)
      {
        if (number == "double")
          run_test<double>(NN, n_repeat);
        else
          run_test<float>(NN, n_repeat);
      }
  else
    {
      if (number == "double")
        run_test<double>(N, n_repeat);
      else
        run_test<float>(N, n_repeat);
    }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
}
