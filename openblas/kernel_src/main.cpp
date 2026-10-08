#include <iostream>
#include <cblas.h>
#include <vector>
#include <chrono>
#include <omp.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace chrono = std::chrono;
using timer = std::chrono::steady_clock;

constexpr int NWARMUP = 10;

// For very small matrix, we will likely get very unstable results
// Especially on more complex architectures.
// We opt to report the minimum execution time instead of the mean or median
// We stop when the minimum execution time has not improved for a certain number of samples, or when we reach a maximum number of samples.
constexpr unsigned int NMIN = 200;    // minimum samples
constexpr unsigned int NMAX = 2000;   // Hard cap
constexpr unsigned int PATIENCE = 30; // stop after this many samples without improvement
constexpr double IMPROVE_REL = 0.01;  // "improvement" = min drops by >1%
constexpr double MAX_WALLTIME_S = 30.0;

void test_openmp_backend()
{
    // Pre-flight check, to ensure that OpenBLAS is compiled with OpenMP support.
    int model = openblas_get_parallel();
    if (model != OPENBLAS_OPENMP)
        throw std::runtime_error("OpenBLAS is not compiled with OpenMP support.");
}

std::unique_ptr<double[]> make_vector(size_t size, double value)
{
    // First touch initialization to avoid NUMA issues.
    // Note that std::vector performs single-threaded initialization, which can lead to NUMA issues on multi-socket systems.
    auto res = std::unique_ptr<double[]>(new double[size]);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        res[i] = value;
    }

    return res;
}

void do_warmup(const double *a, const double *b, double *c, size_t size)
{
    for (int i = 0; i < NWARMUP; i++)
    {
        cblas_dger(CblasRowMajor, size, size, 1.0, a, 1, b, 1, c, size);
    }

    double expected_value = 2 * NWARMUP;
    if (c[0] != expected_value)
    {
        throw std::runtime_error("Warmup failed: c[0] does not match expected value. Expected: " + std::to_string(expected_value) + ", got: " + std::to_string(c[0]));
    }
}

double benchmark_kernel(const double *a, const double *b, double *c, size_t n)
{
    double min_time = std::numeric_limits<double>::max();
    unsigned int since_improvement = 0, count = 0;

    // Clock to measure wall time and ensure we don't run for too long
    auto bench_start = timer::now();
    while (true)
    {
        auto begin = timer::now();
        cblas_dger(CblasRowMajor, n, n, 1.0, a, 1, b, 1, c, n);
        auto end = timer::now();

        double t = chrono::duration<double>(end - begin).count();
        count++;

        // If the new time is significantly better than the previous minimum, reset the counter
        if (t < min_time * (1.0 - IMPROVE_REL))
            since_improvement = 0; // meaningful new minimum
        else
            since_improvement++;

        if (t < min_time)
            min_time = t;

        if (count >= NMIN && since_improvement >= PATIENCE)
            break;

        if (count >= NMAX ||
            chrono::duration<double>(timer::now() - bench_start).count() >= MAX_WALLTIME_S)
            break;
    }
    return min_time;
}

int main(int argc, char **argv)
{
    test_openmp_backend();

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <vec_size> <nthreads>" << std::endl;
        return 1;
    }

    const size_t vec_size = std::stoull(argv[1]);
    const int nthreads = std::stoi(argv[2]);

    // Just to be safe, set the number of threads everywhere
    omp_set_num_threads(nthreads);
    openblas_set_num_threads(nthreads);

    auto a = make_vector(vec_size, 1.0);
    auto b = make_vector(vec_size, 2.0);
    auto c = make_vector(vec_size * vec_size, 0.0);

    do_warmup(a.get(), b.get(), c.get(), vec_size);
    double res = benchmark_kernel(a.get(), b.get(), c.get(), vec_size);

    // wrapper.py parse the very last line of stdout to get the result
    std::cout << res << std::endl;

    return 0;
}