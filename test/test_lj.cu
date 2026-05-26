// test_lj.cu - standalone unit test for the LJ energy device function.
//
// Build:  nvcc test_lj.cu -I ../include -o test_lj
// Run:    ./test_lj          (exit code 0 = all passed, 1 = failure)
//
// Adjust the -I path so the compiler finds fitness.cuh. This file is a
// single translation unit and links no other project sources - it only
// needs the lj_energy() definition from the header.

#include "fitness.cuh"
#include "crossover.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// /// Test kernel to call COM function (uses AoS layout: [x0,y0,z0, x1,y1,z1, ...])
// __global__ void test_com_kernel_aos(const double* points, double* out, int n_points) {
//     double cx = 0.0, cy = 0.0, cz = 0.0;
//     cuga::center_of_mass(points, n_points, &cx, &cy, &cz);
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         out[0] = cx;
//         out[1] = cy;
//         out[2] = cz;
//     }
// }

/// Test kernel to call LJ energy function
__global__ void test_lj_kernel(const double* pop, double* out,
                                int population, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population) return;
    out[idx] = cuga::lennard_jones(pop, idx, population, dim);
}

static std::vector<double> run_lj(const std::vector<std::vector<double>>& clusters,
                                  int n_atoms) {
    int population = static_cast<int>(clusters.size());
    int dim = 3 * n_atoms;

    // host buffer in SoA layout: host[gene * P + individual]
    std::vector<double> host(static_cast<size_t>(population) * dim);
    for (int i = 0; i < population; ++i)
        for (int g = 0; g < dim; ++g)
            host[static_cast<size_t>(g) * population + i] = clusters[i][g];

    double *d_pop = nullptr, *d_out = nullptr;
    cudaMalloc(&d_pop, sizeof(double) * population * dim);
    cudaMalloc(&d_out, sizeof(double) * population);
    cudaMemcpy(d_pop, host.data(), sizeof(double) * population * dim, cudaMemcpyHostToDevice);

    int block = 64;
    int grid  = (population + block - 1) / block;
    test_lj_kernel<<<grid, block>>>(d_pop, d_out, population, dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("  [cuda] kernel launch error: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("  [cuda] kernel run error: %s\n", cudaGetErrorString(err));

    std::vector<double> out(population);
    cudaMemcpy(out.data(), d_out, sizeof(double) * population, cudaMemcpyDeviceToHost);

    cudaFree(d_pop);
    cudaFree(d_out);
    return out;
}

// --------------------------------------------------------------------------
// Tiny assertion helpers - no external test framework needed.
// --------------------------------------------------------------------------
static int g_failures = 0;

static void check_near(const char* name, double got, double expect, double tol) {
    bool ok = std::isfinite(got) && std::fabs(got - expect) <= tol;
    printf("[%s] %-26s got %14.10f  expected %14.10f  (tol %.0e)\n",
           ok ? "PASS" : "FAIL", name, got, expect, tol);
    if (!ok) ++g_failures;
}

static void check_pred(const char* name, double got, bool ok, const char* desc) {
    printf("[%s] %-26s got %14.6f  (%s)\n",
           ok ? "PASS" : "FAIL", name, got, desc);
    if (!ok) ++g_failures;
}

// --------------------------------------------------------------------------
int main() {
    // LJ pair-equilibrium separation: V(R) is exactly the well minimum, -1.
    const double R = std::pow(2.0, 1.0 / 6.0);

    std::vector<std::vector<double>> two = {
        { 0, 0, 0,   R, 0, 0 },                 // A: dimer at the well bottom
        { 5, 5, 5,   5 + R, 5, 5 },             // B: same dimer, translated
        { 0, 0, 0,   0.1, 0, 0 },               // C: overlapping atoms
        { 0, 0, 0,   10.0, 0, 0 },              // D: far-apart atoms
    };
    std::vector<double> e2 = run_lj(two, 2);

    check_near("dimer at r_min",        e2[0], -1.0, 1e-9);
    check_near("dimer translated",      e2[1], -1.0, 1e-9);   // translation invariance
    check_pred("overlap -> finite",     e2[2],
               std::isfinite(e2[2]) && e2[2] > 0.0,
               "finite & positive (repulsive, floor guard works)");
    check_pred("far apart -> ~0",       e2[3],
               e2[3] < 0.0 && e2[3] > -1e-3,
               "small negative (attractive tail)");

    std::vector<std::vector<double>> three = {
        // equilateral triangle, side R: 3 pairs each at -1  ->  total -3
        { 0, 0, 0,
          R, 0, 0,
          R / 2.0, R * std::sqrt(3.0) / 2.0, 0 },
    };
    std::vector<double> e3 = run_lj(three, 3);

    check_near("equilateral triangle",  e3[0], -3.0, 1e-9);

    // // n_atoms = 2, dim = 6, one individual 
    // std::vector<double> com_in = { 0,0,0,  2,4,6 };
    // double *d_points = nullptr, *d_com_out = nullptr;
    // cudaMalloc(&d_points, sizeof(double) * com_in.size());
    // cudaMemcpy(d_points, com_in.data(), sizeof(double) * com_in.size(), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_com_out, sizeof(double) * 3);

    // test_com_kernel_aos<<<1,1>>>(d_points, d_com_out, 2);
    // cudaDeviceSynchronize();

    // double h_com[3] = {0.0, 0.0, 0.0};
    // cudaMemcpy(h_com, d_com_out, sizeof(double) * 3, cudaMemcpyDeviceToHost);

    // check_near("COM x", h_com[0], 1.0, 1e-9);
    // check_near("COM y", h_com[1], 2.0, 1e-9);
    // check_near("COM z", h_com[2], 3.0, 1e-9);

    // cudaFree(d_points);
    // cudaFree(d_com_out);

    // ---- summary ----
    printf("\n%s  (%d failure%s)\n",
           g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
