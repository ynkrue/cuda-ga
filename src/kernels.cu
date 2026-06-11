/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for simulated annealing.
 *
 * @author Yannik Rüfenacht
 *
 * Two launch regimes over the walkers:
 *  - Cooperative (energy, relaxation): one block per walker; the block's threads
 *    cooperate over the walker's atoms to evaluate the O(N^2) energy/forces and
 *    block-reduce. This is what scales to large clusters.
 *  - Flat (init, perturb, accept): one thread per walker; the per-walker work is
 *    O(N) bookkeeping with no cooperation needed.
 *
 * Each walker's coordinates are stored contiguously at coords + walker * dimension.
 */

#include "kernels.cuh"

namespace cusa::kernels {

#define PI 3.14159265358979323846

// Singularity guard for the Lennard-Jones potential, in sigma^2 units.
constexpr double R2_FLOOR = 0.5;

// FIRE local-minimization parameters (Bitzek et al. 2006).
constexpr int    FIRE_MAX_STEPS   = 1000;
constexpr double FIRE_FTOL        = 1e-4;
constexpr double FIRE_DT_START    = 0.05;
constexpr double FIRE_DT_MAX      = 0.20;
constexpr double FIRE_MAX_MOVE    = 0.15;
constexpr int    FIRE_N_MIN       = 5;
constexpr double FIRE_F_INC       = 1.10;
constexpr double FIRE_F_DEC       = 0.50;
constexpr double FIRE_ALPHA_START = 0.10;
constexpr double FIRE_F_ALPHA     = 0.99;


/// Initialization ================================================================= ///
__global__ void init_walkers(double* coords, curandState* states, const Config config) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= config.n_walkers) return;

    // One RNG per walker
    curandState state;
    curand_init(config.seed, w, 0, &state);

    double* c = coords + (size_t)w * config.dimension;
    for (int a = 0; a < config.n_atoms; ++a) {
        double u_r = curand_uniform_double(&state);
        double r = config.init_radius * cbrt(u_r);

        double z = 2.0 * curand_uniform_double(&state) - 1.0; // z in [-1, 1]
        double theta = 2.0 * PI * curand_uniform_double(&state); // theta in [0, 2pi]
        double r_xy = sqrt(1.0 - z*z);

        c[3*a + 0] = r * r_xy * cos(theta);
        c[3*a + 1] = r * r_xy * sin(theta);
        c[3*a + 2] = r * z;
    }

    states[w] = state;
}


/// Energy evaluation ============================================================== ///
__global__ void energy_kernel(const double* coords, double* energy, Config config) {
    int w = blockIdx.x;
    if (w >= config.n_walkers) return;

    int N = config.n_atoms;
    extern __shared__ double s_pos[]; // dimension + blockDim doubles
    double* red = s_pos + config.dimension; // reduction scratch (blockDim)
    const double* c = coords + (size_t)w * config.dimension;

    // Stream this walker's cluster into shared memory.
    for (int i = threadIdx.x; i < config.dimension; i += blockDim.x) {
        s_pos[i] = c[i];
    }
    __syncthreads();

    // Each thread sums the pair energies for a strided subset of atoms.
    double e = 0.0;
    for (int a = threadIdx.x; a < N; a += blockDim.x) {
        double xa = s_pos[3*a + 0];
        double ya = s_pos[3*a + 1];
        double za = s_pos[3*a + 2];

        for (int b = a + 1; b < N; ++b) {
            double dx = xa - s_pos[3*b + 0];
            double dy = ya - s_pos[3*b + 1];
            double dz = za - s_pos[3*b + 2];

            double r2 = dx*dx + dy*dy + dz*dz;
            r2 = fmax(r2, R2_FLOOR);

            double r2_inv = 1.0 / r2;
            double r6_inv = r2_inv * r2_inv * r2_inv;
            double r12_inv = r6_inv * r6_inv;

            e += 4.0 * (r12_inv - r6_inv);
        }
    }

    // Block reduction of the partial energies.
    red[threadIdx.x] = e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        energy[w] = red[0];
    }
}


/// Local relaxation =============================================================== ///
// Fast Inertial Relaxation Engine: damped MD with an adaptive timestep that
// minimizes to a force tolerance. The power (F.v), velocity norm and force norm
// are global over the whole cluster, so they are computed as block-wide
// reductions across the walker's threads.
__global__ void relaxation_kernel(double* coords, Config config) {
    int w = blockIdx.x;
    if (w >= config.n_walkers) return;

    int N = config.n_atoms;
    int D = config.dimension;

    extern __shared__ double s[]; // 3 * dimension + 4 * blockDim doubles
    double* pos = s;          // positions
    double* vel = s + D;      // velocities
    double* frc = s + 2*D;    // forces

    double* c = coords + (size_t)w * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        pos[i] = c[i];
        vel[i] = 0.0;
    }

    // Per-walker FIRE state and broadcast scratch.
    __shared__ double dt, alpha, mix_scale, one_minus_alpha;
    __shared__ int    npos, converged, freeze;
    // Reduction scratch in dynamic shared memory, after pos/vel/frc.
    double* r_p    = s + 3*D;            // power F.v
    double* r_v2   = r_p + blockDim.x;   // |v|^2
    double* r_f2   = r_v2 + blockDim.x;  // |F|^2
    double* r_fmax = r_f2 + blockDim.x;  // max per-atom |f|^2
    if (threadIdx.x == 0) {
        dt = FIRE_DT_START;
        alpha = FIRE_ALPHA_START;
        npos = 0;
        converged = 0;
    }
    __syncthreads();

    for (int step = 0; step < FIRE_MAX_STEPS; ++step) {

        // (1) Forces, tracking this thread's max per-atom force magnitude.
        double local_fmax2 = 0.0;
        for (int a = threadIdx.x; a < N; a += blockDim.x) {
            double xa = pos[3*a + 0];
            double ya = pos[3*a + 1];
            double za = pos[3*a + 2];

            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (int b = 0; b < N; ++b) {
                if (b == a) continue;

                double dx = xa - pos[3*b + 0];
                double dy = ya - pos[3*b + 1];
                double dz = za - pos[3*b + 2];

                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < R2_FLOOR) r2 = R2_FLOOR;

                double r2_inv = 1.0 / r2;
                double r6_inv = r2_inv * r2_inv * r2_inv;

                // Lennard-Jones force magnitude / distance.
                double fm = 24.0 * r2_inv * r6_inv * (2.0 * r6_inv - 1.0);
                fx += fm * dx;
                fy += fm * dy;
                fz += fm * dz;
            }

            frc[3*a + 0] = fx;
            frc[3*a + 1] = fy;
            frc[3*a + 2] = fz;

            double fa2 = fx*fx + fy*fy + fz*fz;
            if (fa2 > local_fmax2) local_fmax2 = fa2;
        }
        __syncthreads();

        // (2) Block reductions: power, |v|^2, |F|^2 (sums) and max per-atom |f|^2.
        double pp = 0.0, vv = 0.0, ff = 0.0;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            pp += frc[i] * vel[i];
            vv += vel[i] * vel[i];
            ff += frc[i] * frc[i];
        }
        r_p[threadIdx.x] = pp;
        r_v2[threadIdx.x] = vv;
        r_f2[threadIdx.x] = ff;
        r_fmax[threadIdx.x] = local_fmax2;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                r_p[threadIdx.x]  += r_p[threadIdx.x + stride];
                r_v2[threadIdx.x] += r_v2[threadIdx.x + stride];
                r_f2[threadIdx.x] += r_f2[threadIdx.x + stride];
                if (r_fmax[threadIdx.x + stride] > r_fmax[threadIdx.x]) {
                    r_fmax[threadIdx.x] = r_fmax[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        // (3) Convergence test + FIRE timestep/mixing update (thread 0).
        if (threadIdx.x == 0) {
            if (sqrt(r_fmax[0]) < FIRE_FTOL) {
                converged = 1;
            } else {
                double P = r_p[0];
                if (P > 0.0) {
                    if (npos > FIRE_N_MIN) {
                        dt = fmin(dt * FIRE_F_INC, FIRE_DT_MAX);
                        alpha *= FIRE_F_ALPHA;
                    }
                    npos++;
                    freeze = 0;
                } else {
                    dt *= FIRE_F_DEC;
                    alpha = FIRE_ALPHA_START;
                    npos = 0;
                    freeze = 1; // uphill: reset velocity
                }
                // v <- (1-alpha) v + alpha |v| F/|F|
                mix_scale = (r_f2[0] > 0.0) ? alpha * sqrt(r_v2[0] / r_f2[0]) : 0.0;
                one_minus_alpha = 1.0 - alpha;
            }
        }
        __syncthreads();
        if (converged) break;

        // (4) Velocity mixing.
        if (freeze) {
            for (int i = threadIdx.x; i < D; i += blockDim.x) vel[i] = 0.0;
        } else {
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                vel[i] = one_minus_alpha * vel[i] + mix_scale * frc[i];
            }
        }
        __syncthreads();

        // (5) Semi-implicit Euler MD step with a per-atom displacement clamp.
        for (int a = threadIdx.x; a < N; a += blockDim.x) {
            double vx = vel[3*a + 0] + dt * frc[3*a + 0];
            double vy = vel[3*a + 1] + dt * frc[3*a + 1];
            double vz = vel[3*a + 2] + dt * frc[3*a + 2];
            vel[3*a + 0] = vx;
            vel[3*a + 1] = vy;
            vel[3*a + 2] = vz;

            double dx = dt * vx;
            double dy = dt * vy;
            double dz = dt * vz;
            double mag = sqrt(dx*dx + dy*dy + dz*dz);
            if (mag > FIRE_MAX_MOVE) {
                double scale = FIRE_MAX_MOVE / mag;
                dx *= scale;
                dy *= scale;
                dz *= scale;
            }
            pos[3*a + 0] += dx;
            pos[3*a + 1] += dy;
            pos[3*a + 2] += dz;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        c[i] = pos[i];
    }
}


/// Perturbation ============================================== ///
__global__ void perturb_kernel(const double* current, double* trial, curandState* states,
                               const double* temps, Config config) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= config.n_walkers) return;

    const double* cur = current + (size_t)w * config.dimension;
    double* tr = trial + (size_t)w * config.dimension;

    // Displace every atom by a uniform random vector in [-step, step]^3, with the
    // step scaled by temperature so hot replicas explore more aggressively.
    double step = config.step_size * temps[w];
    curandState state = states[w];
    for (int a = 0; a < config.n_atoms; ++a) {
        tr[3*a + 0] = cur[3*a + 0] + step * (2.0 * curand_uniform_double(&state) - 1.0);
        tr[3*a + 1] = cur[3*a + 1] + step * (2.0 * curand_uniform_double(&state) - 1.0);
        tr[3*a + 2] = cur[3*a + 2] + step * (2.0 * curand_uniform_double(&state) - 1.0);
    }
    states[w] = state;
}


/// Metropolis acceptance ========================================================== ///
__global__ void accept_kernel(double* current, const double* trial,
                              double* current_energy, const double* trial_energy,
                              double* best, double* best_energy,
                              curandState* states, const double* temps, Config config) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= config.n_walkers) return;

    double te = trial_energy[w];
    double dE = te - current_energy[w];

    curandState state = states[w];
    bool accept = (dE <= 0.0) || (curand_uniform_double(&state) < exp(-dE / temps[w]));
    states[w] = state;

    // On acceptance, the trial becomes the current configuration.
    if (accept) {
        double* cur = current + (size_t)w * config.dimension;
        const double* tr = trial + (size_t)w * config.dimension;
        for (int i = 0; i < config.dimension; ++i) {
            cur[i] = tr[i];
        }
        current_energy[w] = te;
    }

    // Track the best configuration found so far.
    if (current_energy[w] < best_energy[w]) {
        double* bst = best + (size_t)w * config.dimension;
        const double* cur = current + (size_t)w * config.dimension;
        for (int i = 0; i < config.dimension; ++i) {
            bst[i] = cur[i];
        }
        best_energy[w] = current_energy[w];
    }
}


/// Replica exchange =============================================================== ///
__global__ void swap_kernel(double* current, double* current_energy,
                            double* best, double* best_energy,
                            curandState* states, const double* temps,
                            unsigned long long* swap_accepts, Config config) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= config.n_ensembles) return;

    int base = e * config.n_temps;       // index of this ensemble's coldest rung
    int D = config.dimension;
    curandState state = states[base];

    // Walk up the ladder, attempting to swap each adjacent (cold, hot) pair.
    for (int r = 0; r < config.n_temps - 1; ++r) {
        int lo = base + r;       // colder rung (lower T, higher beta)
        int hi = base + r + 1;   // hotter rung

        double E_lo = current_energy[lo];
        double E_hi = current_energy[hi];
        double arg = (1.0 / temps[lo] - 1.0 / temps[hi]) * (E_lo - E_hi);

        if (arg >= 0.0 || curand_uniform_double(&state) < exp(arg)) {
            // Swap configurations between the two rungs.
            double* c_lo = current + (size_t)lo * D;
            double* c_hi = current + (size_t)hi * D;
            for (int i = 0; i < D; ++i) {
                double tmp = c_lo[i];
                c_lo[i] = c_hi[i];
                c_hi[i] = tmp;
            }
            current_energy[lo] = E_hi;
            current_energy[hi] = E_lo;

            // A structure swapped in may be the best either rung has held.
            if (E_hi < best_energy[lo]) {
                double* b = best + (size_t)lo * D;
                for (int i = 0; i < D; ++i) b[i] = c_lo[i];
                best_energy[lo] = E_hi;
            }
            if (E_lo < best_energy[hi]) {
                double* b = best + (size_t)hi * D;
                for (int i = 0; i < D; ++i) b[i] = c_hi[i];
                best_energy[hi] = E_lo;
            }

            atomicAdd(swap_accepts, 1ULL);
        }
    }

    states[base] = state;
}


} // namespace cusa::kernels