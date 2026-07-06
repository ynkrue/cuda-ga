/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for simulated annealing.
 *
 * @author Yannik Rüfenacht
 *
 * Launch regimes:
 *  - Cooperative (energy, relaxation): one block per walker.
 *  - Flat (init, perturb, accept): one thread per walker.
 *  - Swap: one thread per ensemble.
 *
 * Memory layout: coords[walker * dimension + (3*a + c)].
 */

#include "kernels.cuh"

namespace cusa::kernels {

constexpr double PI = 3.14159265358979323846;
constexpr double R2_FLOOR = 0.5; // singularity guard

// FIRE local-minimization parameters (Bitzek et al. 2006), reduced units.
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

    curandState state;
    curand_init(config.seed, w, 0, &state);

    double* c = coords + (size_t)w * config.dimension;
    for (int a = 0; a < config.n_atoms; ++a) {
        double u_r = curand_uniform_double(&state);
        double r = config.init_radius * cbrt(u_r);

        double z = 2.0 * curand_uniform_double(&state) - 1.0;
        double theta = 2.0 * PI * curand_uniform_double(&state);
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
    extern __shared__ double s_pos[];
    double* red = s_pos + config.dimension;
    const double* c = coords + (size_t)w * config.dimension;

    for (int i = threadIdx.x; i < config.dimension; i += blockDim.x) {
        s_pos[i] = c[i];
    }
    __syncthreads();

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
// FIRE: damped MD with adaptive timestep, minimizes to a force tolerance.
__global__ void relaxation_kernel(double* coords, Config config) {
    int w = blockIdx.x;
    if (w >= config.n_walkers) return;

    int D = config.dimension;
    int N = config.n_atoms;

    extern __shared__ double s_mem[];
    double* pos    = s_mem;
    double* vel    = pos + D;
    double* frc    = vel + D;
    double* r_p    = frc + D;
    double* r_v2   = r_p + blockDim.x;
    double* r_f2   = r_v2 + blockDim.x;
    double* r_fmax = r_f2 + blockDim.x;

    double* c = coords + (size_t)w * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        pos[i] = c[i];
        vel[i] = 0.0;
        frc[i] = 0.0;
    }
    // __shared__: broadcasts FIRE state to all threads each step (must not be per-thread).
    __shared__ double dt, alpha, mix_scale, one_minus_alpha;
    __shared__ int    npos, converged, freeze;
    if (threadIdx.x == 0) {
        dt = FIRE_DT_START;
        alpha = FIRE_ALPHA_START;
        one_minus_alpha = 1.0 - alpha;
        mix_scale = 0.0;
        npos = 0;
        freeze = 0;
        converged = 0;
    }
    __syncthreads();

    for (int step = 0; step < config.fire_max_steps; ++step) {
        // Forces (O(N^2)); track max |f| for convergence.
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
                r2 = fmax(r2, R2_FLOOR);
                double r6_inv = pow(r2, -3.0);
                double fm = 4.0 * (12.0 * r6_inv * r6_inv - 6.0 * r6_inv) / r2;

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

        // Block reductions: power, |v|^2, |F|^2, max |f|^2.
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

        // Convergence check + FIRE update (thread 0).
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
                    freeze = 1;
                }
                mix_scale = (r_f2[0] > 0.0) ? alpha * sqrt(r_v2[0] / r_f2[0]) : 0.0;
                one_minus_alpha = 1.0 - alpha;
            }
        }
        __syncthreads();
        if (converged) break;

        // Velocity mixing / reset.
        if (freeze) {
            for (int i = threadIdx.x; i < D; i += blockDim.x) vel[i] = 0.0;
        } else {
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                vel[i] = one_minus_alpha * vel[i] + mix_scale * frc[i];
            }
        }
        __syncthreads();

        // Semi-implicit Euler step with displacement clamp.
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

    if (accept) {
        double* cur = current + (size_t)w * config.dimension;
        const double* tr = trial + (size_t)w * config.dimension;
        for (int i = 0; i < config.dimension; ++i) {
            cur[i] = tr[i];
        }
        current_energy[w] = te;
    }

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

    int base = e * config.n_temps;
    int D = config.dimension;
    curandState state = states[base];

    for (int r = 0; r < config.n_temps - 1; ++r) {
        int lo = base + r;
        int hi = base + r + 1;

        double E_lo = current_energy[lo];
        double E_hi = current_energy[hi];
        double arg = (1.0 / temps[lo] - 1.0 / temps[hi]) * (E_lo - E_hi);

        if (arg >= 0.0 || curand_uniform_double(&state) < exp(arg)) {
            double* c_lo = current + (size_t)lo * D;
            double* c_hi = current + (size_t)hi * D;
            for (int i = 0; i < D; ++i) {
                double tmp = c_lo[i];
                c_lo[i] = c_hi[i];
                c_hi[i] = tmp;
            }
            current_energy[lo] = E_hi;
            current_energy[hi] = E_lo;

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
