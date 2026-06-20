// Step 3.2f Increment-1b microbenchmark — warp-cooperative fp64 CONFORMAL RICCI, the
// TRUE register-fit go/no-go. Ricci(g~) is the first-AND-second-order trunk: a ~100-150
// fp64 working set (inverse, Christoffels G[27], their derivatives dG[81], second-deriv
// contractions) that should push the naive 1-thread layout OVER the 255-register file.
//
//   * naive_kernel    : 1 thread / point — computes the full dG[3][3][3][3] etc. in one
//                       thread's registers -> expected to SPILL (the wall B we are fixing).
//   * warpcoop_kernel : g=4, lane = derivative direction k. Each lane computes only its
//                       k-slice of dG (27 vs 81 values), so the big derivative-of-
//                       Christoffel array is DISTRIBUTED across the group. Ricci's two
//                       derivative contractions are assembled with a shuffle-REDUCE
//                       (Sum_l d_l G^l_ij) and a shuffle-BROADCAST (read lane j for
//                       d_j G^l_il); the derivative-free Gamma*Gamma products are cheap
//                       and recomputed per lane. The go/no-go: does warpcoop stay
//                       register-resident (0 spill) where naive spills?
//
// Math matches ricci_ref.py (validated vs SymPy to 1.1e-16). Inputs per point (60 dbl):
//   g~[6] | d_m g~_c (18, m*6+c) | d_m d_n g~_c (36, symidx(m,n)*6+c). Output: R~[6].
//
// Build/run: see build.sh / README.md. Needs ricci_vectors.bin from `ricci_ref.py`.
// The program self-reports registers, spill, correctness, and timing.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); } } while (0)

__device__ __forceinline__ int symidx(int a, int b) {
    const int tab[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return tab[a][b];
}
__device__ __constant__ int g_PI[6];        // pair first index i
__device__ __constant__ int g_PJ[6];        // pair second index j

__device__ __forceinline__ void inverse_metric(const double* g, double ig[3][3]) {
    double c0 = g[3] * g[5] - g[4] * g[4];
    double c1 = g[2] * g[4] - g[1] * g[5];
    double c2 = g[1] * g[4] - g[2] * g[3];
    double c3 = g[0] * g[5] - g[2] * g[2];
    double c4 = g[1] * g[2] - g[0] * g[4];
    double c5 = g[0] * g[3] - g[1] * g[1];
    double id = 1.0 / (g[0] * c0 + g[1] * c1 + g[2] * c2);
    ig[0][0] = c0 * id; ig[0][1] = c1 * id; ig[0][2] = c2 * id;
    ig[1][0] = c1 * id; ig[1][1] = c3 * id; ig[1][2] = c4 * id;
    ig[2][0] = c2 * id; ig[2][1] = c4 * id; ig[2][2] = c5 * id;
}
// d_m g~_{ab}  and  d_m d_n g~_{ab}  from the packed global input
__device__ __forceinline__ double DG(const double* dg, int m, int a, int b) {
    return dg[m * 6 + symidx(a, b)];
}
__device__ __forceinline__ double DDG(const double* dd, int m, int n, int a, int b) {
    return dd[symidx(m, n) * 6 + symidx(a, b)];
}
// conformal Christoffel G[l][i][j] (cheap; computed in both kernels)
__device__ __forceinline__ void christoffel(const double ig[3][3], const double* dg,
                                            double G[3][3][3]) {
    for (int l = 0; l < 3; ++l)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0.0;
                for (int m = 0; m < 3; ++m)
                    s += ig[l][m] * (DG(dg, i, m, j) + DG(dg, j, m, i) - DG(dg, m, i, j));
                G[l][i][j] = 0.5 * s;
            }
}

// ---------------------------------------------------------------------------
// naive: 1 thread / point — the full trunk in one thread (expected to spill)
// ---------------------------------------------------------------------------
__global__ void naive_kernel(const double* __restrict__ in,
                             double* __restrict__ out, int npts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    const double* g = in + (size_t)p * 60;
    const double* dg = g + 6;
    const double* dd = g + 24;
    double ig[3][3];   inverse_metric(g, ig);
    double G[3][3][3]; christoffel(ig, dg, G);
    // dig[k][l][m] = - g~^{la} g~^{mb} d_k g~_{ab}
    double dig[3][3][3];
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
            for (int m = 0; m < 3; ++m) {
                double s = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        s += ig[l][a] * ig[m][b] * DG(dg, k, a, b);
                dig[k][l][m] = -s;
            }
    // dG[k][l][i][j] = d_k G^l_{ij}
    double dG[3][3][3][3];
    for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double s = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        double t = DG(dg, i, m, j) + DG(dg, j, m, i) - DG(dg, m, i, j);
                        s += dig[k][l][m] * t
                           + ig[l][m] * (DDG(dd, k, i, m, j) + DDG(dd, k, j, m, i)
                                       - DDG(dd, k, m, i, j));
                    }
                    dG[k][l][i][j] = 0.5 * s;
                }
    for (int pp = 0; pp < 6; ++pp) {
        int i = g_PI[pp], j = g_PJ[pp];
        double r = 0.0;
        for (int l = 0; l < 3; ++l) {
            r += dG[l][l][i][j] - dG[j][l][i][l];
            for (int m = 0; m < 3; ++m)
                r += G[l][l][m] * G[m][i][j] - G[l][j][m] * G[m][i][l];
        }
        out[(size_t)p * 6 + pp] = r;
    }
}

// ---------------------------------------------------------------------------
// warp-cooperative: g=4, lane = derivative direction k; dG distributed over lanes
// ---------------------------------------------------------------------------
__global__ void warpcoop_kernel(const double* __restrict__ in,
                                double* __restrict__ out, int npts) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int point = t >> 2, lane = t & 3, k = lane;       // k = derivative direction (3 = pad)
    bool valid = point < npts;
    const double* g = in + (size_t)(valid ? point : 0) * 60;
    const double* dg = g + 6;
    const double* dd = g + 24;
    const unsigned M = 0xffffffffu;

    double ig[3][3];   inverse_metric(g, ig);
    double G[3][3][3]; christoffel(ig, dg, G);        // redundant (cheap), for the GG term

    // this lane's k-slice of dG: dGk[l][i][j] = d_k G^l_{ij}  (27 vs the naive 81)
    double dGk[3][3][3];
    if (k < 3) {
        for (int l = 0; l < 3; ++l)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double s = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        double digklm = 0.0;          // d_k g~^{lm}
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                digklm -= ig[l][a] * ig[m][b] * DG(dg, k, a, b);
                        double tt = DG(dg, i, m, j) + DG(dg, j, m, i) - DG(dg, m, i, j);
                        s += digklm * tt
                           + ig[l][m] * (DDG(dd, k, i, m, j) + DDG(dd, k, j, m, i)
                                       - DDG(dd, k, m, i, j));
                    }
                    dGk[l][i][j] = 0.5 * s;
                }
    } else {
        for (int l = 0; l < 3; ++l) for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) dGk[l][i][j] = 0.0;
    }

    for (int pp = 0; pp < 6; ++pp) {
        int i = g_PI[pp], j = g_PJ[pp];
        // d_l G^l_{ij}: lane k contributes dG[k][k][i][j]; reduce over the 4-lane group
        double divc = dGk[k < 3 ? k : 0][i][j] * (k < 3 ? 1.0 : 0.0);
        divc += __shfl_xor_sync(M, divc, 1, 4);
        divc += __shfl_xor_sync(M, divc, 2, 4);
        // d_j G^l_{il}: this is lane j's value of  Sum_l dG[k][l][i][l]
        double localJ = 0.0;
        if (k < 3) for (int l = 0; l < 3; ++l) localJ += dGk[l][i][l];
        double jterm = __shfl_sync(M, localJ, j, 4);  // read lane j within the group
        // Gamma*Gamma part (derivative-free): recomputed per lane from G
        double gg = 0.0;
        for (int l = 0; l < 3; ++l)
            for (int m = 0; m < 3; ++m)
                gg += G[l][l][m] * G[m][i][j] - G[l][j][m] * G[m][i][l];
        if (valid && lane == 0)
            out[(size_t)point * 6 + pp] = divc - jterm + gg;
    }
}

// ---------------------------------------------------------------------------
// host harness (identical structure to microbench_christoffel.cu)
// ---------------------------------------------------------------------------
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = fmax(m, fabs(a[i] - b[i]));
    return m;
}
static void report_attrs(const char* name, const void* kernel) {
    cudaFuncAttributes at; CK(cudaFuncGetAttributes(&at, kernel));
    printf("   %-18s regs/thread = %3d   spill(local) = %lld B   maxThreads/blk = %d\n",
           name, at.numRegs, (long long)at.localSizeBytes, at.maxThreadsPerBlock);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "ricci_vectors.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (run ricci_ref.py first)\n", path); return 1; }
    int64_t npts = 0;
    if (fread(&npts, sizeof(int64_t), 1, f) != 1) { fprintf(stderr, "bad header\n"); return 1; }
    std::vector<double> in((size_t)npts * 60), expected((size_t)npts * 6);
    if (fread(in.data(), sizeof(double), in.size(), f) != in.size() ||
        fread(expected.data(), sizeof(double), expected.size(), f) != expected.size()) {
        fprintf(stderr, "bad body\n"); return 1;
    }
    fclose(f);
    printf(">> microbench_ricci | %lld points | fp64 | warp-cooperative g=4\n", (long long)npts);

    const int hPI[6] = {0, 0, 0, 1, 1, 2}, hPJ[6] = {0, 1, 2, 1, 2, 2};
    CK(cudaMemcpyToSymbol(g_PI, hPI, sizeof(hPI)));
    CK(cudaMemcpyToSymbol(g_PJ, hPJ, sizeof(hPJ)));

    double *d_in, *d_out;
    CK(cudaMalloc(&d_in, in.size() * sizeof(double)));
    CK(cudaMalloc(&d_out, expected.size() * sizeof(double)));
    CK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(double), cudaMemcpyHostToDevice));

    printf(">> kernel attributes (the register/spill GO/NO-GO):\n");
    report_attrs("naive", (const void*)naive_kernel);
    report_attrs("warpcoop g=4", (const void*)warpcoop_kernel);

    std::vector<double> host_out(expected.size());
    auto run = [&](const char* name, bool coop) {
        CK(cudaMemset(d_out, 0, expected.size() * sizeof(double)));
        int threads = 256;
        long long work = coop ? npts * 4 : npts;
        int blocks = (int)((work + threads - 1) / threads);
        if (coop) warpcoop_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
        else      naive_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        double err = max_abs_err(host_out, expected);
        const int iters = 200;
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int it = 0; it < iters; ++it) {
            if (coop) warpcoop_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
            else      naive_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
        }
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        printf("   %-18s max|err| vs ref = %.2e   time = %.4f ms/launch\n",
               name, err, ms / iters);
        return ms / iters;
    };

    printf(">> correctness + timing:\n");
    double tn = run("naive", false);
    double tc = run("warpcoop g=4", true);
    printf(">> warpcoop / naive time ratio = %.2fx\n", tc / tn);
    printf(">> GO/NO-GO: naive should SPILL (working set > 255 regs); warpcoop g=4 should\n"
           "   stay register-resident (0 spill) — that is the wall-B fix this step tests.\n");

    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    return 0;
}
