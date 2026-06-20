// Step 3.2f Increment-1 microbenchmark — warp-cooperative fp64 inverse metric +
// conformal Christoffels, the GO/NO-GO for the warp-cooperative approach.
//
// Two kernels compute the SAME quantity (validated against christoffel_ref.py):
//   * naive_kernel    : 1 thread / point  (the correctness oracle + register baseline).
//   * warpcoop_kernel : g=4 lanes / point — the contraction Sum_l g~^{il} D_{ljk} is a
//                       fp64 WARP-SHUFFLE reduction over a 4-lane group. This is the
//                       mechanism under test: does the lane programming model + fp64
//                       __shfl work, is it correct to round-off, and what does it cost?
//
// What this microbenchmark DOES settle: fp64 __shfl_xor_sync correctness, the 4-lane
// group reduction pattern (mask/alignment), and the shuffle overhead vs the naive sum.
// What it does NOT yet settle: the register-fit-at-scale (inverse+Christoffel is a small
// working set that fits one thread regardless) — that needs the larger first-order trunk
// (inverse -> Christoffel -> CalGt -> Ricci), the natural next extension. The register/
// spill numbers printed here are the baseline the trunk build is measured against.
//
// Build + run: see build.sh / README.md (needs nvcc, an H200, and test_vectors.bin from
//   `python -m bssn3d.cuda.christoffel_ref`). The program self-reports registers, spill,
//   correctness, and timing — paste its stdout back.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); } } while (0)

// symmetric (a,b) -> packed component index: (0,0)0 (0,1)1 (0,2)2 (1,1)3 (1,2)4 (2,2)5
__device__ __forceinline__ int symidx(int a, int b) {
    const int tab[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return tab[a][b];
}

// inverse of the symmetric 3x3 packed metric (general det) -> packed 6 (matches ref)
__device__ __forceinline__ void inverse_metric(const double* g, double* ig) {
    double c0 = g[3] * g[5] - g[4] * g[4];
    double c1 = g[2] * g[4] - g[1] * g[5];
    double c2 = g[1] * g[4] - g[2] * g[3];
    double c3 = g[0] * g[5] - g[2] * g[2];
    double c4 = g[1] * g[2] - g[0] * g[4];
    double c5 = g[0] * g[3] - g[1] * g[1];
    double idet = 1.0 / (g[0] * c0 + g[1] * c1 + g[2] * c2);
    ig[0] = c0 * idet; ig[1] = c1 * idet; ig[2] = c2 * idet;
    ig[3] = c3 * idet; ig[4] = c4 * idet; ig[5] = c5 * idet;
}

// the 6 (j,k) pairs as plain arrays (host sets these once via cudaMemcpyToSymbol)
__device__ __constant__ int g_PJ[6];
__device__ __constant__ int g_PK[6];

// ---------------------------------------------------------------------------
// naive: 1 thread / point
// ---------------------------------------------------------------------------
__global__ void naive_kernel(const double* __restrict__ in,
                             double* __restrict__ out, int npts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    const double* g = in + (size_t)p * 24;
    const double* dg = g + 6;                 // 18 derivatives, dg[m*6 + comp]
    double ig[6];
    inverse_metric(g, ig);
    for (int i = 0; i < 3; ++i) {
        for (int pp = 0; pp < 6; ++pp) {
            int j = g_PJ[pp], k = g_PK[pp];
            double s = 0.0;
            for (int l = 0; l < 3; ++l) {
                double D = dg[j * 6 + symidx(l, k)] + dg[k * 6 + symidx(l, j)]
                         - dg[l * 6 + symidx(j, k)];
                s += ig[symidx(i, l)] * D;
            }
            out[(size_t)p * 18 + i * 6 + pp] = 0.5 * s;
        }
    }
}

// ---------------------------------------------------------------------------
// warp-cooperative: g=4 lanes / point, contraction over l via shuffle-reduce
// ---------------------------------------------------------------------------
__global__ void warpcoop_kernel(const double* __restrict__ in,
                                double* __restrict__ out, int npts) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int point = t >> 2;                       // 4 lanes per point
    int lane = t & 3;                         // contraction index l in {0,1,2}; 3 = pad
    bool valid = point < npts;
    int pp_pt = valid ? point : 0;            // dummy-read a valid point if past the end
    const double* g = in + (size_t)pp_pt * 24;
    const double* dg = g + 6;

    // inverse metric: cheap, recomputed per lane (the "recompute the cheap shared scalar"
    // philosophy) — avoids inverse-specific shuffles, keeps the demo focused on the
    // contraction reduction.
    double ig[6];
    inverse_metric(g, ig);

    const unsigned mask = 0xffffffffu;        // all lanes participate (no early return)
    for (int i = 0; i < 3; ++i) {
        for (int pp = 0; pp < 6; ++pp) {
            int j = g_PJ[pp], k = g_PK[pp];
            double term = 0.0;
            if (lane < 3) {                   // lane 3 contributes 0 (3-index contraction)
                int l = lane;
                double D = dg[j * 6 + symidx(l, k)] + dg[k * 6 + symidx(l, j)]
                         - dg[l * 6 + symidx(j, k)];
                term = ig[symidx(i, l)] * D;
            }
            // butterfly reduction within the aligned 4-lane group (xor 1 then 2)
            term += __shfl_xor_sync(mask, term, 1);
            term += __shfl_xor_sync(mask, term, 2);
            if (valid && lane == 0)
                out[(size_t)point * 18 + i * 6 + pp] = 0.5 * term;
        }
    }
}

// ---------------------------------------------------------------------------
// host harness: read vectors, report attributes, run, validate, time
// ---------------------------------------------------------------------------
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = fmax(m, fabs(a[i] - b[i]));
    return m;
}

static void report_attrs(const char* name, const void* kernel) {
    cudaFuncAttributes at;
    CK(cudaFuncGetAttributes(&at, kernel));
    printf("   %-18s regs/thread = %3d   spill(local) = %lld B   maxThreads/blk = %d\n",
           name, at.numRegs, (long long)at.localSizeBytes, at.maxThreadsPerBlock);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "test_vectors.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (run christoffel_ref.py first)\n", path); return 1; }
    int64_t npts = 0;
    if (fread(&npts, sizeof(int64_t), 1, f) != 1) { fprintf(stderr, "bad header\n"); return 1; }
    std::vector<double> in((size_t)npts * 24), expected((size_t)npts * 18);
    if (fread(in.data(), sizeof(double), in.size(), f) != in.size() ||
        fread(expected.data(), sizeof(double), expected.size(), f) != expected.size()) {
        fprintf(stderr, "bad body\n"); return 1;
    }
    fclose(f);
    printf(">> microbench_christoffel | %lld points | fp64 | warp-cooperative g=4\n",
           (long long)npts);

    // the (j,k) pair tables
    const int hPJ[6] = {0, 0, 0, 1, 1, 2}, hPK[6] = {0, 1, 2, 1, 2, 2};
    CK(cudaMemcpyToSymbol(g_PJ, hPJ, sizeof(hPJ)));
    CK(cudaMemcpyToSymbol(g_PK, hPK, sizeof(hPK)));

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
        // correctness (1 launch)
        if (coop) warpcoop_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
        else      naive_kernel<<<blocks, threads>>>(d_in, d_out, (int)npts);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        double err = max_abs_err(host_out, expected);
        // timing (many launches)
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
    printf(">> warpcoop / naive time ratio = %.2fx  (shuffle + 4x threads overhead; this\n"
           "   sub-computation is too small to show the register-fit win — see header)\n",
           tc / tn);

    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    return 0;
}
