"""Generate `_bssn_fused_kernel.cuh` — the M4 FUSED BSSN RHS kernel (the thesis target).

M2b showed the standalone 2.5D streaming is sync-bound and the derivative *reads* are L2-served,
not HBM-bound — so the win is **fusion**: compute the 138 derivatives per point from L2-cached
global reads straight into REGISTER scalars (M2a-style, edge-clamp = `mode='edge'`), then run the
1c algebra CSE on them in the SAME kernel, writing the 24 RHS outputs. No derivative HBM
round-trip (the 2.8 GB wall-2 write the standalone stage paid is gone), no streaming syncs.

Each derivative is emitted as a NAMED scalar (`grad_0_At0` …) — not an array — so ptxas can keep
them in registers; the CSE (which references exactly those names) then consumes them. This is the
seam (138 deriv regs + ~452 algebra temps → ptxas spills); M4 measures whether that spill is
cheaper than the verbatim-XLA round-trips (31.27 ms baseline). Validates vs `BSSNSolver.rhs`
(verbatim) to round-off.

Run:  `python -m bssn3d.cuda.gen_fused`  ->  writes  _bssn_fused_kernel.cuh
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

from bssn3d._codegen import parse, lower_pow, _LAMBDA_RE, FIELD_INPUTS, RHS_TO_FIELD, DENDRO_CSE
from mcs_common.derivatives import SpatialDerivative

_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _smem_plan(statements, K):
    """Phase-4.A: pick the K longest-live-range algebra temps to park in SMEM (the ones most
    likely to be ptxas-spilled — born early, used late). Returns (staged_set, slot_map). Live
    range = last_use_index - def_index in emission order; outputs (RHS_TO_FIELD) count as a final
    use. Staging these into per-thread SMEM removes them from the register file -> less spill, at
    the cost of SMEM traffic + occupancy (the mechanism this prototype measures)."""
    if K <= 0:
        return set(), {}
    pos = {lhs: i for i, (lhs, _) in enumerate(statements)}
    last = dict(pos)
    for i, (_lhs, rhs) in enumerate(statements):
        for tok in set(_ID_RE.findall(rhs)):
            if tok in pos:
                last[tok] = i
    nlast = len(statements)
    for tok in RHS_TO_FIELD:                      # outputs use their temp at the very end
        if tok in pos:
            last[tok] = nlast
    span = {t: last[t] - pos[t] for t in pos}
    staged = sorted(span, key=lambda t: (-span[t], t))[:K]
    return set(staged), {t: i for i, t in enumerate(staged)}

OUT = Path(__file__).resolve().parent / "_bssn_fused_kernel.cuh"

# S[0..12] = algebra scalars (same order as gen_algebra_cuda); S[13..15] = dx, dy, dz (for invh).
_SCALAR_ORDER = ["eta", "BSSN_CAHD_C", "dt", "dx_i", "h_ssl", "sig_ssl", "t",
                 "lmbda0", "lmbda1", "lmbda2", "lmbda3", "lambda_f0", "lambda_f1"]


def _off(axis, k, reach=3):              # neighbour k-reach along `axis`, 0 elsewhere
    o = [0, 0, 0]
    o[axis] = k - reach
    return o


def _off2(i, j, a, b):                   # neighbour a-3 along i, b-3 along j
    o = [0, 0, 0]
    o[i] = a - 3
    o[j] = b - 3
    return o


def _fc(fi, o):
    return f"FC({fi},{o[0]},{o[1]},{o[2]})"


def _otok(o):                            # offset -> identifier token, e.g. (-3,0,1) -> m3_p0_p1
    t = lambda v: ("m%d" if v < 0 else "p%d") % abs(v)
    return f"{t(o[0])}_{t(o[1])}_{t(o[2])}"


def _deriv_exprs_legacy(grad1, grad2):
    """Pre-v7 emission: a separate `FC()` read per stencil tap (overlapping points re-read).

    Kept as the A/B control for Step 3.5 v7 — select with `BSSN_FUSED_V7=0`. On a spill-bound
    kernel this short-lived-load form may actually beat the windowed form (shorter live ranges →
    less spill), which is exactly what the A/B measures."""
    fidx = {f: k for k, f in enumerate(FIELD_INPUTS)}
    lines = []
    for a, f in grad1:                   # 1st derivative along axis a
        terms = " + ".join(f"C1_{k}*{_fc(fidx[f], _off(a, k))}" for k in range(7))
        lines.append(f"  double grad_{a}_{f} = invh[{a}]*({terms});")
    for i, j, f in grad2:
        if i == j:                       # diagonal 2nd derivative
            terms = " + ".join(f"C2_{k}*{_fc(fidx[f], _off(i, k))}" for k in range(7))
            lines.append(f"  double grad2_{i}_{j}_{f} = invh[{i}]*invh[{i}]*({terms});")
        else:                            # mixed: d1 along i, then d1 along j (nested)
            outer = []
            for b in range(7):
                inner = " + ".join(f"C1_{a}*{_fc(fidx[f], _off2(i, j, a, b))}" for a in range(7))
                outer.append(f"C1_{b}*({inner})")
            lines.append(f"  double grad2_{i}_{j}_{f} = invh[{i}]*invh[{j}]*("
                         + " + ".join(outer) + ");")
    return lines


def _deriv_exprs(grad1, grad2):
    """Emit the 138 derivs as named register scalars.

    DEFAULT = legacy per-tap (`_deriv_exprs_legacy`); the v7 window-once path below is opt-in via
    `BSSN_FUSED_V7=1` and is a MEASURED NO-GO (slower — see the note on the toggle).

    Group derivatives by field; load each DISTINCT neighbour offset of a field exactly once into a
    `w_<field>_<offset>` register, then build every grad1/grad2 of that field from those shared
    values — instead of the old per-derivative `FC()` reads (which re-read overlapping points:
    2352 FC reads vs 1644 unique points, 1.43x). The centre offset (0,0,0) reuses the field's
    already-declared point value (`<field> = FC(fi,0,0,0)`) — bit-identical, one fewer load. The
    `w_*` scalars for a field are referenced only in that field's block, so ptxas recycles their
    registers across fields (added live pressure is bounded by one field's window, <=49 for a mixed
    plane). ptxas-INDEPENDENT: it changes how many loads are *issued*, which ptxas cannot re-add.
    """
    # v7 is a MEASURED NO-GO (2026-06-19): the kernel is spill-bound, and window-once loads lengthen
    # live ranges → MORE spill → ~6.8% SLOWER than legacy (20.47→21.86 ms @128^3; spill stores
    # 12556→15412 B). nvcc already CSEs the redundant LDG.E.64 loads at zero live-range cost. So
    # LEGACY is the production default; v7 stays reachable via BSSN_FUSED_V7=1 only to reproduce the
    # negative A/B. See step_3.5_ptxas_independent.md §v7.
    if os.environ.get("BSSN_FUSED_V7", "0") != "1":
        return _deriv_exprs_legacy(grad1, grad2)
    from collections import defaultdict
    fidx = {f: k for k, f in enumerate(FIELD_INPUTS)}
    g1_by, g2_by = defaultdict(list), defaultdict(list)
    for a, f in grad1:
        g1_by[f].append(a)
    for i, j, f in grad2:
        g2_by[f].append((i, j))

    lines = []
    for f in sorted(set(g1_by) | set(g2_by)):
        fi = fidx[f]
        offs, order = {}, []                         # distinct offsets for this field (insertion order)

        def need(o):                                 # name for offset o, registering the load
            o = tuple(o)
            if o == (0, 0, 0):
                return f                              # reuse the field point value (FC(fi,0,0,0))
            if o not in offs:
                offs[o] = f"w_{f}_{_otok(o)}"
                order.append(o)
            return offs[o]

        exprs = []                                   # (lhs, rhs) derivative lines, built via need()
        for a in sorted(g1_by[f]):                   # 1st derivative along axis a
            terms = " + ".join(f"C1_{k}*{need(_off(a, k))}" for k in range(7))
            exprs.append((f"grad_{a}_{f}", f"invh[{a}]*({terms})"))
        for i, j in sorted(g2_by[f]):
            if i == j:                               # diagonal 2nd derivative
                terms = " + ".join(f"C2_{k}*{need(_off(i, k))}" for k in range(7))
                exprs.append((f"grad2_{i}_{j}_{f}", f"invh[{i}]*invh[{i}]*({terms})"))
            else:                                    # mixed: d1 along i, then d1 along j (nested)
                outer = []
                for b in range(7):
                    inner = " + ".join(f"C1_{a}*{need(_off2(i, j, a, b))}" for a in range(7))
                    outer.append(f"C1_{b}*({inner})")
                exprs.append((f"grad2_{i}_{j}_{f}",
                              f"invh[{i}]*invh[{j}]*(" + " + ".join(outer) + ")"))

        for o in order:                              # window loads first (once each) ...
            lines.append(f"  double {offs[o]} = {_fc(fi, list(o))};")
        for lhs, rhs in exprs:                       # ... then the derivatives that consume them
            lines.append(f"  double {lhs} = {rhs};")
    return lines


# NOTE: KO-fusion was tried and REJECTED on measurement (2026-06-18). Adding the 8th-order KO
# stencil to each output made the RK4 step SLOWER (93→116 ms, 2.07×→1.66×): M4 is compute-bound, so
# folding KO's arithmetic into it costs more (spill/low-occ regime) than the cheap, full-occupancy
# memory-bound separate `evolve._ko` XLA pass it replaced. KO stays a separate pass. See
# memory `bssn-m4-fused-rhs-win`.
def emit_kernel(statements, grad1, grad2) -> str:
    diff = SpatialDerivative(order=6)
    C1 = np.asarray(diff.C1, dtype=np.float64)
    C2 = np.asarray(diff.C2, dtype=np.float64)
    L, w = [], lambda s: L.append(s)
    smem_k = int(os.environ.get("BSSN_FUSED_SMEM_K", "0"))   # Phase-4.A: # temps to park in SMEM
    staged, slot = _smem_plan(statements, smem_k)
    if staged:
        _pat = re.compile(r"\b(" + "|".join(re.escape(s) for s in staged) + r")\b")
        def _subst(s):
            return _pat.sub(lambda m: f"smem[{slot[m.group(1)]}*bDx+tid]", s)
    else:
        def _subst(s):
            return s
    w("// AUTO-GENERATED by bssn3d.cuda.gen_fused — DO NOT EDIT. The M4 fused BSSN RHS:")
    w("// per-point derivatives (registers) + the 1c algebra CSE, one kernel, no deriv HBM trip.")
    w("#pragma once")
    w(f"#define NF {len(FIELD_INPUTS)}")
    w(f"#define NOUT {len(RHS_TO_FIELD)}")
    w(f"#define NSCAL {len(_SCALAR_ORDER) + 3}")          # + dx, dy, dz (13..15)
    w(f"#define NSMEM {len(staged)}")                      # Phase-4.A SMEM-staged temps/thread (0=off)
    for k in range(7):
        w(f"#define C1_{k} ({int(round(float(C1[k])*60))}.0/60.0)")
    for k in range(7):
        w(f"#define C2_{k} ({int(round(float(C2[k])*180))}.0/180.0)")
    w("")
    w("__device__ __forceinline__ int clampi(int v, int hi){ return v<0?0:(v>hi?hi:v); }")
    w("// clamped (edge-pad) read of field fi at neighbour offset (ox,oy,oz):")
    w("#define FC(fi,ox,oy,oz) F[(size_t)(fi)*P + ((size_t)clampi(x+(ox),Sx-1)*Sy "
      "+ clampi(y+(oy),Sy-1))*Sz + clampi(z+(oz),Sz-1)]")
    w("")
    w("__global__ void bssn_rhs_fused(const double* __restrict__ F,")
    w("                               const double* __restrict__ S,")
    w("                               double* __restrict__ OUT, int Sx, int Sy, int Sz,")
    w("                               int dummy_iters) {")
    w("  const long long Ntot = (long long)Sx*Sy*Sz;")
    w("  const long long pp = (long long)blockIdx.x*blockDim.x + threadIdx.x;")
    w("  if (pp >= Ntot) return;")
    w("  const int z = (int)(pp % Sz); const long long tq = pp / Sz;")
    w("  const int y = (int)(tq % Sy); const int x = (int)(tq / Sy);")
    w("  const size_t P = (size_t)Ntot; const size_t p = (size_t)pp;")
    w("  double eta=S[0], BSSN_CAHD_C=S[1], dt=S[2], dx_i=S[3], h_ssl=S[4], "
      "sig_ssl=S[5], t=S[6];")
    w("  double lmbda[4]={S[7],S[8],S[9],S[10]};")
    w("  double lambda_f[2]={S[11],S[12]};")
    w("  double invh[3]={1.0/S[13], 1.0/S[14], 1.0/S[15]};")
    if staged:                                           # Phase-4.A: per-thread SMEM scratch
        w("  extern __shared__ double smem[];")
        w("  const unsigned bDx=blockDim.x, tid=threadIdx.x;")
    for k, name in enumerate(FIELD_INPUTS):              # field point values (algebra reads these)
        w(f"  double {name}=FC({k},0,0,0);")
    L.extend(_deriv_exprs(grad1, grad2))                 # 138 derivs -> register scalars
    declared = set(FIELD_INPUTS) | {f"grad_{a}_{f}" for a, f in grad1} \
        | {f"grad2_{i}_{j}_{f}" for i, j, f in grad2}
    for lhs, rhs in statements:                          # the 1c algebra CSE
        crhs = _subst(lower_pow(_LAMBDA_RE.sub("lmbda", rhs)))  # pow()->muls; staged temps->SMEM
        if lhs in staged:                                # staged temp: lives in SMEM, no register
            w(f"  smem[{slot[lhs]}*bDx+tid]={crhs};")
        else:
            w(f"  {'' if lhs in declared else 'double '}{lhs}={crhs};")
            declared.add(lhs)
    for k, tok in enumerate(RHS_TO_FIELD):               # KO is a SEPARATE pass (evolve._ko)
        w(f"  OUT[{k}*P+p]={_subst(tok)};")
    # --- Phase-4 dummy-ALU feasibility probe (runtime BSSN_FUSED_DUMMY; 0 = exact M4) -----------
    # 4*dummy_iters dependent fp64 FMAs (4-way ILP), seeded from field values so they cannot be
    # constant-folded, sunk through a never-taken branch so they cannot be DCE'd and the RHS result
    # is BIT-UNCHANGED. Placed after the OUT writes. The A/B (DUMMY=0 vs ~algebra-count) decides
    # latency- vs throughput-bound spill: time ~flat => SM has free issue capacity (BFP pack/unpack
    # would hide -> compression viable); time rises ~linearly => issue-bound (compression = the v7
    # scar, NO-GO). See compression_traffic_model.py.
    f0, f1, f2, f3 = FIELD_INPUTS[:4]
    w("  if (dummy_iters > 0) {")
    w(f"    double _d0={f0}+0.11, _d1={f1}+0.22, _d2={f2}+0.33, _d3={f3}+0.44;")
    w("    for (int _i=0;_i<dummy_iters;_i++) {")
    w(f"      _d0=fma(_d0,1.0000000001,{f0}); _d1=fma(_d1,1.0000000001,{f1});")
    w(f"      _d2=fma(_d2,1.0000000001,{f2}); _d3=fma(_d3,1.0000000001,{f3});")
    w("    }")
    w("    if (_d0+_d1+_d2+_d3 == 1.7976931348623157e308) OUT[p] += _d0+_d1+_d2+_d3;")
    w("  }")
    w("}")
    return "\n".join(L) + "\n"


def generate(out: Path = OUT) -> Path:
    statements, grad1, grad2 = parse(DENDRO_CSE)
    out.write_text(emit_kernel(statements, grad1, grad2))
    nmix = sum(1 for i, j, _ in grad2 if i != j)
    mode = "v7 window-once loads" if os.environ.get("BSSN_FUSED_V7", "0") == "1" \
        else "legacy per-tap loads (default)"
    smem_k = int(os.environ.get("BSSN_FUSED_SMEM_K", "0"))
    smem = f", SMEM-staged {smem_k} temps/thread (Phase-4.A)" if smem_k > 0 else ""
    print(f">> wrote {out.name} [{mode}]: {len(statements)} CSE stmts, {len(grad1)} grad1 + "
          f"{len(grad2)} grad2 ({nmix} mixed) derivs, {len(RHS_TO_FIELD)} outputs{smem}")
    return out


if __name__ == "__main__":
    generate()
