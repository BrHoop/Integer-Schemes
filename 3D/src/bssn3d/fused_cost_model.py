"""Fused-kernel cost model — predict, on CPU, whether a fused BSSN RHS kernel can
beat the XLA verbatim baseline *before* spending a Marylou compile/run.

Motivation (2026-06-15). The committed `fused_tiled` kernel (cube haloed tiles,
HP=16/BS=8) lost on H200: **54.29 ms vs verbatim 31.27 ms (1.7x slower)** at
128^3. The cube geometry pays an 8x = ((BS+2NG)/BS)^3 halo-redundancy penalty on
the derivative stage — a *cubic* cost that exists only because all three axes are
tiled. This module asks whether the always-intended **2.5D** geometry (large
horizontal slab, march in z; halo redundancy drops to the 2D ratio) combined with
a wall-B strategy that keeps the per-point working set off HBM in **fp64** can
cross the verbatim baseline.

Two walls (see `docs/algebra.md`, `[[bssn-codegen-staging]]`):
  * Wall A — derivative read / halo redundancy (geometry). 2.5D attacks this.
  * Wall B — per-point register pressure (~138 derivs + ~128 first-order trunk ~=
    266 fp64 values = 532 regs >> 255 -> spill). fp32 would fit but is OFF THE
    TABLE (accuracy over ~10^5 steps). Three fp64 escapes are modelled:
      - "spill"      : 1 thread / point, accept the spill (today's baseline).
      - "smem_trunk" : trunk -> SMEM scratch, recompute the bulk (the `fused_fp64`
                       strategy). Low recompute? No — ~7x ops; and SMEM for the
                       trunk over a large slab busts the budget -> small tiles ->
                       halo redundancy returns.
      - "warp_coop"  : `group` threads share one point; the 266-value set is
                       distributed across their register files (266/group each),
                       contractions become warp-shuffle reductions. The only escape
                       that is simultaneously fp64, register-resident (no spill),
                       *and* recompute-free (1x ops) — at the cost of shuffle
                       traffic and `group`x the threads/point.

This is a SCREENING model, not a simulator. The exact pieces (redundancy, register
fit, occupancy) are computed transparently; the absolute millisecond is calibrated
to the single cube-fused anchor so the *relative* ranking is trustworthy. The
authoritative number is still `profile_regime --compare` on H200.

Run:  `python -m bssn3d.fused_cost_model`
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from ._codegen import parse, DENDRO_CSE, FIELD_INPUTS
from .staging import build_dag, output_fanout, select_trunk_schedule


# ---------------------------------------------------------------------------
# Hardware (NVIDIA H200, Hopper sm_90a) — the fit/occupancy ceilings.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HW:
    name: str = "H200"
    regs_per_thread: int = 255        # usable architectural max
    regs_per_sm: int = 65536          # 64K 32-bit registers / SM
    threads_per_sm: int = 2048        # 64 warps
    smem_per_sm_kb: float = 228.0     # Hopper opt-in dynamic SMEM
    warp: int = 32
    n_sm: int = 132
    fp64_tflops: float = 34.0         # non-tensor-core FP64 peak (~34 TFLOP/s)
    hbm_tbs: float = 4.8              # HBM3e bandwidth


H200 = HW()


# ---------------------------------------------------------------------------
# Per-point algebra/derivative quantities — pulled from the REAL CSE DAG so the
# model can never drift from the emitted RHS.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RHSStats:
    n_grad1: int
    n_grad2_diag: int
    n_grad2_mixed: int
    n_deriv: int                  # = grad1 + grad2 (the wall-2 array count)
    n_temps: int
    algebra_ops: int              # verbatim CSE arithmetic-op total (1x recompute)
    trunk_size: int               # output-fanout>=12 first-order trunk (|M|)
    trunk_recompute_mult: float   # ops multiplier when bulk is recomputed
    n_contractions: int           # proxy: temps whose expr is an index sum (shuffle count)


def gather_stats(min_outfanout: int = 12) -> RHSStats:
    statements, grad1, grad2 = parse(DENDRO_CSE)
    dag = build_dag(statements)
    diag = sum(1 for (i, j, _f) in grad2 if i == j)
    mixed = len(grad2) - diag
    algebra_ops = sum(dag.op_cost.values())
    trunk = select_trunk_schedule(dag, min_outfanout=min_outfanout)
    # Contraction proxy: temps that sum >=3 products (a spatial index contraction)
    # become a warp reduction in the warp_coop layout. Approximate by op_cost: a
    # tensor contraction term has many '+'; use temps with op_cost >= 6 as a proxy.
    n_contr = sum(1 for n in dag.temps if dag.op_cost[n] >= 6)
    return RHSStats(
        n_grad1=len(grad1), n_grad2_diag=diag, n_grad2_mixed=mixed,
        n_deriv=len(grad1) + len(grad2), n_temps=len(dag.temps),
        algebra_ops=algebra_ops, trunk_size=len(trunk.materialize),
        trunk_recompute_mult=trunk.multiplier, n_contractions=n_contr,
    )


# ---------------------------------------------------------------------------
# Config — a candidate kernel design point.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    label: str
    geometry: str               # "cube" | "2p5d"
    tile: int                   # interior side: cube -> BS (3D), 2p5d -> T (horizontal)
    ng: int = 4                 # ghost width (NG=4 production: 8th-order KO)
    wall_b: str = "spill"       # "spill" | "smem_trunk" | "warp_coop"
    group: int = 1              # warp_coop threads/point (1 for the others)
    dtype: str = "fp64"
    mixed_composition: bool = False   # mixed 2nd derivs via two 1D passes (49->~2*tap)
    shuffle_penalty: float = 0.15     # warp_coop: fraction of algebra FLOP added as shuffle cost


def _regs_per_value(dtype: str) -> int:
    return 1 if dtype == "fp32" else 2


# ---- Wall A: derivative redundancy + derivative FLOP -----------------------
def redundancy(cfg: Config) -> float:
    """Halo-redundancy factor on the DERIVATIVE stage (algebra runs interior-only
    in both geometries, so it is never redundant)."""
    r = (cfg.tile + 2 * cfg.ng) / cfg.tile
    return r ** 3 if cfg.geometry == "cube" else r ** 2  # 2.5D streams z -> no z-halo


def deriv_flop_per_pt(cfg: Config, s: RHSStats, fd_order: int = 6) -> float:
    tap = fd_order + 1                       # 6th-order centred -> 7-tap
    f1 = 2 * tap - 1                          # first deriv: tap mul + (tap-1) add
    f2_diag = 2 * tap - 1                     # diagonal 2nd deriv: same 1D stencil
    if cfg.mixed_composition:
        f2_mixed = 2 * (2 * tap - 1)         # two 1D passes
    else:
        f2_mixed = 2 * tap * tap - 1         # full (tap x tap) outer product
    return (s.n_grad1 * f1 + s.n_grad2_diag * f2_diag + s.n_grad2_mixed * f2_mixed)


# ---- Wall B: register working set, recompute, occupancy --------------------
def algebra_recompute_mult(cfg: Config, s: RHSStats) -> float:
    """Algebra op multiplier vs verbatim, set by how the working set is held."""
    if cfg.wall_b == "smem_trunk":
        return s.trunk_recompute_mult        # recompute the ~650-temp bulk (~7x)
    # spill and warp_coop both store everything (no recompute); warp_coop has the
    # register room across lanes, spill pays it back as HBM spill traffic instead.
    return 1.0


def working_set_values(s: RHSStats) -> int:
    """Per-point co-resident fp-value count: 138 derivatives + the first-order trunk."""
    return s.n_deriv + s.trunk_size


# DERIVATIVES ARE ALWAYS fp64 (accuracy decision: 2nd-derivative cancellation; fp32
# is off the table for ~10^5-step BBH runs). So they cost 2 regs each REGARDLESS of
# the algebra precision — and 138 derivs = 276 regs overflow the 255 file ON THEIR
# OWN, before any algebra. This is the spill source the --smi (MEM 92%) confirmed.
DERIV_REGS = 2                              # fp64 derivatives, 2 32-bit regs each
TRANSIENT_VALUES = 40                       # peak live recompute-tree transients (proxy)


def reg_values_per_thread(cfg: Config, s: RHSStats) -> int:
    """fp-VALUE working set one thread must hold (before the warp_coop split)."""
    alg_rpv = _regs_per_value(cfg.dtype)     # algebra precision (fp32 as-built, or fp64)
    deriv_regs = s.n_deriv * DERIV_REGS      # 138 fp64 derivs = 276 regs
    trunk_regs = s.trunk_size * alg_rpv      # first-order trunk at algebra precision
    return deriv_regs, trunk_regs


def regs_per_thread(cfg: Config, s: RHSStats) -> int:
    deriv_regs, trunk_regs = reg_values_per_thread(cfg, s)
    if cfg.wall_b == "warp_coop":
        # the whole working set (derivs + trunk) is distributed across the group,
        # so BOTH the fp64 derivs and the algebra trunk shrink by `group`.
        per_lane = -(-(deriv_regs + trunk_regs) // cfg.group)   # ceil
        return per_lane + 16                 # + loop/index/transient overhead
    if cfg.wall_b == "smem_trunk":
        # trunk -> SMEM, but the 138 fp64 DERIVS stay in registers (276 regs) and
        # overflow by themselves + recompute transients. This is why smem_trunk alone
        # does NOT fit — the as-built fused_tiled is essentially this.
        return deriv_regs + TRANSIENT_VALUES * _regs_per_value(cfg.dtype)
    # spill: one thread holds the whole working set.
    return deriv_regs + trunk_regs


# Spill amplification: a spilled value drags its recompute-tree transients, so the
# real local-memory traffic is much larger than (over-regs * 4 B). Calibrated to the
# MEASURED fp64 one-point spill (~25 KB/pt, A100/H200): 532-reg working set overflows
# the 255 file by 277 regs -> ~25000 B  =>  ~90 B per over-register.
SPILL_B_PER_OVER_REG = 90.0


def spill_bytes_per_pt(cfg: Config, s: RHSStats) -> float:
    over = max(0, regs_per_thread(cfg, s) - H200.regs_per_thread)
    return over * SPILL_B_PER_OVER_REG


MAX_THREADS_PER_BLOCK = 1024                 # hardware cap


def threads_per_block(cfg: Config) -> int:
    if cfg.geometry == "cube":
        base = cfg.tile ** 3                 # one thread per interior point
    else:
        base = cfg.tile * cfg.tile           # one thread per interior column (march z)
    tpb = base * (cfg.group if cfg.wall_b == "warp_coop" else 1)
    return min(tpb, MAX_THREADS_PER_BLOCK)   # larger tiles span multiple blocks


def smem_per_block_kb(cfg: Config, s: RHSStats) -> float:
    """SMEM bytes/block: streamed field halo planes (+ trunk if smem_trunk)."""
    halo = cfg.tile + 2 * cfg.ng
    planes = 2 * cfg.ng + 1                   # z-window for 2nd z-derivative
    rpv_bytes = _regs_per_value(cfg.dtype) * 4
    # Field-STREAMED: only ONE field's halo window is SMEM-resident at a time (the
    # derivatives accumulate into per-point register/lane state as fields stream by).
    if cfg.geometry == "cube":
        field_smem = (halo ** 3) * rpv_bytes                 # one haloed cube
    else:
        field_smem = (halo * halo) * planes * rpv_bytes      # one field's z-window
    trunk_smem = 0.0
    if cfg.wall_b == "smem_trunk":
        # trunk stored per interior point of the in-flight tile
        interior = cfg.tile ** 3 if cfg.geometry == "cube" else cfg.tile * cfg.tile
        trunk_smem = s.trunk_size * interior * rpv_bytes
    return (field_smem + trunk_smem) / 1024.0


def warps_per_sm(cfg: Config, s: RHSStats) -> int:
    """Achieved warps/SM = threads/SM (min over reg, thread, SMEM limits) / 32.

    The register limit is the dominant one here: a spilling kernel is capped at the
    255-reg ceiling (ptxas holds regs back for occupancy), a fitting kernel uses its
    actual reg count. threads/SM = regs_per_sm / regs_per_thread (then capped)."""
    rpt = min(regs_per_thread(cfg, s), H200.regs_per_thread)  # spill caps at 255
    by_regs = H200.regs_per_sm // max(rpt, 1)
    tpb = threads_per_block(cfg)
    smem = smem_per_block_kb(cfg, s)
    blocks_by_smem = int(H200.smem_per_sm_kb // smem) if smem > 0 else 999
    by_smem = blocks_by_smem * tpb
    threads_sm = min(H200.threads_per_sm, by_regs, by_smem)
    return max(1, threads_sm // H200.warp)


# ---- Putting it together: FLOP, occupancy efficiency, predicted time -------
SAT_WARPS = 16          # warps/SM to hide FP64 latency (heuristic)
OCC_FLOOR = 0.40        # min efficiency at very low occupancy (ILP carries it)


def occupancy_eff(cfg: Config, s: RHSStats) -> float:
    w = warps_per_sm(cfg, s)
    return max(OCC_FLOOR, min(1.0, w / SAT_WARPS))


def total_flop_per_pt(cfg: Config, s: RHSStats) -> float:
    d = deriv_flop_per_pt(cfg, s) * redundancy(cfg)
    a = s.algebra_ops * algebra_recompute_mult(cfg, s)
    if cfg.wall_b == "warp_coop":
        a *= (1.0 + cfg.shuffle_penalty)      # shuffle reductions for the contractions
    return d + a


@dataclass
class Prediction:
    cfg: Config
    redundancy: float
    flop_per_pt: float
    regs_per_thread: int
    fits: bool
    hbm_b_per_pt: float
    warps_per_sm: int
    occ_eff: float
    compute_ms: float
    mem_ms: float
    pred_ms: float
    vs_verbatim: float


# Anchors (H200, 128^3, wall-clock per RHS eval).
#   verbatim    = 31.27 ms, MEM 97% -> memory-bound (97-kernel intermediate round-trips)
#   cube fused  = 54.29 ms, MEM 92% -> memory-bound (fp64-derivative SPILL, confirmed --smi)
VERBATIM_MS = 31.27
CUBE_FUSED_MS = 54.29
N3 = 128 ** 3

# Compute side: complex fp64 RHS (long dep chains, 55 divisions, transcendentals)
# achieves only ~10-15% of the 34 TFLOP/s peak. This conservative effective rate is
# the dominant uncertainty on the compute-bound (warp_coop) predictions -> sensitivity
# matters; flagged in the output. (Independently, the broken first cut of this model
# back-solved ~4 TFLOP/s from the anchor, consistent with this.)
EFF_FP64_TFLOPS = 4.0
SPILL_REUSE = 4.0                            # spilled temps re-load/store across uses
L2_MB = 50.0                                 # H200 L2 (the field grid does NOT fit at 128^3)
FIELD_BYTES = 8                              # fp64 STATE (production intent is fp64 fields)


def hbm_bytes_per_pt(cfg: Config, s: RHSStats) -> float:
    """HBM traffic per interior point: field reads (x halo-redundancy IF the field
    working set exceeds L2 -> at 128^3 it does, 24x8x128^3 ~= 400 MB >> 50 MB, so the
    redundant cube/halo reads hit HBM, not L2) + output writes + spill traffic."""
    working_set_mb = 24 * N3 * FIELD_BYTES / 1e6
    read_redundancy = redundancy(cfg) if working_set_mb > L2_MB else 1.0
    field_read = 24 * FIELD_BYTES * read_redundancy
    output_write = 24 * FIELD_BYTES
    spill = spill_bytes_per_pt(cfg, s) * SPILL_REUSE
    return field_read + output_write + spill


def _calibrate_bw(s: RHSStats) -> float:
    """Effective HBM BW s.t. the cube-fused anchor (memory-bound) reproduces 54.29 ms."""
    anchor = Config("anchor", "cube", tile=8, wall_b="spill", dtype="fp32")
    return hbm_bytes_per_pt(anchor, s) * N3 / (CUBE_FUSED_MS * 1e-3)


def predict(cfg: Config, s: RHSStats, bw: float) -> Prediction:
    flop = total_flop_per_pt(cfg, s)
    eff = occupancy_eff(cfg, s)
    rpt = regs_per_thread(cfg, s)
    bytes_pt = hbm_bytes_per_pt(cfg, s)
    compute_ms = flop * N3 / (EFF_FP64_TFLOPS * 1e12 * eff) * 1e3
    mem_ms = bytes_pt * N3 / bw * 1e3
    pred_ms = max(compute_ms, mem_ms)
    return Prediction(
        cfg=cfg, redundancy=redundancy(cfg), flop_per_pt=flop, regs_per_thread=rpt,
        fits=rpt <= H200.regs_per_thread, hbm_b_per_pt=bytes_pt,
        warps_per_sm=warps_per_sm(cfg, s), occ_eff=eff,
        compute_ms=compute_ms, mem_ms=mem_ms, pred_ms=pred_ms,
        vs_verbatim=VERBATIM_MS / pred_ms,
    )


def candidate_configs() -> List[Config]:
    return [
        # the as-built fused_tiled (cube, fp32 algebra, fp64 derivs) — the anchor.
        # MEASURED: 54.29 ms, MEM 92% (spill-bound on the fp64 derivs).
        Config("cube fp32-alg BS8 (as-built)", "cube", 8, wall_b="spill", dtype="fp32"),
        # 2.5D geometry alone (still spills the fp64 derivs -> not the fix)
        Config("2.5D/spill T32 fp64", "2p5d", 32, wall_b="spill", dtype="fp64"),
        Config("2.5D/smem_trunk T32 fp64", "2p5d", 32, wall_b="smem_trunk", dtype="fp64"),
        # warp_coop — distributes the fp64 derivs across the group (the wall-B fix),
        # full fp64 (derivs AND algebra fp64); the user's accuracy-safe target.
        Config("2.5D/warp_coop g4 T32 fp64", "2p5d", 32, wall_b="warp_coop",
               group=4, dtype="fp64"),
        Config("2.5D/warp_coop g8 T32 fp64", "2p5d", 32, wall_b="warp_coop",
               group=8, dtype="fp64"),
        Config("2.5D/warp_coop g4 T32 fp64 +comp", "2p5d", 32, wall_b="warp_coop",
               group=4, dtype="fp64", mixed_composition=True),
        Config("2.5D/warp_coop g8 T64 fp64 +comp", "2p5d", 64, wall_b="warp_coop",
               group=8, dtype="fp64", mixed_composition=True),
    ]


def main() -> None:
    s = gather_stats()
    bw = _calibrate_bw(s)
    W = working_set_values(s)
    print(f">> BSSN fused-kernel cost model | {H200.name} | 128^3 | source={DENDRO_CSE.name}")
    print(f">> RHS DAG: {s.n_temps} temps, {s.algebra_ops} algebra ops, "
          f"{s.n_deriv} derivs ({s.n_grad1} grad1 + {s.n_grad2_diag} grad2-diag "
          f"+ {s.n_grad2_mixed} grad2-mixed)")
    print(f">> first-order trunk (out-fanout>=12): |M|={s.trunk_size}, "
          f"bulk-recompute {s.trunk_recompute_mult:.1f}x")
    print(f">> WALL B: 138 fp64 derivs = {s.n_deriv*DERIV_REGS} regs OVERFLOW the "
          f"{H200.regs_per_thread} file on their own (before algebra) -> spill -> "
          f"the measured MEM 92%")
    print(f">> per-point working set = {s.n_deriv} derivs + {s.trunk_size} trunk = "
          f"{W} values (fp64 = {W*2} regs)")
    print(f">> calibrated effective HBM BW = {bw/1e9:.0f} GB/s "
          f"(cube anchor {CUBE_FUSED_MS} ms, MEM 92%); compute @ {EFF_FP64_TFLOPS} "
          f"TFLOP/s eff fp64")
    print(f">> verbatim baseline = {VERBATIM_MS} ms (MEM 97%; target to beat)\n")

    hdr = (f"   {'config':<34} {'redund':>6} {'reg/thr':>7} {'fit':>5} "
           f"{'HBM B':>6} {'wrp':>4} {'cmp ms':>7} {'mem ms':>7} {'pred':>6} {'xverb':>6}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for cfg in candidate_configs():
        p = predict(cfg, s, bw)
        flag = "yes" if p.fits else "SPILL"
        win = "WIN" if (p.fits and p.vs_verbatim > 1.0) else ("~spill" if not p.fits else "")
        print(f"   {cfg.label:<34} {p.redundancy:6.2f} {p.regs_per_thread:7d} "
              f"{flag:>5} {p.hbm_b_per_pt:6.0f} {p.warps_per_sm:4d} "
              f"{p.compute_ms:7.1f} {p.mem_ms:7.1f} {p.pred_ms:6.1f} "
              f"{p.vs_verbatim:5.2f}x {win}")

    print("\n   redund = derivative halo redundancy (wall A) | reg/thr+fit = wall-B fit")
    print("   cmp/mem ms = compute- vs memory-roofline terms; pred = max of the two")
    print("   xverb>1 = beats verbatim. WIN only for register-RESIDENT (no-spill) configs;")
    print("   spillers' time depends on L2-vs-HBM spill behaviour CPU cannot resolve.")
    print("   Dominant uncertainty = the {0} TFLOP/s eff fp64 rate (compute side).".format(EFF_FP64_TFLOPS))
    print("   Authoritative number = profile_regime --compare on H200.")


if __name__ == "__main__":
    main()
