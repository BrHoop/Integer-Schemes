"""BSSN right-hand side: derivative bundle + transliterated Dendro-GR algebra.

This is the production CAHD+SSL RHS — the pointwise algebra is exactly Dendro-GR's
generated CSE (``_bssn_rhs_generated.py``, from ``bssneqs_SSL_HD_dxsq.cpp``), fed by
the Phase-1 centred FD (``derivative_bundle``). No KO dissipation, no boundary
enforcement, no conformal re-normalization yet: those are deliberately separate so
the algebra can be bit-compared against Dendro-GR in isolation (the decisive gate).

The CAHD+SSL variant is **not** a pure function of ``state``: the SSL lapse-locking
term carries ``exp(-t^2 / 2 sig_ssl^2)`` (time-dependent) and the CAHD chi-damping
term carries ``dx^2/dt`` (grid/step-dependent). So ``rhs``/``rhs_dict`` take the
current time ``t`` and step ``dt``; the grid spacing ``dx_i`` is read from the grid.
"""

from typing import Dict

import jax.numpy as jnp

from mcs_common.derivatives import SpatialDerivative
from .state import BSSNState, PhysicsParams, VAR_NAMES
from .grid import Grid
from .derivative_bundle import derivative_bundle, field_dict
from ._bssn_rhs_generated import bssn_rhs_algebra as _rhs_verbatim


def _select_algebra(scheme: str):
    """Resolve the pointwise-algebra callable for ``scheme`` (staged imported
    lazily so a missing/un-regenerated staged module never breaks the verbatim
    path)."""
    if scheme == "verbatim":
        return _rhs_verbatim
    if scheme == "staged":
        from ._bssn_rhs_staged import bssn_rhs_algebra as _rhs_staged
        return _rhs_staged
    if scheme == "pallas":
        from ._bssn_rhs_pallas import bssn_rhs_algebra as _rhs_pallas
        return _rhs_pallas
    if scheme == "fused":
        # Derivative-fused kernel (3.2/1.2): computes the FD bundle on-chip, so it
        # takes a different call path (fields + spacings, no D) — see rhs_dict.
        from ._bssn_rhs_fused import bssn_rhs_fused
        return bssn_rhs_fused
    if scheme == "fused_fp64":
        # fp64 + SMEM-trunk sibling of "fused" (no-fp32 path); same call path.
        from ._bssn_rhs_fused_fp64 import bssn_rhs_fused_fp64
        return bssn_rhs_fused_fp64
    raise ValueError(
        f"unknown scheme {scheme!r} (use 'verbatim', 'staged', 'pallas', "
        f"'fused', or 'fused_fp64')")


class BSSNSolver:
    """Wires the FD operator + grid + params into a BSSN RHS evaluator.

    ``scheme`` selects the pointwise-algebra implementation (identical math, all
    fed by the same FD bundle): ``"verbatim"`` is the bit-validated Dendro-GR
    transliteration (the oracle anchor); ``"staged"`` is the Step-3.1 probe variant
    with ``optimization_barrier`` cut points (``_bssn_rhs_staged.py``); ``"pallas"``
    is the Step-3.2c register-resident Pallas kernel realizing the 3.2b materialize/
    recompute schedule (``_bssn_rhs_pallas.py``). Use the A/B to compare
    spill/kernel-count/regime on the H200 and CPU round-off equivalence.
    """

    def __init__(self, grid: Grid, params: PhysicsParams = None, order: int = 6,
                 dt: float = None, scheme: str = "verbatim",
                 precision: str = "fp64"):
        self.grid = grid
        self.params = params or PhysicsParams()
        self.scheme = scheme
        # Mixed-precision probe (Phase 3.2c register-fit lever). "fp64" = baseline;
        # "fp32_contraction" computes the FD bundle in fp64 (preserving the
        # cancellation-sensitive second derivatives) then runs the pointwise algebra
        # in fp32 (the wide Ricci-contraction working set is what overflows the
        # register file — fp32 ~halves it). Output is cast back to fp64. The
        # accuracy/constraint cost is measured before any GPU spill claim.
        if precision not in ("fp64", "fp32_contraction"):
            raise ValueError(f"unknown precision {precision!r}")
        self.precision = precision
        self._algebra = _select_algebra(scheme)
        self.diff_op = SpatialDerivative(order=order)
        if self.diff_op.ng != grid.ng:
            raise ValueError(
                f"FD order {order} needs ng={self.diff_op.ng} but grid has ng={grid.ng}."
            )
        # Default step for the CAHD dx^2/dt factor when no stepper supplies one
        # (e.g. the spill probe, or a static-RHS eval). A CFL-ish 0.25*dx keeps the
        # damping coefficient finite/physical; the actual evolution overrides it.
        self.dt = dt if dt is not None else 0.25 * grid.dx

    def rhs_dict(self, state: BSSNState, t: float = 0.0,
                 dt: float = None) -> Dict[str, jnp.ndarray]:
        """Raw ``{field_name: d/dt array}`` (algebra output, full padded shape).

        ``t`` is the current simulation time (SSL ramp); ``dt`` the step (CAHD
        ``dx^2/dt`` factor) — defaults to ``self.dt``. ``dx_i`` is the x-spacing.
        """
        g = self.grid
        p = self.params
        F = field_dict(state)
        if self.scheme in ("fused", "fused_fp64"):
            # The fused kernels compute the FD bundle on-chip (no derivative_bundle,
            # no HBM round-trip for the 138 derivatives). "fused" selects fp32 via
            # BSSN_PALLAS_FP32 (FD stays fp64 there); "fused_fp64" is fp64-locked with
            # the output-fanout trunk schedule. precision="fp32_contraction" is an
            # algebra-only lever, so it does not apply to either.
            out = self._algebra(
                F, g.dx, g.dy, g.dz, p.eta, p.lmbda, p.lambda_f,
                p.cahd_c, self.dt if dt is None else dt, p.ssl_h, p.ssl_sigma, t,
            )
            return out
        D = derivative_bundle(state, self.diff_op, g.dx, g.dy, g.dz)
        if self.precision == "fp32_contraction":
            # FD already done in fp64 above; drop to fp32 only for the algebra.
            F = {k: v.astype(jnp.float32) for k, v in F.items()}
            D = {k: v.astype(jnp.float32) for k, v in D.items()}
        out = self._algebra(
            F, D, p.eta, p.lmbda, p.lambda_f,
            p.cahd_c, self.dt if dt is None else dt, g.dx, p.ssl_h, p.ssl_sigma, t,
        )
        if self.precision != "fp64":
            out = {k: v.astype(jnp.float64) for k, v in out.items()}
        return out

    def rhs(self, state: BSSNState, t: float = 0.0, dt: float = None) -> BSSNState:
        """d/dt state as a ``BSSNState`` (fields stacked in VAR_NAMES order)."""
        out = self.rhs_dict(state, t, dt)
        return BSSNState(jnp.stack([out[name] for name in VAR_NAMES]))
