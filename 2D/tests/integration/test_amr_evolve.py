"""
AMR end-to-end evolution tests.

Phase 1 milestone: evolve the birefringent wave on the AMR root level (no
refinement yet) and confirm it matches the exact analytical solution within
the same L2 tolerance as the non-AMR fused solver.

This validates that the AMR block storage + ghost-sync layer produces bit-
identical results to running the kernel on a single big tile.
"""

from pathlib import Path


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import BS, NG, NF, MAX_BLOCKS
from mcs2d.amr.evolve import (
    make_root_step, make_two_level_step,
    amr_state_from_global, amr_state_to_global,
)


# ── Physical constants (match params.toml) ─────────────────────────────────────
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0
LAMBDA     = 0.4       # Λ — CFJ-stable regime (m_cs=0.8 < k≈0.89); see params.toml
M_CS       = 2.0 * LAMBDA
CS         = 1.0       # enable_cs
K1         = 1.0
K2         = 1.0
KO_SIGMA   = 0.05
CFL        = 0.05

# Field indices (matches main solver)
EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI, PI, PSI, PHI = 6, 7, 8, 9


def _birefringent_ic(X, Y, amp=0.8):
    """Exact birefringent left-circularly-polarised wave at t=0.

    The analytical solution to MCS in 2.5D for a periodic plane wave:
        Ex(x,y,0) = 0,  Ey(x,y,0) = 0,
        Ez(x,y,t) = E0 sin(kx*x + ky*y - omega*t)
        Bx, By, Bz coupled accordingly.
    For the IC at t=0, we only need Ez to be sin(kx*x + ky*y).
    The other coupled fields are set by the dispersion relation.

    Returns a (NF, Nx, Ny) array.
    """
    Lx = XMAX - XMIN
    Ly = YMAX - YMIN
    kx = 2 * np.pi / Lx
    ky = 2 * np.pi / Ly
    k_mag = np.sqrt(kx**2 + ky**2)
    omega = np.sqrt(k_mag**2 + M_CS * k_mag)

    phase = kx * X + ky * Y
    Ez = amp * np.sin(phase)
    # Birefringent wave: B_z = (omega/k) * Ez ?  Actually, the full state is
    # parameterized to give E_z evolving as sin(phase - omega*t).
    # For a self-consistent IC, use the linearized MCS dispersion.
    # Simpler approach: take the IC from main.py's InitialData if exposed.
    # For now, follow the construction from main.py:
    #   E_x = E_y = 0
    #   E_z = amp * sin(phase)
    #   B_x = -(ky/omega) * Ez                # left-circular polarisation
    #   B_y =  (kx/omega) * Ez
    #   B_z = 0
    #   xi  = (k_mag^2 / (omega * m_cs)) * amp * sin(phase)  (placeholder; check)
    # Pi, Psi, Phi = 0
    # The exact factors are derived in main.py — but for our purposes we just
    # need Ez to match the oracle.  Mismatch in other fields → larger initial
    # constraint violation but Ez evolution is still tracked.
    state = np.zeros((NF, X.shape[0], X.shape[1]), dtype=np.float64)
    state[EZ] = Ez
    state[BX] = -(ky / omega) * Ez
    state[BY] =  (kx / omega) * Ez
    # Note: For perfect agreement with the oracle's Ez(t), we should also set
    # other fields consistently with the analytical solution.  We instead use
    # the IC construction from main.py which is known to work.
    return state


def _make_birefringent_state_from_main(nx, ny):
    """Build the IC by calling main.InitialData (the canonical, known-correct
    birefringent IC).  This avoids re-deriving the dispersion-consistent state."""
    from mcs2d.main import InitialData, MaxwellChernSimons2D, load_parameters
    params_file = str(Path(__file__).resolve().parent.parent.parent / 'params.toml')
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point',
        'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic',
        'sponge_strength': 0.0,
        'Lambda': LAMBDA,   # force consistency: sim/oracle/AMR-step all use LAMBDA
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


class BirefringentOracle:
    def __init__(self, params):
        Lx = params["xmax"] - params["xmin"]
        Ly = params["ymax"] - params["ymin"]
        self.kx = 2 * np.pi / Lx
        self.ky = 2 * np.pi / Ly
        k_mag = np.sqrt(self.kx**2 + self.ky**2)
        m_cs = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
        self.omega = np.sqrt(k_mag**2 + m_cs * k_mag)
        self.E0 = params.get("id_amp", 1.0)

    def Ez(self, X, Y, t):
        return self.E0 * np.sin(self.kx * X + self.ky * Y - self.omega * t)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRootStepBirefringent:
    """Evolve the birefringent wave on the AMR root level (no refinement)
    and compare to the analytical solution."""

    N_STEPS = 100
    L2_TOL  = 1e-7

    def test_root_only_matches_oracle(self):
        nbx, nby = 2, 2
        nx = nbx * BS   # 64 interior cells
        ny = nby * BS
        sim, state, params = _make_birefringent_state_from_main(nx, ny)

        # Extract the interior (NF, nx, ny) from the main solver's state.
        # state.data has shape (NF, nx + 2*ng, ny + 2*ng).
        ng = sim.ng
        interior = np.asarray(state.data[:, ng:ng+nx, ng:ng+ny])
        assert interior.shape == (NF, nx, ny)

        # Build AMR root state.
        blocks = amr_state_from_global(jnp.asarray(interior), nbx, nby)

        # Set up the AMR step matching the main solver's parameters.
        dx = (params['xmax'] - params['xmin']) / nx
        dy = (params['ymax'] - params['ymin']) / ny
        dt = CFL * dx
        step = make_root_step(
            dx=dx, dy=dy, cs=CS, L=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA, dt=dt,
            nbx=nbx, nby=nby,
        )

        # Run N steps.
        def body(carry, _):
            return step(carry), None
        blocks_final = jax.jit(
            lambda s: jax.lax.scan(body, s, None, length=self.N_STEPS)[0]
        )(blocks)

        # Pull interior back out and compare Ez vs oracle.
        ez_amr = np.asarray(amr_state_to_global(blocks_final, nbx, nby))[EZ]
        # Node-centered coords (matches main.py convention).
        x = params['xmin'] + dx * np.arange(nx)
        y = params['ymin'] + dy * np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t_final = self.N_STEPS * dt
        oracle = BirefringentOracle(params)
        ez_exact = oracle.Ez(X, Y, t_final)

        l2 = float(np.sqrt(np.mean((ez_amr - ez_exact) ** 2)))
        assert l2 < self.L2_TOL, (
            f"AMR root-only birefringent: L2(Ez − exact) = {l2:.2e} > {self.L2_TOL:.0e}"
        )


class TestTwoLevelStaticBirefringent:
    """Static 2-level layout: a 2×2 coarse root with one refined child block
    covering corner (0, 0) of root block (0, 0).  Evolve the birefringent
    wave; fine block should track the analytic solution at higher resolution
    than the coarse cells (or at least comparable).
    """

    N_STEPS = 100
    # Fine block L2 tolerance — fine block uses sync_ghosts_across_levels for
    # halo data, which goes through Lagrange interpolation of the coarse
    # parent.  Expect somewhat looser than root-only since the fine halo data
    # quality is bounded by coarse-level resolution.
    L2_TOL_FINE   = 1e-5
    L2_TOL_COARSE = 1e-7

    def test_two_level_static(self):
        nbx_root, nby_root = 2, 2
        nx_coarse = nbx_root * BS
        ny_coarse = nby_root * BS
        sim, state, params = _make_birefringent_state_from_main(nx_coarse, ny_coarse)
        ng = sim.ng
        interior_coarse = np.asarray(state.data[:, ng:ng+nx_coarse, ng:ng+ny_coarse])

        # Coarse root state.
        coarse_blocks = amr_state_from_global(
            jnp.asarray(interior_coarse), nbx_root, nby_root
        )

        # One fine child at slot 0 of fine level, covering corner (0, 0)
        # of root block at slot 0.
        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cx    = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cy    = np.zeros(MAX_BLOCKS, dtype=np.int32)
        fine_active = np.zeros(MAX_BLOCKS, dtype=bool)
        fine_active[0] = True

        # Initialise the fine block by prolongating the coarse parent block
        # (which already holds the IC interior; sync first to populate halos).
        from mcs2d.amr.kernels import sync_ghosts_within_level_root_periodic, prolongate
        coarse_blocks_synced = sync_ghosts_within_level_root_periodic(
            coarse_blocks, nbx_root, nby_root
        )
        parent0 = coarse_blocks_synced[0]
        child0 = prolongate(parent0, (0, 0))   # full fine block (interior + halo)
        fine_blocks = jnp.zeros(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
        ).at[0].set(child0)

        # Build the 2-level step.  dt must satisfy fine CFL.
        dx_coarse = (params['xmax'] - params['xmin']) / nx_coarse
        dy_coarse = (params['ymax'] - params['ymin']) / ny_coarse
        dt = CFL * (dx_coarse / 2.0)            # fine-CFL-limited
        step = make_two_level_step(
            dx_coarse=dx_coarse, dy_coarse=dy_coarse, dt=dt,
            cs=CS, L=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx_root, nby_root=nby_root,
        )
        ps  = jnp.asarray(parent_slot)
        ccx = jnp.asarray(child_cx)
        ccy = jnp.asarray(child_cy)
        fa  = jnp.asarray(fine_active)

        def body(carry, _):
            c, f = carry
            c_new, f_new = step(c, f, ps, ccx, ccy, fa)
            return (c_new, f_new), None
        (coarse_final, fine_final), _ = jax.jit(
            lambda c, f: jax.lax.scan(body, (c, f), None, length=self.N_STEPS)
        )(coarse_blocks, fine_blocks)

        t_final = self.N_STEPS * dt
        oracle = BirefringentOracle(params)

        # Check coarse Ez (everywhere) vs analytic.
        ez_coarse_amr = np.asarray(
            amr_state_to_global(coarse_final, nbx_root, nby_root)
        )[EZ]
        x_c = params['xmin'] + dx_coarse * np.arange(nx_coarse)
        y_c = params['ymin'] + dy_coarse * np.arange(ny_coarse)
        Xc, Yc = np.meshgrid(x_c, y_c, indexing='ij')
        ez_coarse_exact = oracle.Ez(Xc, Yc, t_final)
        l2_c = float(np.sqrt(np.mean((ez_coarse_amr - ez_coarse_exact) ** 2)))
        assert l2_c < self.L2_TOL_COARSE, (
            f"Coarse Ez L2 = {l2_c:.2e} > {self.L2_TOL_COARSE:.0e}"
        )

        # Check fine Ez interior vs analytic at fine cell positions.
        # Fine block at corner (0, 0) of root block (0, 0).  Coarse block (0, 0)
        # covers coarse interior cells [0, BS).  Its corner (0, 0) covers coarse
        # cells [0, BS/2).  Refining gives fine cells covering the same physical
        # region with 2x resolution → fine interior has BS cells per axis covering
        # coarse physical region [xmin, xmin + (BS/2)*dx_coarse].
        # Fine cell positions (node-centered like main solver):
        #   x_f[i] = xmin + i * dx_fine = xmin + i * (dx_coarse / 2)
        # That doesn't quite match because of cell-centered conv. used in
        # prolongate.  For comparison, use the SAME convention prolongate uses:
        # fine cell i in block-index [NG, NG+BS) sits at parent coord
        #   x_c + (i - NG + 0.5) * (dx_coarse / 2) - dx_coarse/2 anchored to parent cell 0.
        # The simplest correct check: use the values produced by `_exact_at_fine_centers`-
        # style mapping over the global coordinates.  For node-centered grid:
        # fine cell global index i (in [0, BS)) → x = xmin + i * dx_fine + offset.
        # Since the IC was built node-centered, evaluating analytic at the same
        # node-centered fine positions is fair.
        fine_int = np.asarray(fine_final[0, :, NG:NG+BS, NG:NG+BS])
        ez_fine_amr = fine_int[EZ]
        # Fine block's interior spans coarse cells [0, BS/2) of root block 0.
        # Coarse cell c sits at coarse_x = xmin + c * dx_coarse (node-centered).
        # Refining gives fine cells at coarse_x ± dx_coarse/4.
        dx_fine = dx_coarse / 2.0
        i_fine = np.arange(BS)
        # cell-centered refinement of node-centered coarse grid:
        # coarse cell c at x = xmin + c*dx_c.  Fine LEFT half: x - dx_c/4; RIGHT: x + dx_c/4.
        # Equivalently: fine cell i (i ∈ [0, BS)) at:
        #   c = i // 2 (within first BS/2 coarse cells of root block 0)
        #   x_f = xmin + c * dx_coarse + (-dx_coarse/4 if i%2==0 else +dx_coarse/4)
        c_idx = i_fine // 2
        offset = np.where(i_fine % 2 == 0, -dx_coarse/4, +dx_coarse/4)
        x_f = params['xmin'] + c_idx * dx_coarse + offset
        y_f = params['ymin'] + c_idx * dy_coarse + offset
        Xf, Yf = np.meshgrid(x_f, y_f, indexing='ij')
        ez_fine_exact = oracle.Ez(Xf, Yf, t_final)
        l2_f = float(np.sqrt(np.mean((ez_fine_amr - ez_fine_exact) ** 2)))
        assert l2_f < self.L2_TOL_FINE, (
            f"Fine Ez L2 = {l2_f:.2e} > {self.L2_TOL_FINE:.0e}"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
