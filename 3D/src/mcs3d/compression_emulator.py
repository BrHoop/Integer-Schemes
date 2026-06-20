"""Phase-4 BFP compression emulator on the 3D MCS RHS (Step 4.1).

Faithfully models "intermediates stored compressed, computed in fp64" by wrapping the
derivative operator so every derivative output is round-tripped through BFP storage of a
chosen mantissa width *before* it is consumed by the (fp64) algebra. The RHS body in
`main.py` is untouched — every derivative routes through `diff_op`, so wrapping `diff_op`
captures the analog of the BSSN kernel's 138 live (compressible) derivatives, per field.

This is the CPU numerics half of Step 4.1: it answers "does BFP-k storage of the
intermediates change the answer?" with no GPU. The grouping here is per-field-per-derivative
(each derivative array gets its own block exponent inside the per-field vmap) — adequate for
MCS, whose fields are all O(1); the heterogeneous-scale grouping stress lives in BSSN.

See docs/phases/phase_4_compression/step_4.1_cpu_emulator.md.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mcs_common.bfp_compress import QuantizingDeriv
from mcs_common.wave_state import WaveState
from mcs3d.main import MaxwellChernSimons3D, get_physical, l2norm
from mcs3d.validate import FullBirefringentOracle

# Mantissa width at/above which compression is a no-op (fp64 reference path).
_FP64 = 53


class CompressedMCS3D(MaxwellChernSimons3D):
    """MCS solver whose derivative intermediates are stored at `mant_bits` precision.

    ``mant_bits >= 53`` is the exact fp64 reference (the wrapper passes through), so the
    reference and compressed runs share one code path — an honest A/B.
    """

    def __init__(self, *a, mant_bits=_FP64, **k):
        self._mant_bits = mant_bits
        super().__init__(*a, **k)

    def _init_derivative_operator(self):
        super()._init_derivative_operator()
        self.diff_op = QuantizingDeriv(self.diff_op, self._mant_bits)


def default_params(N=16, order=6, cfl=0.25, Lambda=0.4):
    """A periodic single-wavelength box carrying the analytic birefringent wave."""
    L = 2.0 * np.pi
    return {
        "cfl": cfl, "Order": order, "scheme": "floating_point",
        "Nx": N, "Ny": N, "Nz": N,
        "xmin": 0.0, "xmax": L, "ymin": 0.0, "ymax": L, "zmin": 0.0, "zmax": L,
        "bc_type": "periodic", "enable_cs": 1.0, "Lambda": Lambda,
        "id_amp": 1.0, "ko_sigma": 0.05, "K1": 1.0, "K2": 1.0,
    }


def build_sim(params, mant_bits=_FP64):
    L = params["xmax"] - params["xmin"]
    dx = L / params["Nx"]
    return CompressedMCS3D(dx, dx, dx, params["Lambda"], params, mant_bits=mant_bits)


def initial_state(sim, oracle, t=0.0):
    """Exact field stack (incl. ghost zones) at time `t` — the IC and the oracle."""
    return WaveState(jnp.asarray(oracle.state(sim.X, sim.Y, sim.Z, t)))


def single_eval_rhs_error(params, mant_bits):
    """Global relative L2 error of ONE compressed RHS eval vs fp64, over the interior stack.

    Both sims start from the exact birefringent state; only the derivative storage width
    differs. The norm is taken over ALL fields jointly (NOT per-field): several fields have a
    near-zero RHS by constraint (dt_Psi, dt_Phi at t=0), so a per-field relative error divides
    tiny-by-tiny and is meaningless. The compression perturbation relative to the overall RHS
    magnitude is the physically meaningful quantity; secular effects on the near-zero
    constraint channels are caught by the long-run gate, not here.
    """
    oracle = FullBirefringentOracle(params)
    ref = build_sim(params, mant_bits=_FP64)
    comp = build_sim(params, mant_bits=mant_bits)
    state = initial_state(ref, oracle)

    ng = ref.ng
    a = get_physical(ref.rhs(state).data, ng)
    b = get_physical(comp.rhs(state).data, ng)
    return float(l2norm(b - a)) / float(l2norm(a))


def run_evolution(params, mant_bits, n_steps):
    """Evolve `n_steps` RK4 steps; return (final WaveState, sim) for the long-run gate."""
    oracle = FullBirefringentOracle(params)
    sim = build_sim(params, mant_bits=mant_bits)
    state = initial_state(sim, oracle)
    step = jax.jit(lambda s: sim.step_rk4(s, sim.dt))
    for _ in range(n_steps):
        state = step(state)
    return state, sim


def evolution_error_vs_analytic(params, mant_bits, n_steps):
    """Relative L2 error (Ez interior) vs the analytic solution after `n_steps`."""
    state, sim = run_evolution(params, mant_bits, n_steps)
    oracle = FullBirefringentOracle(params)
    exact = oracle.state(sim.X, sim.Y, sim.Z, n_steps * sim.dt)
    ng = sim.ng
    ez_num = get_physical(state.data[sim.EZ], ng)
    ez_exact = get_physical(jnp.asarray(exact[sim.EZ]), ng)
    return float(l2norm(ez_num - ez_exact)) / float(l2norm(ez_exact))
