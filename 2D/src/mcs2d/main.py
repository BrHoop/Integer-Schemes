# 1. Standard library imports
import os
import sys
import time
import tomllib
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import jax
jax.config.update("jax_enable_x64", True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import jax.numpy as jnp
import numpy as np

import Utils.ioxdmf as iox
from Utils.wave_state import WaveState

class InitialData:
    """Generates initial conditions for the 2D wave simulation."""
    EX, EY, EZ, BX, BY, BZ, XI, PI, PSI, PHI = range(10)

    def __init__(self, sim: Any, params: Dict[str, Any]):
        self.sim = sim
        self.params = params

    def generate(self):
        id_type = self.params.get("id_type", "gaussian")
        generators = {
            "gaussian": self._gaussian_pulse,
            "birefringent": self._birefringent_wave
        }

        if id_type not in generators:
            raise ValueError(f"Initial data type '{id_type}' not recognized.")
        
        return generators[id_type]()

    def _gaussian_pulse(self):
        x0, y0 = self.params.get("id_x0", 0.0), self.params.get("id_y0", 0.0)
        B0, sigma = self.params.get("id_amp", 0.8), self.params.get("id_sigma", 0.5)
        
        psi = B0 * jnp.exp(-((self.sim.x[:, None] - x0) ** 2 + (self.sim.y[None, :] - y0) ** 2) / (2 * sigma**2))
        
        data = jnp.zeros((10, self.sim.Nx_tot, self.sim.Ny_tot), dtype=jnp.float64)
        
        data = data.at[self.EX].set(self.sim.d_dy(psi))
        data = data.at[self.EY].set(-self.sim.d_dx(psi))
        data = data.at[self.EZ].set(psi * 0.1)
        
        data = data.at[self.BX].set(data[self.EX])
        data = data.at[self.BY].set(data[self.EY])
        data = data.at[self.BZ].set(psi * 0.1)
        
        data = data.at[self.XI].set(0.0)
        data = data.at[self.PI].set(0.0)
        data = data.at[self.PSI].set(0.0)
        data = data.at[self.PHI].set(0.0)
        
        return WaveState(data)

    def _birefringent_wave(self):
        """Analytical 2.5D Left-Circularly Polarized wave for exact testing."""
        E0 = self.params.get("id_amp", 1.0)
        
        # We need the CS parameters to perfectly set the Pi field
        cs = self.params.get("enable_cs", 1.0)
        L_param = self.params.get("Lambda", 1.0)
        m_cs = self.params.get("id_m_cs", L_param * 2.0) 
        
        Lx = self.params.get("xmax", 5.0) - self.params.get("xmin", -5.0)
        Ly = self.params.get("ymax", 5.0) - self.params.get("ymin", -5.0)
        
        k_x = 2.0 * jnp.pi / Lx
        k_y = 2.0 * jnp.pi / Ly
        k = jnp.sqrt(k_x**2 + k_y**2)
        
        omega_minus = jnp.sqrt(k**2 + m_cs * k)
        
        e1_x = -k_y / k
        e1_y = k_x / k
        
        Phi = k_x * self.sim.X + k_y * self.sim.Y
        cos_Phi = jnp.cos(Phi)
        sin_Phi = jnp.sin(Phi)
        
        Ex = E0 * e1_x * cos_Phi
        Ey = E0 * e1_y * cos_Phi
        Ez = E0 * sin_Phi 
        
        Bx = (k_y * Ez) / omega_minus
        By = (-k_x * Ez) / omega_minus
        Bz = (k_x * Ey - k_y * Ex) / omega_minus
        
        # THE FIX: Calculate the required Pi field to sustain the m_cs background
        Pi_0 = m_cs / (2.0 * cs * L_param)
        
        data = jnp.zeros((10, self.sim.Nx_tot, self.sim.Ny_tot), dtype=jnp.float64)
        
        data = data.at[self.EX].set(Ex)
        data = data.at[self.EY].set(Ey)
        data = data.at[self.EZ].set(Ez)
        
        data = data.at[self.BX].set(Bx)
        data = data.at[self.BY].set(By)
        data = data.at[self.BZ].set(Bz)
        
        data = data.at[self.XI].set(0.0)
        data = data.at[self.PI].set(Pi_0)   # <--- Injected here
        data = data.at[self.PSI].set(0.0)
        data = data.at[self.PHI].set(0.0)
        
        return WaveState(data)

class MaxwellChernSimons2D:
    """Solves the 2D Maxwell-Chern-Simons equations."""
    EX, EY, EZ = 0, 1, 2
    BX, BY, BZ = 3, 4, 5
    XI, PI = 6, 7
    PSI, PHI = 8, 9

    def __init__(self, dx: float, dy: float, Lambda: float, params: Dict[str, Any]):
        self.dx, self.dy = dx, dy
        self.Lambda = Lambda
        self.params = params
        self.dt = params["cfl"] * dx
        self.order = params["Order"]
        
        self.scheme = params.get("scheme", "floating_point").lower()
        self._init_derivative_operator()
        self.ng = self.diff_op.ng 

        self._init_grid(params)
        self._init_sponge(params)

        self.K1 = params.get("K1", 100.0)
        self.K2 = params.get("K2", 100.0)
        self.ko_sigma = params.get("ko_sigma", 0.05)

    def _init_derivative_operator(self):
        if self.scheme == "ozaki":
            from ozaki import OzakiDerivative
            self.diff_op = OzakiDerivative(block_size=64, halo=4)
        else:
            from floating_point import SpatialDerivative as FloatDerivative
            self.diff_op = FloatDerivative(order=self.order)

    def _init_grid(self, p: Dict[str, Any]):
        self.Nx, self.Ny = p["Nx"], p["Ny"]
        self.Nx_tot, self.Ny_tot = self.Nx + 2*self.ng, self.Ny + 2*self.ng

        self.x = p["xmin"] + jnp.arange(-self.ng, self.Nx + self.ng) * self.dx
        self.y = p["ymin"] + jnp.arange(-self.ng, self.Ny + self.ng) * self.dy
        
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        self.R = jnp.sqrt(self.X**2 + self.Y**2) + 1e-15

    def _init_sponge(self, p: Dict[str, Any]):
        width = 0.1 * (p["xmax"] - p["xmin"])
        dist_x = jnp.maximum(0.0, jnp.abs(self.x) - (p["xmax"] - width))
        dist_y = jnp.maximum(0.0, jnp.abs(self.y) - (p["ymax"] - width))
        mask = (dist_x[:, None] / width)**2 + (dist_y[None, :] / width)**2
        self.sponge_mask = jnp.clip(mask, 0.0, 1.0)
        self.sponge_strength = p.get("sponge_strength", 10.0)

    def d_dx(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d1(u, self.dx, axis=0)
    def d_dy(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d1(u, self.dy, axis=1)
    def d2_dx2(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d2(u, self.dx, axis=0)
    def d2_dy2(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d2(u, self.dy, axis=1)

    def apply_ko(self, u: jnp.ndarray) -> jnp.ndarray:
        ko = (self.diff_op.compute_ko(u, self.dx, self.ko_sigma, axis=0) + 
              self.diff_op.compute_ko(u, self.dy, self.ko_sigma, axis=1))
        return self.bc_zero(ko)

    def bc_zero(self, field: jnp.ndarray) -> jnp.ndarray:
        field = field.at[:self.ng, :].set(0.0)
        field = field.at[-self.ng:, :].set(0.0)
        field = field.at[:, :self.ng].set(0.0)
        field = field.at[:, -self.ng:].set(0.0)
        return field

    def bc_sommerfeld(self, u: jnp.ndarray, dtu: jnp.ndarray) -> jnp.ndarray:
        du_dx_f = (jnp.roll(u, -1, axis=0) - u) / self.dx
        du_dx_b = (u - jnp.roll(u, 1, axis=0)) / self.dx
        du_dy_f = (jnp.roll(u, -1, axis=1) - u) / self.dy
        du_dy_b = (u - jnp.roll(u, 1, axis=1)) / self.dy
        du_dx_c, du_dy_c = self.d_dx(u), self.d_dy(u)

        def calc_bc(slc, dx_op, dy_op):
            return (-self.X[slc] * dx_op[slc] - self.Y[slc] * dy_op[slc]) / self.R[slc]

        dtu = dtu.at[:self.ng, :].set(calc_bc(jnp.s_[:self.ng, :], du_dx_f, du_dy_c))
        dtu = dtu.at[-self.ng:, :].set(calc_bc(jnp.s_[-self.ng:, :], du_dx_b, du_dy_c))
        dtu = dtu.at[:, -self.ng:].set(calc_bc(jnp.s_[:, -self.ng:], du_dx_c, du_dy_b))
        dtu = dtu.at[:, :self.ng].set(calc_bc(jnp.s_[:, :self.ng], du_dx_c, du_dy_f))
        return dtu
    
    def bc_periodic(self, field: jnp.ndarray) -> jnp.ndarray:
        """Applies exact periodic boundary conditions to the ghost zones."""
        # Top and Bottom boundaries
        field = field.at[:self.ng, :].set(field[-2*self.ng:-self.ng, :])
        field = field.at[-self.ng:, :].set(field[self.ng:2*self.ng, :])
        
        # Left and Right boundaries
        field = field.at[:, :self.ng].set(field[:, -2*self.ng:-self.ng])
        field = field.at[:, -self.ng:].set(field[:, self.ng:2*self.ng])
        
        return field
    
    def bc_directional(self, u: jnp.ndarray, dtu: jnp.ndarray) -> jnp.ndarray:
        """Advective boundary condition that perfectly swallows a specific plane wave."""
        v_x = self.params.get("v_x", 1.0)
        v_y = self.params.get("v_y", 1.0)

        # 1st-order upwind stencils (identical to your Sommerfeld setup)
        du_dx_f = (jnp.roll(u, -1, axis=0) - u) / self.dx
        du_dx_b = (u - jnp.roll(u, 1, axis=0)) / self.dx
        du_dy_f = (jnp.roll(u, -1, axis=1) - u) / self.dy
        du_dy_b = (u - jnp.roll(u, 1, axis=1)) / self.dy
        du_dx_c, du_dy_c = self.d_dx(u), self.d_dy(u)

        # The new directional advection math
        def calc_bc(slc, dx_op, dy_op):
            return -v_x * dx_op[slc] - v_y * dy_op[slc]

        # Apply to the ghost zones
        dtu = dtu.at[:self.ng, :].set(calc_bc(jnp.s_[:self.ng, :], du_dx_f, du_dy_c))
        dtu = dtu.at[-self.ng:, :].set(calc_bc(jnp.s_[-self.ng:, :], du_dx_b, du_dy_c))
        dtu = dtu.at[:, -self.ng:].set(calc_bc(jnp.s_[:, -self.ng:], du_dx_c, du_dy_b))
        dtu = dtu.at[:, :self.ng].set(calc_bc(jnp.s_[:, :self.ng], du_dx_c, du_dy_f))
        
        return dtu

    def rhs(self, state: WaveState) -> WaveState:
        cs = self.params.get("enable_cs", 1.0)
        L = self.Lambda
        bc_type = self.params.get("bc_type", "sommerfeld")

        synced_data = state.data
        if bc_type == "periodic":
            synced_data = jax.vmap(self.bc_periodic)(synced_data)
        
        s = WaveState(synced_data)

        dt_Ex = self.d_dy(s.Bz) - self.d_dx(s.Psi) - cs*2*L*(s.Pi*s.Bx - s.Ez*self.d_dy(s.xi))
        dt_Ey = -self.d_dx(s.Bz) - self.d_dy(s.Psi) - cs*2*L*(s.Pi*s.By + s.Ez*self.d_dx(s.xi))
        dt_Ez = self.d_dx(s.By) - self.d_dy(s.Bx) - cs*2*L*(s.Pi*s.Bz + s.Ex*self.d_dy(s.xi) - s.Ey*self.d_dx(s.xi))
        
        dt_Bx, dt_By = -self.d_dy(s.Ez) + self.d_dx(s.Phi), self.d_dx(s.Ez) + self.d_dy(s.Phi)
        dt_Bz = -self.d_dx(s.Ey) + self.d_dy(s.Ex)
        
        dt_xi = -s.Pi * cs
        dt_Pi = (-self.d2_dx2(s.xi) - self.d2_dy2(s.xi) + 2*L*(s.Bx*s.Ex + s.By*s.Ey + s.Bz*s.Ez)) * cs
        
        dt_Psi = -self.d_dx(s.Ex) - self.d_dy(s.Ey) - self.K1*s.Psi - cs*2*L*(s.Bx*self.d_dx(s.xi) + s.By*self.d_dy(s.xi))
        dt_Phi = self.d_dx(s.Bx) + self.d_dy(s.By) - self.K2 * s.Phi

        dt_data = jnp.stack([dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz, dt_xi, dt_Pi, dt_Psi, dt_Phi])

        for i in [self.EX, self.EY, self.EZ, self.BX, self.BY, self.BZ, self.PSI, self.PHI, self.XI, self.PI]:
            if bc_type == "periodic":
                dt_data = dt_data.at[i].set(self.bc_periodic(dt_data[i]))
            elif bc_type == "directional":
                if i in [self.XI, self.PI]:
                    dt_data = dt_data.at[i].set(self.bc_zero(dt_data[i]))
                else:
                    # Swallow the wave in a straight line!
                    dt_data = dt_data.at[i].set(self.bc_directional(s.data[i], dt_data[i]))
            else:
                if i in [self.XI, self.PI]:
                    dt_data = dt_data.at[i].set(self.bc_zero(dt_data[i]))
                else:
                    dt_data = dt_data.at[i].set(self.bc_sommerfeld(s.data[i], dt_data[i]))

        if self.ko_sigma > 0:
            dt_data += jax.vmap(self.apply_ko)(s.data)
            
        dt_data -= (self.sponge_strength * self.sponge_mask) * state.data
        return WaveState(dt_data)

    def step_rk4(self, state: WaveState, dt: float) -> WaveState:
        k1 = self.rhs(state)
        k2 = self.rhs(jax.tree_util.tree_map(lambda s, k: s + 0.5 * dt * k, state, k1))
        k3 = self.rhs(jax.tree_util.tree_map(lambda s, k: s + 0.5 * dt * k, state, k2))
        k4 = self.rhs(jax.tree_util.tree_map(lambda s, k: s + dt * k, state, k3))
        
        return jax.tree_util.tree_map(
            lambda s, v1, v2, v3, v4: s + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4),
            state, k1, k2, k3, k4
        )

def calc_constraints(sim: 'MaxwellChernSimons2D', state: 'WaveState') -> Tuple[jnp.ndarray, jnp.ndarray]:
    cs = sim.params.get("enable_cs", 1.0)
    L = sim.Lambda
    divE_error = sim.d_dx(state.Ex) + sim.d_dy(state.Ey) + cs * 2 * L * (state.Bx * sim.d_dx(state.xi) + state.By * sim.d_dy(state.xi))
    divB_error = sim.d_dx(state.Bx) + sim.d_dy(state.By)
    return divE_error, divB_error

def l2norm(u: jnp.ndarray) -> float:
    return jnp.sqrt(jnp.mean(u**2))

def get_physical(arr: jnp.ndarray, ng: int) -> jnp.ndarray:
    """Removes Ghost Zones"""
    if arr.ndim == 1:
        return arr[ng:-ng]
    elif arr.ndim == 2:
        return arr[ng:-ng, ng:-ng]
    elif arr.ndim == 3:
        return arr[:, ng:-ng, ng:-ng]
    return arr

def save_output(step: int, sim: 'MaxwellChernSimons2D', state: 'WaveState', output_dir: str):
    if not iox: return
    names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Psi", "Phi"]
    
    phys_data = get_physical(state.data, sim.ng)
    
    x_coords = np.array(sim.x[sim.ng:-sim.ng])
    y_coords = np.array(sim.y[sim.ng:-sim.ng])
    
    iox.write_hdf5(step, np.asarray(phys_data), x_coords, y_coords, unames=names, output_dir=output_dir)

def load_parameters(parfile: str) -> Dict[str, Any]:
    if os.path.exists(parfile):
        with open(parfile, "rb") as f:
            return tomllib.load(f)
    return {
        "Nx": 256, "Ny": 256, "Nt": 1000, "output_interval": 10,
        "cfl": 0.05, "ko_sigma": 0.05, "Lambda": 0.1, 
        "enable_cs": 1.0, "sponge_strength": 10.0,
        "scheme": "floating_point", 
        "id_amp": 0.8, "id_sigma": 0.5, "id_y0": 0.0, "id_x0": 0.0,
        "xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0,
        "K1": 100.0, "K2": 100.0  
    }

def main(parfile: str, output_dir: str):
    params = load_parameters(parfile)
    nx, ny = params["Nx"], params["Ny"]    
    nt, out_int = params["Nt"], params["output_interval"]
    
    dx = (params["xmax"] - params["xmin"]) / (nx - 1)
    dy = (params["ymax"] - params["ymin"]) / (ny - 1)

    sim = MaxwellChernSimons2D(dx, dy, params.get("Lambda", 0.1), params)
    state = InitialData(sim, params).generate()

    save_output(0, sim, state, output_dir)
    
    print(f"\n>> Starting 2D Sim | Math: {sim.scheme.upper()} | Precision: 64-bit")
    print(f">> Grid: {nx}x{ny} | Steps: {nt} | CFL: {params.get('cfl')}\n")
    
    @jax.jit
    def time_step(i, current_state):
        return sim.step_rk4(current_state, sim.dt)

    start_time = time.time()
    for s in range(out_int, nt + 1, out_int):
        state = jax.lax.fori_loop(0, out_int, time_step, state)
        state.data.block_until_ready()
        
        divE, divB = calc_constraints(sim, state)
        print(f"Step {s:04d}/{nt} | Wall: {time.time() - start_time:.1f}s | divB: {l2norm(get_physical(divB, sim.ng)):.2e}")
        
        save_output(s, sim, state, output_dir)
            
    if iox:
        names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Psi", "Phi"]
        iox.write_xdmf(output_dir, nt, nx, ny, unames=names, output_interval=out_int, dt=sim.dt)
        print(f"\n>> Simulation complete. XDMF metadata written to {output_dir}")

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else "Utils/params.toml"
    out = sys.argv[2] if len(sys.argv) > 2 else "Maxwell-Chern-Simons_2D/mcs_data"
    
    if os.path.exists(out):
        print(f">> Wiping previous run data in: {out}")
        shutil.rmtree(out)
        
    os.makedirs(out, exist_ok=True)
    main(par, out)