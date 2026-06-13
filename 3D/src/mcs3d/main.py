# 1. Standard library imports
import os
import sys
import time
import shutil
import tomllib
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

import mcs_common.ioxdmf as iox
from mcs_common.wave_state import WaveState

class InitialData:
    """Generates divergence-free initial conditions for the 3D wave simulation."""
    EX, EY, EZ, BX, BY, BZ, XI, PI, PSI, PHI = range(10)

    def __init__(self, sim: Any, params: Dict[str, Any]):
        self.sim = sim
        self.params = params

    def generate(self):
        id_type = self.params.get("id_type", "gaussian")
        generators = {
            "gaussian": self._gaussian_pulse,
            "birefringent": self._birefringent_wave_3d
        }

        if id_type not in generators:
            raise ValueError(f"Initial data type '{id_type}' not recognized.")
        
        return generators[id_type]()

    def _gaussian_pulse(self):
        x0, y0, z0 = self.params.get("id_x0", 0.0), self.params.get("id_y0", 0.0), self.params.get("id_z0", 0.0)
        B0, sigma = self.params.get("id_amp", 0.8), self.params.get("id_sigma", 0.5)
        
        psi = B0 * jnp.exp(-(
            (self.sim.x[:, None, None] - x0)**2 + 
            (self.sim.y[None, :, None] - y0)**2 + 
            (self.sim.z[None, None, :] - z0)**2
        ) / (2 * sigma**2))
        
        data = jnp.zeros((10, self.sim.Nx_tot, self.sim.Ny_tot, self.sim.Nz_tot), dtype=jnp.float64)
        
        dy_psi = self.sim.d_dy(psi)
        dx_psi = self.sim.d_dx(psi)
        dz_psi = self.sim.d_dz(psi)
        
        data = data.at[self.EX].set(dy_psi - dz_psi)
        data = data.at[self.EY].set(dz_psi - dx_psi)
        data = data.at[self.EZ].set(dx_psi - dy_psi)
        
        data = data.at[self.BX].set(data[self.EX])
        data = data.at[self.BY].set(data[self.EY])
        data = data.at[self.BZ].set(data[self.EZ])
        
        data = data.at[self.XI].set(0.0)
        data = data.at[self.PI].set(0.0)
        data = data.at[self.PSI].set(0.0)
        data = data.at[self.PHI].set(0.0)
        
        return WaveState(data)

    def _birefringent_wave_3d(self):
        """Analytical 3D Left-Circularly Polarized wave (Flawless Polarization Triad)."""
        E0 = self.params.get("id_amp", 1.0)
        cs = self.params.get("enable_cs", 1.0)
        L_param = self.params.get("Lambda", 1.0)
        m_cs = self.params.get("id_m_cs", L_param * 2.0) 
        
        Lx = self.params.get("xmax", 5.0) - self.params.get("xmin", -5.0)
        Ly = self.params.get("ymax", 5.0) - self.params.get("ymin", -5.0)
        Lz = self.params.get("zmax", 5.0) - self.params.get("zmin", -5.0)
        
        k_x = 2.0 * jnp.pi / Lx
        k_y = 2.0 * jnp.pi / Ly
        k_z = 2.0 * jnp.pi / Lz
        k = jnp.sqrt(k_x**2 + k_y**2 + k_z**2)
        
        # Unconditionally stable dispersion relation
        omega = jnp.sqrt(k**2 + m_cs * k)
        
        # Generate Orthogonal Polarization Basis (e1, e2 perpendicular to k)
        norm_factor = jnp.sqrt(k_x**2 + k_y**2)
        e1_x = k_y / norm_factor
        e1_y = -k_x / norm_factor
        e1_z = 0.0
        
        e2_x = -k_x * k_z / (k * norm_factor)
        e2_y = -k_y * k_z / (k * norm_factor)
        e2_z = norm_factor / k
        
        Phi = k_x * self.sim.X + k_y * self.sim.Y + k_z * self.sim.Z
        cos_Phi = jnp.cos(Phi)
        sin_Phi = jnp.sin(Phi)
        
        # FIXED: E = E0 * (e1 * cos_Phi - e2 * sin_Phi)
        Ex = E0 * (e1_x * cos_Phi - e2_x * sin_Phi)
        Ey = E0 * (e1_y * cos_Phi - e2_y * sin_Phi)
        Ez = E0 * (e1_z * cos_Phi - e2_z * sin_Phi)
        
        # FIXED: B = (k/w) * E0 * (-e1 * sin_Phi - e2 * cos_Phi)
        b_scale = k / omega
        Bx = E0 * b_scale * (-e1_x * sin_Phi - e2_x * cos_Phi)
        By = E0 * b_scale * (-e1_y * sin_Phi - e2_y * cos_Phi)
        Bz = E0 * b_scale * (-e1_z * sin_Phi - e2_z * cos_Phi)
        
        Pi_0 = m_cs / (2.0 * cs * L_param)
        
        data = jnp.zeros((10, self.sim.Nx_tot, self.sim.Ny_tot, self.sim.Nz_tot), dtype=jnp.float64)
        
        data = data.at[self.EX].set(Ex)
        data = data.at[self.EY].set(Ey)
        data = data.at[self.EZ].set(Ez)
        data = data.at[self.BX].set(Bx)
        data = data.at[self.BY].set(By)
        data = data.at[self.BZ].set(Bz)
        
        data = data.at[self.XI].set(0.0)
        data = data.at[self.PI].set(Pi_0)
        data = data.at[self.PSI].set(0.0)
        data = data.at[self.PHI].set(0.0)
        
        return WaveState(data)

class MaxwellChernSimons3D:
    """Solves the 3D Maxwell-Chern-Simons equations."""

    EX, EY, EZ = 0, 1, 2
    BX, BY, BZ = 3, 4, 5
    XI, PI = 6, 7
    PSI, PHI = 8, 9

    def __init__(self, dx: float, dy: float, dz: float, Lambda: float, params: Dict[str, Any]):
        self.dx, self.dy, self.dz = dx, dy, dz
        self.Lambda = Lambda
        self.params = params
        self.dt = params["cfl"] * dx
        self.order = params["Order"]
        
        self.scheme = params.get("scheme", "floating_point").lower()
        self._init_derivative_operator()
        self.ng = self.diff_op.ng 

        self._init_grid(params)

        self.K1 = params.get("K1", 1.0)
        self.K2 = params.get("K2", 1.0)
        self.ko_sigma = params.get("ko_sigma", 0.05)

    def _init_derivative_operator(self):
        from mcs3d.schemes.floating_point import SpatialDerivative as FloatDerivative
        self.diff_op = FloatDerivative(order=self.order)

    def _init_grid(self, p: Dict[str, Any]):
        self.Nx, self.Ny, self.Nz = p["Nx"], p["Ny"], p["Nz"]
        self.Nx_tot, self.Ny_tot, self.Nz_tot = self.Nx + 2*self.ng, self.Ny + 2*self.ng, self.Nz + 2*self.ng

        # Using arange to prevent the duplicate endpoint trap
        self.x = p["xmin"] + jnp.arange(-self.ng, self.Nx + self.ng) * self.dx
        self.y = p["ymin"] + jnp.arange(-self.ng, self.Ny + self.ng) * self.dy
        self.z = p["zmin"] + jnp.arange(-self.ng, self.Nz + self.ng) * self.dz
        
        self.R = jnp.sqrt(self.x[:, None, None]**2 + 
                          self.y[None, :, None]**2 + 
                          self.z[None, None, :]**2) + 1e-15
                          
        # Only needed if you do full 3D spatial evaluations like the analytical wave:
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')

    def d_dx(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d1(u, self.dx, axis=0)
    def d_dy(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d1(u, self.dy, axis=1)
    def d_dz(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d1(u, self.dz, axis=2)

    def d2_dx2(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d2(u, self.dx, axis=0)
    def d2_dy2(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d2(u, self.dy, axis=1)
    def d2_dz2(self, u: jnp.ndarray) -> jnp.ndarray: return self.diff_op.compute_d2(u, self.dz, axis=2)

    def _d1_batched(self, u_batch: jnp.ndarray, dx: float, axis: int) -> jnp.ndarray:
        return jax.vmap(lambda u: self.diff_op.compute_d1(u, dx, axis))(u_batch)

    def _d2_batched(self, u_batch: jnp.ndarray, dx: float, axis: int) -> jnp.ndarray:
        return jax.vmap(lambda u: self.diff_op.compute_d2(u, dx, axis))(u_batch)

    def _ko_batched(self, u_batch: jnp.ndarray, dx: float, sigma: float, axis: int) -> jnp.ndarray:
        return jax.vmap(lambda u: self.diff_op.compute_ko(u, dx, sigma, axis))(u_batch)

    def apply_ko(self, u: jnp.ndarray) -> jnp.ndarray:
        ko = (self.diff_op.compute_ko(u, self.dx, self.ko_sigma, axis=0) + 
              self.diff_op.compute_ko(u, self.dy, self.ko_sigma, axis=1) +
              self.diff_op.compute_ko(u, self.dz, self.ko_sigma, axis=2))
        return self.bc_zero(ko)

    def bc_zero(self, field: jnp.ndarray) -> jnp.ndarray:
        field = field.at[:self.ng, :, :].set(0.0)
        field = field.at[-self.ng:, :, :].set(0.0)
        field = field.at[:, :self.ng, :].set(0.0)
        field = field.at[:, -self.ng:, :].set(0.0)
        field = field.at[:, :, :self.ng].set(0.0)
        field = field.at[:, :, -self.ng:].set(0.0)
        return field

    def bc_periodic(self, field: jnp.ndarray) -> jnp.ndarray:
        """Applies exact periodic boundary conditions to the 3D ghost zones."""
        # X boundaries
        field = field.at[:self.ng, :, :].set(field[-2*self.ng:-self.ng, :, :])
        field = field.at[-self.ng:, :, :].set(field[self.ng:2*self.ng, :, :])
        
        # Y boundaries
        field = field.at[:, :self.ng, :].set(field[:, -2*self.ng:-self.ng, :])
        field = field.at[:, -self.ng:, :].set(field[:, self.ng:2*self.ng, :])
        
        # Z boundaries
        field = field.at[:, :, :self.ng].set(field[:, :, -2*self.ng:-self.ng])
        field = field.at[:, :, -self.ng:].set(field[:, :, self.ng:2*self.ng])
        return field

    def bc_sommerfeld(self, u: jnp.ndarray, dtu: jnp.ndarray) -> jnp.ndarray:
        du_dx_f = (jnp.roll(u, -1, axis=0) - u) / self.dx
        du_dx_b = (u - jnp.roll(u, 1, axis=0)) / self.dx
        du_dy_f = (jnp.roll(u, -1, axis=1) - u) / self.dy
        du_dy_b = (u - jnp.roll(u, 1, axis=1)) / self.dy
        du_dz_f = (jnp.roll(u, -1, axis=2) - u) / self.dz
        du_dz_b = (u - jnp.roll(u, 1, axis=2)) / self.dz
        
        du_dx_c, du_dy_c, du_dz_c = self.d_dx(u), self.d_dy(u), self.d_dz(u)

        def calc_bc(slc, dx_op, dy_op, dz_op):
            return (-self.X[slc] * dx_op[slc] - self.Y[slc] * dy_op[slc] - self.Z[slc] * dz_op[slc]) / self.R[slc]

        dtu = dtu.at[:self.ng, :, :].set(calc_bc(jnp.s_[:self.ng, :, :], du_dx_f, du_dy_c, du_dz_c))
        dtu = dtu.at[-self.ng:, :, :].set(calc_bc(jnp.s_[-self.ng:, :, :], du_dx_b, du_dy_c, du_dz_c))
        dtu = dtu.at[:, :self.ng, :].set(calc_bc(jnp.s_[:, :self.ng, :], du_dx_c, du_dy_f, du_dz_c))
        dtu = dtu.at[:, -self.ng:, :].set(calc_bc(jnp.s_[:, -self.ng:, :], du_dx_c, du_dy_b, du_dz_c))
        dtu = dtu.at[:, :, :self.ng].set(calc_bc(jnp.s_[:, :, :self.ng], du_dx_c, du_dy_c, du_dz_f))
        dtu = dtu.at[:, :, -self.ng:].set(calc_bc(jnp.s_[:, :, -self.ng:], du_dx_c, du_dy_c, du_dz_b))
        
        return dtu

    def rhs(self, state: WaveState) -> WaveState:
        cs = self.params.get("enable_cs", 1.0)
        L = self.Lambda
        bc_type = self.params.get("bc_type", "sommerfeld")

        synced_data = state.data
        if bc_type == "periodic":
            synced_data = jax.vmap(self.bc_periodic)(synced_data)

        s = WaveState(synced_data)

        # Batch all 9 differentiated fields (PI only appears as a multiplier).
        # Three vmap calls replace ~27 individual compute_d1 calls.
        _d1_idx = jnp.array([self.EX, self.EY, self.EZ, self.BX, self.BY, self.BZ,
                              self.XI, self.PSI, self.PHI])
        d1_in = s.data[_d1_idx]
        dx_all = self._d1_batched(d1_in, self.dx, axis=0)
        dy_all = self._d1_batched(d1_in, self.dy, axis=1)
        dz_all = self._d1_batched(d1_in, self.dz, axis=2)

        dEx_dx, dEy_dx, dEz_dx, dBx_dx, dBy_dx, dBz_dx, dxi_dx, dPsi_dx, dPhi_dx = dx_all
        dEx_dy, dEy_dy, dEz_dy, dBx_dy, dBy_dy, dBz_dy, dxi_dy, dPsi_dy, dPhi_dy = dy_all
        dEx_dz, dEy_dz, dEz_dz, dBx_dz, dBy_dz, dBz_dz, dxi_dz, dPsi_dz, dPhi_dz = dz_all

        d2xi_dx2 = self.diff_op.compute_d2(s.xi, self.dx, axis=0)
        d2xi_dy2 = self.diff_op.compute_d2(s.xi, self.dy, axis=1)
        d2xi_dz2 = self.diff_op.compute_d2(s.xi, self.dz, axis=2)

        dt_Ex  = (dBz_dy  - dBy_dz) - dPsi_dx - cs*2*L*(s.Pi*s.Bx - s.Ez*dxi_dy  + s.Ey*dxi_dz)
        dt_Ey  = (dBx_dz  - dBz_dx) - dPsi_dy - cs*2*L*(s.Pi*s.By - s.Ex*dxi_dz  + s.Ez*dxi_dx)
        dt_Ez  = (dBy_dx  - dBx_dy) - dPsi_dz - cs*2*L*(s.Pi*s.Bz - s.Ey*dxi_dx  + s.Ex*dxi_dy)
        dt_Bx  = -dEz_dy + dEy_dz + dPhi_dx
        dt_By  = -dEx_dz + dEz_dx + dPhi_dy
        dt_Bz  = -dEy_dx + dEx_dy + dPhi_dz
        dt_xi  = -s.Pi * cs
        dt_Pi  = (-d2xi_dx2 - d2xi_dy2 - d2xi_dz2 + 2*L*(s.Bx*s.Ex + s.By*s.Ey + s.Bz*s.Ez)) * cs
        dt_Psi = -dEx_dx - dEy_dy - dEz_dz - self.K1*s.Psi - cs*2*L*(s.Bx*dxi_dx + s.By*dxi_dy + s.Bz*dxi_dz)
        dt_Phi =  dBx_dx + dBy_dy + dBz_dz - self.K2*s.Phi

        dt_data = jnp.stack([dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz,
                              dt_xi, dt_Pi, dt_Psi, dt_Phi])

        if self.ko_sigma > 0:
            ko = (self._ko_batched(s.data, self.dx, self.ko_sigma, axis=0) +
                  self._ko_batched(s.data, self.dy, self.ko_sigma, axis=1) +
                  self._ko_batched(s.data, self.dz, self.ko_sigma, axis=2))
            dt_data += jax.vmap(self.bc_zero)(ko)

        if bc_type == "periodic":
            dt_data = jax.vmap(self.bc_periodic)(dt_data)
        else:
            for i in [self.EX, self.EY, self.EZ, self.BX, self.BY, self.BZ,
                      self.PSI, self.PHI, self.XI, self.PI]:
                if i in [self.XI, self.PI]:
                    dt_data = dt_data.at[i].set(self.bc_zero(dt_data[i]))
                else:
                    dt_data = dt_data.at[i].set(self.bc_sommerfeld(s.data[i], dt_data[i]))

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

def calc_constraints(sim: 'MaxwellChernSimons3D', state: 'WaveState') -> Tuple[jnp.ndarray, jnp.ndarray]:
    cs = sim.params.get("enable_cs", 1.0)
    L = sim.Lambda
    
    divE_error = (sim.d_dx(state.Ex) + sim.d_dy(state.Ey) + sim.d_dz(state.Ez) + 
                  cs * 2 * L * (state.Bx * sim.d_dx(state.xi) + state.By * sim.d_dy(state.xi) + state.Bz * sim.d_dz(state.xi)))
    
    divB_error = sim.d_dx(state.Bx) + sim.d_dy(state.By) + sim.d_dz(state.Bz)
    
    return divE_error, divB_error

def l2norm(u: jnp.ndarray) -> float:
    return jnp.sqrt(jnp.mean(u**2))

def get_physical(arr: jnp.ndarray, ng: int) -> jnp.ndarray:
    """Removes ghost zones"""
    if arr.ndim == 1:
        return arr[ng:-ng]
    elif arr.ndim == 3: # 3D single variable
        return arr[ng:-ng, ng:-ng, ng:-ng]
    elif arr.ndim == 4: # 3D 10-variable stack
        return arr[:, ng:-ng, ng:-ng, ng:-ng]
    return arr

def save_output(step: int, sim: 'MaxwellChernSimons3D', state: 'WaveState', output_dir: str):
    if not iox: return
    names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Psi", "Phi"]
    
    phys_data = get_physical(state.data, sim.ng)
    
    x_coords = np.array(sim.x[sim.ng:-sim.ng])
    y_coords = np.array(sim.y[sim.ng:-sim.ng])
    z_coords = np.array(sim.z[sim.ng:-sim.ng])
    
    iox.write_hdf5(step, np.asarray(phys_data), x_coords, y_coords, z_coords, unames=names, output_dir=output_dir)

def load_parameters(parfile: str) -> Dict[str, Any]:
    if os.path.exists(parfile):
        with open(parfile, "rb") as f:
            return tomllib.load(f)
            
    print(f">> WARNING: Parameter file '{parfile}' not found. Using defaults.")
    return {
        "Nx": 64, "Ny": 64, "Nz": 64, "Nt": 1000, "output_interval": 10,
        "cfl": 0.05, "ko_sigma": 0.05, "Lambda": 0.1, "Order": 6,
        "enable_cs": 1.0, "sponge_strength": 10.0,
        "scheme": "floating_point",
        "id_amp": 0.8, "id_sigma": 0.5, "id_y0": 0.0, "id_x0": 0.0, "id_z0": 0.0,
        "xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0,
        "K1": 100.0, "K2": 100.0
    }

def main(parfile: str, output_dir: str):
    params = load_parameters(parfile)
    
    nx, ny, nz = params["Nx"], params["Ny"], params["Nz"]
    nt, out_int = params["Nt"], params["output_interval"]
    
    # Grid calculation explicitly updated to remove the duplicated endpoints!
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny
    dz = (params["zmax"] - params["zmin"]) / nz

    sim = MaxwellChernSimons3D(dx, dy, dz, params.get("Lambda", 0.1), params)
    state = InitialData(sim, params).generate()

    save_output(0, sim, state, output_dir)
    
    status = "ON" if params.get("enable_cs", 1.0) == 1.0 else "OFF"
    print(f"\n>> Starting 3D Sim | Math: {sim.scheme.upper()} | CS: {status} | Precision: 64-bit")
    print(f">> Grid: {nx}x{ny}x{nz} | Steps: {nt} | CFL: {params.get('cfl')}\n")
    
    @jax.jit
    def time_step(i, current_state):
        return sim.step_rk4(current_state, sim.dt)

    start_time = time.time()
    for s in range(out_int, nt + 1, out_int):
        
        state = jax.lax.fori_loop(0, out_int, time_step, state)
        state.data.block_until_ready() 
        
        divE, divB = calc_constraints(sim, state)
        
        print(
            f"Step {s:04d}/{nt} | Wall: {time.time() - start_time:.2f}s | "
            f"divB: {l2norm(get_physical(divB, sim.ng)):.2e} | "
            f"divE: {l2norm(get_physical(divE, sim.ng)):.2e}"
        )
        
        save_output(s, sim, state, output_dir)
            
    if iox:
        names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Psi", "Phi"]
        iox.write_xdmf(output_dir, nt, nx, ny, nz, unames=names, output_interval=out_int, dt=sim.dt)
        print(f"\n>> Simulation complete. XDMF metadata written to {output_dir}")

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "params.toml")
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent / "output")
    
    if os.path.exists(out):
        print(f">> Removing previous run data in: {out}")
        shutil.rmtree(out)
        
    os.makedirs(out, exist_ok=True)
    main(par, out)