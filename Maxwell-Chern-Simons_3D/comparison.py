import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

jax.config.update("jax_enable_x64", True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import your existing 3D simulation classes
from main import (
    MaxwellChernSimons3D, 
    InitialData, 
    calc_constraints, 
    get_physical, 
    load_parameters, 
    l2norm
)

class MCSBirefringentWave3D:
    """
    The 3D 'Oracle': Generates the exact analytical state of the 3D Birefringent Plane Wave.
    """
    def __init__(self, Lx, Ly, Lz, E0=1.0, m_cs=1.0):
        self.E0 = E0
        self.m_cs = m_cs
        
        self.k_x = 2.0 * jnp.pi / Lx
        self.k_y = 2.0 * jnp.pi / Ly
        self.k_z = 2.0 * jnp.pi / Lz
        self.k = jnp.sqrt(self.k_x**2 + self.k_y**2 + self.k_z**2)
        
        self.omega = jnp.sqrt(self.k**2 + self.m_cs * self.k)
        
        norm_factor = jnp.sqrt(self.k_x**2 + self.k_y**2)
        self.e1_z = 0.0
        self.e2_z = norm_factor / self.k

    def get_exact_Ez(self, X, Y, Z, t):
        """Returns the exact 3D volume of the Ez component."""
        Phi = self.k_x * X + self.k_y * Y + self.k_z * Z - self.omega * t
        # FIXED: Inverted sign perfectly mirrors the physical E-field components
        return self.E0 * (self.e1_z * jnp.cos(Phi) - self.e2_z * jnp.sin(Phi))

def main():
    # 1. Setup Parameters
    parfile = sys.argv[1] if len(sys.argv) > 1 else "Utils/params.toml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "Maxwell-Chern-Simons_3D"
    os.makedirs(out_dir, exist_ok=True)
    
    params = load_parameters(parfile)
    
    # ==========================================================================
    # TEST HARNESS OVERRIDES 
    # ==========================================================================
    params["id_type"] = "birefringent"  # Force 3D birefringent wave
    params["bc_type"] = "periodic"      # Trigger the 3D periodic boundary
    params["sponge_strength"] = 0.0     # Disable the sponge
    params["ko_sigma"] = 0.05           # Whisper of friction
    # ==========================================================================
    
    nx, ny, nz = params["Nx"], params["Ny"], params["Nz"]    
    nt, out_int = params["Nt"], params["output_interval"]
    
    Lx = params["xmax"] - params["xmin"]
    Ly = params["ymax"] - params["ymin"]
    Lz = params["zmax"] - params["zmin"]
    
    # Exact non-redundant grid steps
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # 2. Initialize Sim, State, and Oracle
    sim = MaxwellChernSimons3D(dx, dy, dz, params.get("Lambda", 0.1), params)
    state = InitialData(sim, params).generate()
    oracle = MCSBirefringentWave3D(Lx, Ly, Lz, E0=params.get("id_amp", 1.0), m_cs=params.get("Lambda", 0.1)*2.0)

    # 3D Physical coordinates (stripping ghost zones)
    x_phys = sim.x[sim.ng:-sim.ng]
    y_phys = sim.y[sim.ng:-sim.ng]
    z_phys = sim.z[sim.ng:-sim.ng]
    X_phys, Y_phys, Z_phys = jnp.meshgrid(x_phys, y_phys, z_phys, indexing='ij')

    print(f"\n>> Starting 3D Video Comparison Test | Math: {sim.scheme.upper()} | Grid: {nx}x{ny}x{nz}")
    
    # ==========================================================================
    # ANIMATION SETTINGS (Using a 2D slice through the 3D volume)
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calculate baseline 3D error
    init_num = np.array(get_physical(state.data[sim.EZ], sim.ng))
    init_exact = np.array(oracle.get_exact_Ez(X_phys, Y_phys, Z_phys, 0.0))
    
    # We will visualize a flat XY plane exactly in the middle of the Z-axis
    mid_z = nz // 2
    slice_num = init_num[:, :, mid_z]
    slice_exact = init_exact[:, :, mid_z]
    slice_err = np.abs(slice_num - slice_exact)
    
    max_val = max(np.max(np.abs(slice_exact)), 1e-10)
    
    im0 = axes[0].pcolormesh(x_phys, y_phys, slice_num.T, cmap='RdBu_r', vmin=-max_val, vmax=max_val, shading='auto')
    axes[0].set_title(f"Numerical $E_z$ (Slice z={mid_z})")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x_phys, y_phys, slice_exact.T, cmap='RdBu_r', vmin=-max_val, vmax=max_val, shading='auto')
    axes[1].set_title(f"Exact Analytical $E_z$ (Slice z={mid_z})")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(x_phys, y_phys, slice_err.T, cmap='magma', vmin=0, vmax=0.1, shading='auto')
    axes[2].set_title(r"Absolute Error (Slice)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    video_filename = os.path.join(out_dir, "birefringent_wave_3d_comparison.gif")
    writer = PillowWriter(fps=15)
    
    # ==========================================================================
    # EVOLUTION LOOP
    # ==========================================================================
    @jax.jit
    def time_step(i, current_state):
        return sim.step_rk4(current_state, sim.dt)

    start_time = time.time()
    
    with writer.saving(fig, video_filename, dpi=150):
        
        for s in range(0, nt + 1, out_int):
            if s > 0:
                state = jax.lax.fori_loop(0, out_int, time_step, state)
                state.data.block_until_ready()
            
            current_t = s * sim.dt
            
            # 1. Extract FULL 3D data arrays
            num_Ez_np = np.array(get_physical(state.data[sim.EZ], sim.ng))
            exact_Ez_np = np.array(oracle.get_exact_Ez(X_phys, Y_phys, Z_phys, current_t))
            error_Ez_np = np.abs(num_Ez_np - exact_Ez_np)
            
            # 2. Compute console telemetry metrics across the ENTIRE 3D VOLUME
            l2_err = l2norm(error_Ez_np)
            linf_err = np.max(error_Ez_np)
            divE, divB = calc_constraints(sim, state)
            divB_norm = l2norm(get_physical(divB, sim.ng))
            
            print(f"Step {s:04d} | t={current_t:.3f} | L2 Err: {l2_err:.2e} | Linf Err: {linf_err:.2e} | divB: {divB_norm:.2e}")
            
            # 3. Slice the 3D volume down to a 2D sheet for visualization
            slice_num_update = num_Ez_np[:, :, mid_z]
            slice_exact_update = exact_Ez_np[:, :, mid_z]
            slice_err_update = error_Ez_np[:, :, mid_z]

            im0.set_array(slice_num_update.T.ravel())
            im1.set_array(slice_exact_update.T.ravel())
            
            im2.set_array(slice_err_update.T.ravel())
            im2.set_clim(0, max(np.max(slice_err_update), 1e-15))
            
            fig.suptitle(f"3D Birefringent Wave | Step: {s:04d} | t = {current_t:.4f}", fontsize=14)
            writer.grab_frame()

    plt.close()
    print(f"\n>> Evolution complete. Video saved directly to: {video_filename}")

if __name__ == "__main__":
    main()