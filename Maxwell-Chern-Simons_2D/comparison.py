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

# Import your existing simulation classes
from main import (
    MaxwellChernSimons2D, 
    InitialData, 
    calc_constraints, 
    get_physical, 
    load_parameters, 
    l2norm
)

class MCSBirefringentWave2D:
    """
    The 'Oracle': Generates the exact analytical state of the 2.5D Birefringent Plane Wave 
    at any time 't' for error comparison.
    """
    def __init__(self, L=1.0, E0=1.0, m_cs=1.0):
        self.L = L
        self.E0 = E0
        self.m_cs = m_cs
        
        # Diagonal wave vector
        self.k_x = 2.0 * jnp.pi / L
        self.k_y = 2.0 * jnp.pi / L
        self.k = jnp.sqrt(self.k_x**2 + self.k_y**2)
        
        # The true phase velocity of your specific PDE engine
        self.omega_minus = jnp.sqrt(self.k**2 + self.m_cs * self.k)
        
        # Polarization Basis Vectors
        self.e1_x = -self.k_y / self.k
        self.e1_y = self.k_x / self.k
        self.e2_z = 1.0

    def get_exact_Ez(self, X, Y, t):
        """Returns only the exact Ez component for fast heatmap comparison."""
        # Because omega_minus is > 0, as 't' increases, Phi changes, and the wave moves.
        Phi = self.k_x * X + self.k_y * Y - self.omega_minus * t
        return self.E0 * self.e2_z * jnp.sin(Phi)

def main():
    # 1. Setup Parameters
    parfile = sys.argv[1] if len(sys.argv) > 1 else "Utils/params.toml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "Maxwell-Chern-Simons_2D"
    os.makedirs(out_dir, exist_ok=True)
    
    params = load_parameters(parfile)
    
    # ==========================================================================
    # TEST HARNESS OVERRIDES & DIRECTIONAL BOUNDARY SETUP
    # ==========================================================================
    params["id_type"] = "birefringent"  # Force birefringent wave for exact test
    params["sponge_strength"] = 0.0     # Disable the sponge to avoid reflection
    params["ko_sigma"] = 0.05           # Whisper of friction to kill aliasing

    L = params["xmax"] - params["xmin"]
    k_x = 2.0 * jnp.pi / L
    k_y = 2.0 * jnp.pi / L
    k_sq = k_x**2 + k_y**2
    k = jnp.sqrt(k_sq)
    
    # Calculate exact advection velocities for the boundary condition
    m_cs = params.get("Lambda", 0.1) * 2.0
    omega_minus = jnp.sqrt(k_sq + m_cs * k)
    
    # v_phase = (omega / k) * (k_x / k) = omega * k_x / k^2
    params["v_x"] = float((omega_minus * k_x) / k_sq)
    params["v_y"] = float((omega_minus * k_y) / k_sq)
    # ==========================================================================
    
    nx, ny = params["Nx"], params["Ny"]    
    nt, out_int = params["Nt"], params["output_interval"]
    dx = L / (nx)
    dy = (params["ymax"] - params["ymin"]) / (ny)

    # 2. Initialize Sim, State, and Oracle
    sim = MaxwellChernSimons2D(dx, dy, params.get("Lambda", 0.1), params)
    state = InitialData(sim, params).generate()
    oracle = MCSBirefringentWave2D(L=L, E0=params.get("id_amp", 1.0), m_cs=params.get("Lambda", 0.1)*2.0)

    x_phys = sim.x[sim.ng:-sim.ng]
    y_phys = sim.y[sim.ng:-sim.ng]
    X_phys, Y_phys = jnp.meshgrid(x_phys, y_phys, indexing='ij')

    print(f"\n>> Starting Video Comparison Test | Math: {sim.scheme.upper()} | Grid: {nx}x{ny}")
    
    # ==========================================================================
    # ANIMATION SETTINGS (Initialize the plot once before the loop)
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get baseline initialization data
    init_num = np.array(get_physical(state.data[sim.EZ], sim.ng))
    init_exact = np.array(oracle.get_exact_Ez(X_phys, Y_phys, 0.0))
    init_err = np.abs(init_num - init_exact)
    
    max_val = max(np.max(np.abs(init_exact)), 1e-10)
    
    # Create permanent plot elements that we will update dynamically
    im0 = axes[0].pcolormesh(x_phys, y_phys, init_num.T, cmap='RdBu_r', vmin=-max_val, vmax=max_val, shading='auto')
    axes[0].set_title("Numerical (Ozaki/FP64)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x_phys, y_phys, init_exact.T, cmap='RdBu_r', vmin=-max_val, vmax=max_val, shading='auto')
    axes[1].set_title("Exact Analytical")
    fig.colorbar(im1, ax=axes[1])

    # Dynamic error scale based on numerical threshold
    im2 = axes[2].pcolormesh(x_phys, y_phys, init_err.T, cmap='magma', vmin=0, vmax=0.1, shading='auto')
    axes[2].set_title(r"Absolute Error")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    # Set up the FFMpeg Video Writer (Targeting 15 frames per second)
    video_filename = os.path.join(out_dir, "birefringent_wave_comparison.gif")
    writer = PillowWriter(fps=15)
    
    # ==========================================================================
    # EVOLUTION LOOP
    # ==========================================================================
    @jax.jit
    def time_step(i, current_state):
        return sim.step_rk4(current_state, sim.dt)

    start_time = time.time()
    
    # Open the video file stream context
    with writer.saving(fig, video_filename, dpi=150):
        
        for s in range(0, nt + 1, out_int):
            if s > 0:
                state = jax.lax.fori_loop(0, out_int, time_step, state)
                state.data.block_until_ready()
            
            current_t = s * sim.dt
            
            # Extract data as standard numpy arrays for plotting
            num_Ez_np = np.array(get_physical(state.data[sim.EZ], sim.ng))
            exact_Ez_np = np.array(oracle.get_exact_Ez(X_phys, Y_phys, current_t))
            error_Ez_np = np.abs(num_Ez_np - exact_Ez_np)
            
            # Compute console telemetry metrics
            l2_err = l2norm(error_Ez_np)
            linf_err = np.max(error_Ez_np)
            divE, divB = calc_constraints(sim, state)
            divB_norm = l2norm(get_physical(divB, sim.ng))
            
            print(f"Step {s:04d} | t={current_t:.3f} | L2 Err: {l2_err:.2e} | Linf Err: {linf_err:.2e} | divB: {divB_norm:.2e}")
            
            # DYNAMICALLY UPDATE PLOTS (No disk saving overhead)
            # Must flatten arrays using .ravel() to update pcolormesh accurately in matplotlib
            im0.set_array(num_Ez_np.T.ravel())
            im1.set_array(exact_Ez_np.T.ravel())
            
            # Automatically scale the error map limit so you can see precision drift
            im2.set_array(error_Ez_np.T.ravel())
            im2.set_clim(0, max(np.max(error_Ez_np), 1e-15))
            
            fig.suptitle(f"Birefringent Wave Evolution | Step: {s:04d} | t = {current_t:.4f}", fontsize=14)
            
            # Grab the current frame state and append it directly into the MP4 file
            writer.grab_frame()

    plt.close()
    print(f"\n>> Evolution complete. Video saved directly to: {video_filename}")

if __name__ == "__main__":
    main()