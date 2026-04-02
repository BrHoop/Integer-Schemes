import os
#Prevents error messages from my GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tomllib
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class

try:
    import iox
except ImportError:
    iox = None
    print("Warning: iox module not found. Output will be disabled.")


# 1. Config
jax.config.update("jax_enable_x64", True)

#Written by Gemini Pro 3.1
def get_boundary_weights(patch_index, num_points=9, deriv_order=2):
    offsets = jnp.arange(num_points) - patch_index
    A = jnp.power(offsets[:, None], jnp.arange(num_points))
    b = jnp.zeros(num_points)
    b[deriv_order] = math.factorial(deriv_order)
    return jnp.linalg.solve(A, b)

W0_1 = get_boundary_weights(0, 9, 1)
W1_1 = get_boundary_weights(1, 9, 1)
W2_1 = get_boundary_weights(2, 9, 1)
W3_1 = get_boundary_weights(3, 9, 1)  

W0_2 = get_boundary_weights(0, 9, 2)
W1_2 = get_boundary_weights(1, 9, 2)
W2_2 = get_boundary_weights(2, 9, 2)
W3_2 = get_boundary_weights(3, 9, 2)

#WaveState class code Written by Gemini 3.1 Pro
@register_pytree_node_class
class WaveState:
    def __init__(self, arr):
        self.data = arr

    @property
    def Ex(self):
        return self.data[0]
    @property
    def Ey(self):
        return self.data[1]
    @property
    def Ez(self):
        return self.data[2]
    @property
    def Bx(self):
        return self.data[3]
    @property
    def By(self):
        return self.data[4]
    @property
    def Bz(self):
        return self.data[5]
    @property
    def Pi(self):
        return self.data[6]
    @property
    def xi(self):
        return self.data[7]
    @property
    def Phi(self):
        return self.data[8]
    @property
    def Psi(self):
        return self.data[9]

    def tree_flatten(self):
        return ((self.data,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def d_dx(u, dx, bc = "Sommerfeld"):
    weights = jnp.array([3, -32, 168, -672, 0,  672, -168, 32, -3]).reshape(1,1,9,1) / (840.0 * dx)
    du = jax.lax.conv_general_dilated(u[None, None, :, :], weights, (1,1), 'SAME', ('NCHW', 'OIHW', 'NCHW'))
    du = du.reshape(u.shape)

    if bc == "Sommerfeld":
        #Written by Gemini Pro 3.1 
        left_block = u[:9, :]
        right_block = u[-9:, :]
        du = du.at[0, :].set(jnp.dot(W0_1, left_block) / dx)
        du = du.at[1, :].set(jnp.dot(W1_1, left_block) / dx)
        du = du.at[2, :].set(jnp.dot(W2_1, left_block) / dx)
        du = du.at[3, :].set(jnp.dot(W3_1, left_block) / dx)
        du = du.at[-1, :].set(jnp.dot(-W0_1[::-1], right_block) / dx)
        du = du.at[-2, :].set(jnp.dot(-W1_1[::-1], right_block) / dx)
        du = du.at[-3, :].set(jnp.dot(-W2_1[::-1], right_block) / dx)
        du = du.at[-4, :].set(jnp.dot(-W3_1[::-1], right_block) / dx)
    
    return du

def d_dy(u, dy, bc = "Sommerfeld"):
    weights = jnp.array([3, -32, 168, -672, 0,  672, -168, 32, -3]).reshape(1,1,1,9) / (840.0 * dy)
    du = jax.lax.conv_general_dilated(u[None, None, :, :], weights, (1,1), 'SAME', ('NCHW', 'OIHW', 'NCHW'))
    du = du.reshape(u.shape)

    if bc == "Sommerfeld":
        #Written by Gemini Pro 3.1
        left_block = u[:, :9]
        right_block = u[:, -9:]
        du = du.at[:, 0].set(jnp.dot(left_block, W0_1) / dy)
        du = du.at[:, 1].set(jnp.dot(left_block, W1_1) / dy)
        du = du.at[:, 2].set(jnp.dot(left_block, W2_1) / dy)
        du = du.at[:, 3].set(jnp.dot(left_block, W3_1) / dy)
        du = du.at[:, -1].set(jnp.dot(right_block, -W0_1[::-1]) / dy)
        du = du.at[:, -2].set(jnp.dot(right_block, -W1_1[::-1]) / dy)
        du = du.at[:, -3].set(jnp.dot(right_block, -W2_1[::-1]) / dy)
        du = du.at[:, -4].set(jnp.dot(right_block, -W3_1[::-1]) / dy)

    return du

def d2_dx2(u, dx, bc = "Sommerfeld"):
    weights = jnp.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]).reshape(1,1,9,1) / (5040.0 * dx ** 2)
    du = jax.lax.conv_general_dilated(u[None, None, :, :], weights, (1,1), 'SAME', ('NCHW', 'OIHW', 'NCHW'))
    du = du.reshape(u.shape)

    if bc == "Sommerfeld":
        #Written by Gemini Pro 3.1
        left_block = u[:9, :]
        right_block = u[-9:, :]
        du = du.at[0, :].set(jnp.dot(W0_2, left_block) / dx**2)
        du = du.at[1, :].set(jnp.dot(W1_2, left_block) / dx**2)
        du = du.at[2, :].set(jnp.dot(W2_2, left_block) / dx**2)
        du = du.at[3, :].set(jnp.dot(W3_2, left_block) / dx**2)
        du = du.at[-1, :].set(jnp.dot(W0_2[::-1], right_block) / dx**2)
        du = du.at[-2, :].set(jnp.dot(W1_2[::-1], right_block) / dx**2)
        du = du.at[-3, :].set(jnp.dot(W2_2[::-1], right_block) / dx**2)
        du = du.at[-4, :].set(jnp.dot(W3_2[::-1], right_block) / dx**2)

    return du

def d2_dy2(u, dy, bc = "Sommerfeld"):
    weights = jnp.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]).reshape(1,1,1,9) / (5040.0 * dy ** 2)
    du = jax.lax.conv_general_dilated(u[None, None, :, :], weights, (1,1), 'SAME', ('NCHW', 'OIHW', 'NCHW'))
    du = du.reshape(u.shape)

    if bc == "Sommerfeld":
        #Written by Gemini Pro 3.1
        left_block = u[:, :9]
        right_block = u[:, -9:]
        du = du.at[:, 0].set(jnp.dot(left_block, W0_2) / dy**2)
        du = du.at[:, 1].set(jnp.dot(left_block, W1_2) / dy**2)
        du = du.at[:, 2].set(jnp.dot(left_block, W2_2) / dy**2)
        du = du.at[:, 3].set(jnp.dot(left_block, W3_2) / dy**2)
        du = du.at[:, -1].set(jnp.dot(right_block, W0_2[::-1]) / dy**2)
        du = du.at[:, -2].set(jnp.dot(right_block, W1_2[::-1]) / dy**2)
        du = du.at[:, -3].set(jnp.dot(right_block, W2_2[::-1]) / dy**2)
        du = du.at[:, -4].set(jnp.dot(right_block, W3_2[::-1]) / dy**2)

    return du

class MaxwellChernSimons2D:
    def __init__(self, dx, dy, Lambda, params):
        self.dx = dx
        self.dy = dy
        self.dz = self.dx
        self.Lambda = Lambda
        self.dt = params["cfl"] * self.dx
        self.bc = params["bc"]
        self.Nx = params["Nx"]
        self.Ny = params["Ny"]

        self.xmax, self.xmin = params["xmax"], params["xmin"]
        self.ymax, self.ymin = params["ymax"], params["ymin"]
        x = jnp.linspace(self.xmin, self.xmax, self.Nx)
        y = jnp.linspace(self.ymin, self.ymax, self.Ny)
        self.X, self.Y = jnp.meshgrid(x, y, indexing='ij')
        self.R = jnp.sqrt(self.X**2 + self.Y**2)

    def rhs(self, state):
        # These terms included without density, charge density, phi, psi, or potential terms
        dt_Ex = d_dy(state.Bz, self.dy, self.bc) - 2 * self.Lambda * (state.Pi * state.Bx - state.Bz * d_dy(state.xi, self.dy, self.bc))
        dt_Ey = -d_dx(state.Bz, self.dx, self.bc) - 2 * self.Lambda * (state.Pi * state.By + state.Bz * d_dx(state.xi, self.dx, self.bc))
        dt_Ez = d_dx(state.By, self.dx, self.bc) - d_dy(state.Bx, self.dy, self.bc) - 2 * self.Lambda * (state.Pi * state.Bz + state.Ex * \
            d_dy(state.xi, self.dy, self.bc) - state.Ey * d_dx(state.xi, self.dx, self.bc))
        dt_Bx = -d_dy(state.Ez, self.dy, self.bc)
        dt_By = d_dx(state.Ez, self.dx, self.bc)
        dt_Bz = -d_dx(state.Ey, self.dx, self.bc) + d_dy(state.Ex, self.dy, self.bc)
        dt_xi =  -state.Pi
        dt_Pi = -d2_dx2(state.xi, self.dx, self.bc) - d2_dy2(state.xi, self.dx, self.bc) + 2 * self.Lambda * (state.Bx * state.Ex \
            + state.By * state.Ey + state.Bz * state.Ez)
        dt_Psi = jnp.zeros_like(state.Ex)
        dt_Phi = jnp.zeros_like(state.Ex)

        #Written by Gemini Pro 3.1
        dt_Ex = self.bc_sommerfeld(state.Ex, dt_Ex)
        dt_Ey = self.bc_sommerfeld(state.Ey, dt_Ey)
        dt_Ez = self.bc_sommerfeld(state.Ez, dt_Ez)
        dt_Bx = self.bc_sommerfeld(state.Bx, dt_Bx)
        dt_By = self.bc_sommerfeld(state.By, dt_By)
        dt_Bz = self.bc_sommerfeld(state.Bz, dt_Bz)

        #Written by Gemini Pro 3.1
        rhs_data = jnp.zeros_like(state.data)
        rhs_data = rhs_data.at[0].set(dt_Ex)
        rhs_data = rhs_data.at[1].set(dt_Ey)
        rhs_data = rhs_data.at[2].set(dt_Ez)
        rhs_data = rhs_data.at[3].set(dt_Bx)
        rhs_data = rhs_data.at[4].set(dt_By)
        rhs_data = rhs_data.at[5].set(dt_Bz)
        rhs_data = rhs_data.at[6].set(dt_xi)
        rhs_data = rhs_data.at[7].set(dt_Pi)
        rhs_data = rhs_data.at[8].set(dt_Psi)
        rhs_data = rhs_data.at[9].set(dt_Phi)
        return WaveState(rhs_data)

        #Boundary Conditions

    def bc_sommerfeld(self, u, dtu, depth = 4):
        strip = 2 * depth + 1

        #Xmax
        left = jnp.s_[:depth, :]
        dtu = dtu.at[left].set((-self.X[left] * d_dx(u[:strip, :], self.dx)[:depth, :] - self.Y[left]*d_dy(u[:strip, :], self.dy)[:depth, :]) / self.R[left])

        #Xmin
        right = jnp.s_[-depth:, :]
        dtu = dtu.at[ right].set((-self.X[ right] * d_dx(u[-strip:, :], self.dx)[-depth: , :] - self.Y[ right]*d_dy(u[-strip:, :], self.dy)[-depth: , :]) / self.R[ right])

        #Ymax
        top = jnp.s_[depth: -depth, -depth:]
        dtu = dtu.at[top].set((-self.X[top] * d_dx(u[depth: -depth, -strip:], self.dx)[:, -depth:] - self.Y[top]*d_dy(u[depth: -depth, -strip:], self.dy)[:, -depth:]) / self.R[top])

        #Ymin
        bottom = jnp.s_[depth: -depth, :depth]
        dtu = dtu.at[bottom].set((-self.X[bottom] * d_dx(u[depth: -depth, :strip], self.dx)[:, :depth] - self.Y[bottom]*d_dy(u[depth: -depth, :strip], self.dy)[:, :depth]) / self.R[bottom])

        return dtu

    #Written by Gemini Pro 3.1
    def step_rk4(self, state, dt):
        k1 = self.rhs(state)
        state_k2 = jax.tree_map(lambda s, k: s + 0.5 * dt * k, state, k1)
        k2 = self.rhs(state_k2)
        state_k3 = jax.tree_map(lambda s, k: s + 0.5 * dt * k, state, k2)
        k3 = self.rhs(state_k3)
        state_k4 = jax.tree_map(lambda s, k: s + dt * k, state, k3)
        k4 = self.rhs(state_k4)
        def rk4_combine(s, k1_val, k2_val, k3_val, k4_val):
            return s + (dt / 6.0) * (k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val)
            
        new_state = jax.tree_map(rk4_combine, state, k1, k2, k3, k4)
        return new_state

def initialize_gaussian(X, Y, R, dx, dy, params):
    x0, y0 = params["id_x0"], params["id_y0"]
    B0, sigma = params["id_amp"], params["id_sigma"]
    data = jnp.zeros((10, X.shape[0], X.shape[1]))
    Az = B0 * jnp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
    data = data.at[0].set(d_dy(Az, dy)) #Ex as gaussian pulse
    data = data.at[1].set(-(d_dx(Az, dx))) #Ey from constraint equations, assumes that xi is 0
    #Ez, is freely specifiable, left as 0s for convencience
    data = data.at[3].set(d_dy(Az, dy)) #Bx
    data = data.at[4].set(-(d_dx(Az, dx))) #By from constraint equations
    #Bz, xi, Pi, Phi, Psi, are all intially set to 0 for convencience (but bz, xi, and pi can be changed)
    return WaveState(data)

def cal_constraints(state, dx, dy, Lambda):
    divE_error = d_dx(state.Ex, dx) + d_dy(state.Ey, dy) + 2 * Lambda * (state.Bx * d_dx(state.xi, dx) \
        + state.By * d_dy(state.xi, dy))
    divB_error = d_dx(state.Bx, dx) + d_dy(state.By, dy)
    return divE_error, divB_error

def l2norm(u):
    return np.sqrt(np.mean(u**2))

def main(parfile, output_dir):
    if not os.path.exists(parfile):
        print(f"Error: Parameter file {parfile} not found.")
        # Create a dummy params dict for testing if file missing
        params = {
            "Nx": 256, "Ny": 256, "Nt": 100, "output_interval": 10,
            "cfl": 0.25, "ko_sigma": 0.01, "Lambda": 1,
            "id_amp": 0.8, "id_sigma": 1.5, "id_y0": 0.0, "id_x0": 0.0,
            "xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "bc": "Sommerfeld"
        }
    else:
        with open(parfile, "rb") as f: 
            params = tomllib.load(f)
        
    nx, ny = params["Nx"], params["Ny"]    
    nt, out_int = params["Nt"], params["output_interval"]
    dx = (params["xmax"] - params["xmin"]) / (nx - 1)
    dy = (params["ymax"] - params["ymin"]) / (ny - 1)
    Lambda = params.get("Lambda")
    names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Phi", "Psi"]

    sim = MaxwellChernSimons2D(dx, dy, Lambda, params)
    state = initialize_gaussian(sim.X, sim.Y, sim.R, dx, dy, params)

    if iox:
        iox.write_hdf5(0, np.asarray(state.data), np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
    
    print(f"Starting NLSM Simulation | Nx={nx} Ny={ny} Nt={nt}")
    
    @jax.jit
    def time_step(i, current_state):
        return sim.step_rk4(current_state, sim.dt)

    for s in range(out_int, nt + 1, out_int):
        state = jax.lax.fori_loop(0, out_int, time_step, state)
        state.data.block_until_ready()
        
        if iox:
            iox.write_hdf5(s, state.data, np.array(sim.X[:,0]), np.array(sim.Y[0,:]), names, output_dir)
            
    if iox:
        iox.write_xdmf(output_dir, nt, nx, ny, names, out_int, sim.dt)

if __name__ == "__main__":
    par = sys.argv[1] if len(sys.argv) > 1 else "params.toml"
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    os.makedirs(out, exist_ok=True)
    main(par, out)