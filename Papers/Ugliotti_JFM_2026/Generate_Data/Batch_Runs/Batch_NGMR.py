import argparse
import sys
import dataclasses
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
import re

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax_cfd.spectral.subgrid_models as subgrid_models

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run JAX-CFD simulation on specific node.')
    parser.add_argument('--node_id', type=int, default=1, help='ID of the current node/job (e.g., 1, 2, 3, 4)')
    args = parser.parse_args()
    
    # Generate the ID string (e.g., "001")
    node_str = f"{args.node_id:03d}"
    print(f"Starting simulation for Node ID: {node_str}")

    # --- 2. Load Data ---
    # Update this path to your specific location
    filename = '/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq/IC/IC_4096_2.nc'
    
    try:
        with xr.open_dataset(filename) as ds:
            # Load initial velocity field
            w = ds.w.values
            params = ds.attrs
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    amp = params['amp']   
    nu = params['nu']
    drag = params['drag']
    kf = params['kf']

    # --- 3. Setup Grid and Physics ---
    file_name_dir = "LCR_48"
    nt = 5000
    inner_loop = 50
    n_dns = w.shape[1]
    n_les = 48
    Delta = 6 *jnp.pi / n_les
    grid_dns = grids.Grid(shape=(n_dns,n_dns), domain=((0, 2*np.pi), (0, 2*np.pi)))
    # Build les grid
    grid = grids.Grid(shape=(n_les,n_les), domain=((0, 2*np.pi), (0, 2*np.pi)))
    forcing_fn = lambda grid: spectral.forcings.checkerboard_forcing_module(
            grid, freq=kf, amp=amp)
    
    lcr = subgrid_models.LES_w_TR(nu,grid,Delta,smooth=True,drag=drag,k_filter=2/3,forcing_fn=forcing_fn)

    w_to_v = spectral.utils.vorticity_to_velocity(grid)
    w_to_v_dns = spectral.utils.vorticity_to_velocity(grid_dns)
    fourier_filter = spectral.utils.get_filter_fn(Delta=Delta,grid=grid) 	
    # Calculate stable time step
    def vmax(w_hat):
        u_hat = jnp.stack(w_to_v_dns(w_hat),axis=-1)
        u = jnp.fft.irfft2(u_hat,axes=(0,1))**2
        u_mag = jnp.sqrt(jnp.sum(u,axis=-1))
        return u_mag.max()
    w_hat = jnp.fft.rfft2(w)
    v_rms = vmax(w_hat)
    dt = cfd.equations.stable_time_step(v_rms, .4, nu, grid, implicit_diffusion=True)

    # Fixed Coords definition
    xs = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
    coords = {
        'x': xs,
        'y': xs
    }

    # --- 4. Setup Solver ---
    dT = 5
    inner_steps = int(dT // dt)
    step_fn = spectral.time_stepping.crank_nicolson_rk4(lcr, dt)
    adv_fn = cfd.funcutils.repeated(step_fn, inner_steps)

    def make_adv_sim(grid):
        def adv_sim_fn(state, _):
            state = adv_fn(state)
            u_les, Rij_hat = state
            Rij = jnp.fft.irfft2(Rij_hat,axes=(0,1))
            w_hat = spectral.utils.spectral_curl_2d(grid, (u_les[...,0],u_les[...,1]))
            w = jnp.fft.irfft2(w_hat,axes=(0,1))
            carry = w, Rij
            return state, carry
        return adv_sim_fn

    # --- 5. Directory Setup ---
    # Ensure this points to your scratch directory
    base_dir = Path("/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq").expanduser() 
    data_set_path = base_dir / file_name_dir
    data_set_path.mkdir(parents=True, exist_ok=True)

    # --- 6. Initialization & Perturbation ---
    # Convert velocity to spectral velocity (from DNS data)
    _,_,R_fn = spectral.utils.get_LCR_map(Delta=Delta,grid=grid_dns)

    def init_R(u):
        Rij = R_fn(u)
        Rij = jnp.reshape(Rij,[n_dns,n_dns,-1])
        Rij = spectral.utils.down_res(Rij,[n_les,n_les])
        Rij_hat = jnp.stack([0.5*(Rij[:, :, 0] - Rij[:, :, 3]), Rij[:, :, 1]],axis=-1)
        Rij = jnp.fft.irfft2(Rij_hat,axes=(0,1))
        return Rij_hat
    def incomp_noise(noise):
        noise = jnp.fft.rfft2(noise,axes=(0,1))
        Dx, Dy = tuple( 2j*jnp.pi * k for k in grid_dns.rfft_mesh() )
        # Leray projection: u_perp = u - k (k·u)/|k|^2
        div_hat = Dx * noise[..., 0] + Dy * noise[..., 1] 		 # k·u
        laplace = (Dx**2 + Dy**2)
        inv_laplace = jnp.where(laplace == 0, 0.0, 1 / laplace)
        proj_x = noise[..., 0] - Dx * div_hat * inv_laplace
        proj_y = noise[..., 1] - Dy * div_hat * inv_laplace
        u_noise = jnp.stack([proj_x, proj_y], axis=-1)
        return u_noise
    # Check for existing files for THIS specific node ID
    # Pattern: data_id{ID}_{step}.nc, where {step} can be 3 to 6 digits
    # Example: data_id001_0000.nc, data_id001_123456.nc
    # ------------------------------------------------------------
    # Restart: look for latest saved block for THIS node_id
    # You SAVE: data_comp_id{node_str}_{ii:05d}.nc
    # ------------------------------------------------------------
    file_pattern = re.compile(rf"data_comp_id{node_str}_(\d+)\.nc$")
    existing_files = sorted(data_set_path.glob(f"data_comp_id{node_str}_*.nc"))

    last_step = -1
    last_file = None

    for file_path in existing_files:
        match = file_pattern.match(file_path.name)
        if match:
            step = int(match.group(1))  # this is your outer-loop index ii
            if step > last_step:
                last_step = step
                last_file = file_path

    if last_file is not None:
        print(f"Found existing files for ID {node_str}.")
        print(f"Restarting from outer-loop index ii={last_step} (file: {last_file.name}).")

        try:
            ds = xr.load_dataset(last_file)

            # TAKE LAST TIME SLICE (you saved w as (t,x,y) and R as (t,x,y,a))
            w_real_last = jnp.asarray(ds["w"].isel(t=-1).values)   # (x,y)
            R_real_last = jnp.asarray(ds["R"].isel(t=-1).values)   # (x,y,a)

            # Back to spectral
            w_hat_last = jnp.fft.rfft2(w_real_last, axes=(0, 1))          # (kx,ky_rfft)
            Rij_hat = jnp.fft.rfft2(R_real_last, axes=(0, 1))             # (kx,ky_rfft,a)

            # Spectral velocity from spectral vorticity (LES grid)
            u_les = jnp.stack(spectral.utils.vorticity_to_velocity(grid)(w_hat_last), axis=-1)

            state = (u_les, Rij_hat)

            # Continue at next outer-loop block
            start = last_step + 1
            print(f"Successfully loaded {last_file.name}. Starting loop at ii={start}.")

        except Exception as e:
            print(f"Error loading or processing last file {last_file}: {e}")
            print("Falling back to fresh initialization from DNS IC.")
            last_file = None  # force fresh init below

    if last_file is None:
        # --- Initial Setup (unchanged logic, just guaranteed to run cleanly) ---
        print(f"No existing files found for ID {node_str}. Starting fresh from DNS data.")
        if args.node_id > 0:
            print(f"Applying perturbation for ensemble member {args.node_id}")
            key = jax.random.PRNGKey(args.node_id)
            noise = jax.random.normal(key, shape=(n_dns, n_dns)) * 1e-10
            w = w + noise

        u = jnp.fft.irfft2(jnp.stack(w_to_v_dns(jnp.fft.rfft2(w)), axis=-1), axes=(0, 1))
        Rij_hat = init_R(u)
        u_les = spectral.utils.down_res(u, [n_les, n_les])
        u_les = fourier_filter(u_les)

        state = (u_les, Rij_hat)
        start = 0

            
    else:
        # --- Initial Setup ---
        print(f"No existing files found for ID {node_str}. Starting fresh from DNS data.")
        if args.node_id > 0:
            print(f"Applying perturbation for ensemble member {args.node_id}")
            key = jax.random.PRNGKey(args.node_id)
            # Create random noise in spectral space
            noise = jax.random.normal(key, shape=(n_dns,n_dns)) * 1e-10
            w = w + noise
            u = jnp.fft.irfft2(jnp.stack(w_to_v_dns(jnp.fft.rfft2(w)),axis=-1),axes=(0,1))
            # Initialize u and R
            Rij_hat = init_R(u)
            u_les = spectral.utils.down_res(u,[n_les,n_les])
            u_les = fourier_filter(u_les)
        
        state = u_les, Rij_hat
        start = 0

    
    print(Rij_hat.shape)
    

    # --- 7. Main Loop ---
    # JIT compile the step function for performance
    adv_sim_jitted = jax.jit(make_adv_sim(grid))

    print(f"Starting loop for {nt} steps...")
    outer_loops = nt//inner_loop
    pbar = tqdm(range(start, outer_loops))
    for ii in pbar:
        # Updated naming convention: data_id{ID}_{step}.nc
        iter_name = f"data_comp_id{node_str}_{ii:05d}.nc"
        full_path = data_set_path / iter_name
        
        # Step simulation
        state, trajectory = jax.lax.scan(adv_sim_jitted,state,length=inner_loop)
        w, Rij = trajectory
        
        t = (np.arange(inner_loop) + ii)*dT
        pbar.set_postfix({"Enstrophy": f"{(w[0,...]**2).mean():.3e}"})
        ds = xr.Dataset(
            data_vars={"w": (["t","x", "y"], w),
                       "R": (["t","x", "y","a"], Rij)},
            coords={**coords,"t":t},
            attrs={'nu': nu, 'dt': dT, 'node_id': args.node_id, 'delta': Delta}
        )
        
        ds.to_netcdf(full_path)

    print("Simulation complete.")

if __name__ == "__main__":
    main()
