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
import jax_cfd.analysis.util as util
import jax_cfd.spectral.utils as spu

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
    filename = '/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq/IC/IC_8192.nc'
    #filename = '/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq/IC/IC_8192_highfreq.nc'
    
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
    nt = 400
    n = w.shape[1]
    grid = grids.Grid(shape=(n, n), domain=((0, 2*np.pi), (0, 2*np.pi)))
    
    forcing_fn = lambda grid: spectral.forcings.checkerboard_forcing_module(
            grid, freq=kf, amp=amp)
    
    ns = spectral.equations.NavierStokes2D(nu, grid, drag=drag,smooth=True, forcing_fn=forcing_fn)
    w_to_v = spectral.utils.vorticity_to_velocity(grid)

    # Calculate stable time step
    def vmax(w_hat):
        u_hat = jnp.stack(w_to_v(w_hat),axis=-1)
        u = jnp.fft.irfft2(u_hat,axes=(0,1))**2
        u_mag = jnp.sqrt(jnp.sum(u,axis=-1))
        return u_mag.max()

    v_rms = vmax(jnp.fft.rfft2(w))
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
    step_fn = spectral.time_stepping.crank_nicolson_rk4(ns, dt)
    adv_fn = cfd.funcutils.repeated(step_fn, inner_steps)

    # --- 5. Directory Setup ---
    file_name_dir = "DNS_8192"
    #file_name_dir = "DNS_8192_highfreq"
    # Ensure this points to your scratch directory
    base_dir = Path("/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq").expanduser() 
    data_set_path = base_dir / file_name_dir
    data_set_path.mkdir(parents=True, exist_ok=True)

    # --- 6. Initialization & Perturbation ---
    # Convert velocity to spectral velocity (from DNS data)

    # Check for existing files for THIS specific node ID
    # Pattern: data_id{ID}_{step}.nc, where {step} can be 3 to 6 digits
    # Example: data_id001_0000.nc, data_id001_123456.nc
    file_pattern = re.compile(rf"data_id{node_str}_(\d+)\.nc")
    existing_files = sorted(data_set_path.glob(f"data_id{node_str}_*.nc"))

    # Find the latest file based on the step number in the filename
    last_step = -1
    last_file = None

    for file_path in existing_files:
        match = file_pattern.match(file_path.name)
        if match:
            step = int(match.group(1))
            if step > last_step:
                last_step = step
                last_file = file_path

    if last_file:
        print(f"Found existing files for ID {node_str}.")
        print(f"Restarting from step {last_step}.")
        
        # --- Load Last State ---
        try:
            ds = xr.load_dataset(last_file)
            # The saved variable is real-space vorticity 'w'
            w_real_last = ds['w'].values
            
            # Convert real-space vorticity back to spectral velocity 'state'
            # 1. Real Vorticity -> Spectral Vorticity (w_hat)
            w_hat = jnp.fft.rfft2(w_real_last, axes=(0, 1))

            start = last_step + 1
            print(f"Successfully loaded {last_file.name}. Starting loop at step {start}.")
            
        except Exception as e:
            print(f"Error loading or processing last file {last_file}: {e}")
            # If loading fails, fall through to initial DNS setup
            w_hat = jnp.fft.rfft2(w)
            start = 0
            
    else:
        # --- Initial Setup ---
        print(f"No existing files found for ID {node_str}. Starting fresh from DNS data.")
        w_hat = jnp.fft.rfft2(w)
        start = 0

    # If starting fresh (start == 0) and a perturbation is needed for the ensemble
    if start == 0 and args.node_id > 0:
        print(f"Applying perturbation for ensemble member {args.node_id}")
        key = jax.random.PRNGKey(args.node_id)

        # compute current vorticity RMS (real space)
        w0 = jnp.fft.irfft2(w_hat, s=(n, n), axes=(0, 1))
        w_rms = jnp.sqrt(jnp.mean(w0**2))

        # choose relative perturbation level (good default)
        rel = 1e-3   # try 1e-6; if too slow separation use 1e-5

        # Create random noise in spectral space (same as you do, just scaled)
        noise = jnp.fft.rfft2(jax.random.normal(key, shape=(n, n)) * (rel * w_rms), axes=(0, 1))

        # keep mean vorticity unchanged (recommended)
        noise = noise.at[0, 0].set(0.0 + 0.0j)

        w_hat = w_hat + noise

        # optional: print relative size you actually injected
        noise_real = jnp.fft.irfft2(noise, s=(n, n), axes=(0, 1))
        print("Injected relative noise RMS =", jnp.sqrt(jnp.mean(noise_real**2)) / w_rms)

    # --- 7. Main Loop ---
    # JIT compile the step function for performance
    adv_fn_jitted = jax.jit(adv_fn)

    print(f"Starting loop for {nt} steps...")
    for ii in tqdm(range(start, nt)):
        # Updated naming convention: data_id{ID}_{step}.nc
        iter_name = f"data_id{node_str}_{ii:03d}.nc"
        full_path = data_set_path / iter_name
        
        # Step simulation
        w_hat = adv_fn_jitted(w_hat)
        
        # Convert to real space for saving
        # Note: w_hat is vorticity (scalar field), so we save as 2D (x, y)
        w_real = jnp.fft.irfftn(w_hat, axes=(0, 1))
        
        ds = xr.Dataset(
            data_vars={"w": (["x", "y"], w_real)},
            coords=coords,
            attrs={'nu': nu,'drag':drag, 'dt': dT, 'node_id': args.node_id}
        )
        
        ds.to_netcdf(full_path)

    print("Simulation complete.")

if __name__ == "__main__":
    main()
