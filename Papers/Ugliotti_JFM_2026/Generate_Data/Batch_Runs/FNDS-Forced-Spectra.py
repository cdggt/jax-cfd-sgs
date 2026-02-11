#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import time

import numpy as np
from tqdm import tqdm
import os
import xarray as xr

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# =========================================================
# Parameters
# =========================================================
Nx_LES = 256
DELTA_DEFAULT = 6.0 * np.pi / Nx_LES
DEALIAS_FRAC = 2.0 / 3.0   # box cutoff at N/3


# =========================================================
# Gaussian filter + box spectral cutoff
# =========================================================
def gaussian_filter_and_box_cutoff(w, delta, Lx=2.0*np.pi, Ly=2.0*np.pi):
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D array, got {w.shape}")

    Nx, Ny = w.shape
    dx = Lx / Nx
    dy = Ly / Ny

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2

    # Gaussian filter
    G_hat = np.exp(-(k2 * delta**2) / 24.0)

    # Box de-aliasing mask: cutoff at 2/3 Nyquist
    kx_nyq = (Nx // 2) * (2.0 * np.pi / Lx)
    ky_nyq = (Ny // 2) * (2.0 * np.pi / Ly)
    kx_cut = DEALIAS_FRAC * kx_nyq
    ky_cut = DEALIAS_FRAC * ky_nyq
    box_mask = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)

    w_hat = np.fft.fft2(w)
    w_hat_filt = w_hat * G_hat * box_mask
    return np.fft.ifft2(w_hat_filt).real


# =========================================================
# Energy / enstrophy spectrum from vorticity
# =========================================================
def spectrum_from_vorticity(w, Lx=2.0*np.pi, Ly=2.0*np.pi):
    Nx, Ny = w.shape
    dx = Lx / Nx
    dy = Ly / Ny

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2

    w_hat = np.fft.fft2(w)

    psi_hat = np.zeros_like(w_hat, dtype=np.complex128)
    mask = k2 > 0
    psi_hat[mask] = -w_hat[mask] / k2[mask]

    u_hat = 1j * KY * psi_hat
    v_hat = -1j * KX * psi_hat

    E2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    k_mag = np.sqrt(k2)
    k_idx = np.rint(k_mag).astype(int)

    kmax = k_idx.max()
    Ek = np.bincount(k_idx.ravel(), weights=E2d.ravel(), minlength=kmax + 1)

    return np.arange(kmax + 1), Ek


# =========================================================
# Robust open helper (retry + skip)
# =========================================================
def open_w_robust(fpath: Path, varname="w", retries=1, sleep_s=0.25):
    """
    Returns w array or raises the last exception.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            ds = xr.open_dataset(fpath, engine="netcdf4", cache=False)
            try:
                if varname not in ds:
                    raise KeyError(f"Variable '{varname}' not found. Vars: {list(ds.data_vars)}")
                w = ds[varname].values
                return w
            finally:
                ds.close()
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s)
            else:
                raise last_err


# =========================================================
# Optional: pre-scan files to only keep readable ones
# =========================================================
def filter_good_files(files, varname="w", retries=1):
    good = []
    bad = []
    for f in tqdm(files, desc="Pre-scan netCDF", leave=False):
        try:
            _ = open_w_robust(f, varname=varname, retries=retries)
            good.append(f)
        except Exception as e:
            bad.append((str(f), repr(e)))
    return good, bad


# =========================================================
# Spectrum for ONE ID (time-mean + time-uncertainty)
# =========================================================
def compute_spectrum_for_id(base_dir, folder, node_id, spectrum_type, delta, start_file,
                            prescan=False, retries=1, log_dir=Path(".")):
    node_str = f"{node_id:03d}"
    print(f"\n--- ID {node_str} ---")

    data_dir = Path(base_dir) / folder
    files = sorted(data_dir.glob(f"data_id{node_str}_*.nc"))

    if len(files) <= start_file:
        raise RuntimeError(f"Not enough files for ID {node_str} (found {len(files)})")

    files = files[start_file:]
    print(f"Using {len(files)} files (skipped first {start_file})")

    bad_files = []

    if prescan:
        files, bad0 = filter_good_files(files, varname="w", retries=retries)
        bad_files.extend(bad0)
        print(f"[INFO] Pre-scan kept {len(files)} good files, skipped {len(bad0)} bad files")

    k_ref = None
    sum1 = None
    sum2 = None
    nsamp = 0

    for fpath in tqdm(files, desc=f"Spectra ID {node_str}"):
        try:
            w = open_w_robust(fpath, varname="w", retries=retries)
        except Exception as e:
            bad_files.append((str(fpath), repr(e)))
            print(f"\n[WARN] Skipping unreadable file: {fpath}\n       {e}\n")
            continue

        # compute
        w_f = gaussian_filter_and_box_cutoff(w, delta=delta)
        k, Ek = spectrum_from_vorticity(w_f)

        if spectrum_type == "enstrophy":
            Ek = (k**2) * Ek

        if sum1 is None:
            k_ref = k
            sum1 = Ek.astype(np.float64, copy=True)
            sum2 = (Ek.astype(np.float64, copy=False) ** 2)
        else:
            n = min(len(sum1), len(Ek))
            sum1 = sum1[:n]
            sum2 = sum2[:n]
            Ek = Ek[:n]
            k_ref = k_ref[:n]
            sum1 += Ek
            sum2 += Ek**2

        nsamp += 1

    if bad_files:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"bad_files_spectrum_ID{node_id:03d}.txt"
        with open(log_path, "w") as fp:
            for path, err in bad_files:
                fp.write(f"{path}\t{err}\n")
        print(f"[INFO] Wrote bad-file log: {log_path} (skipped {len(bad_files)} files)")

    if nsamp == 0:
        raise RuntimeError(f"No samples processed for ID {node_str} (all files bad?)")

    mean_t = sum1 / nsamp
    var_t = sum2 / nsamp - mean_t**2
    var_t = np.maximum(var_t, 0.0)
    std_t = np.sqrt(var_t)
    sem_t = std_t / np.sqrt(nsamp)

    return k_ref, mean_t, std_t, sem_t, nsamp


# =========================================================
# MAIN
# =========================================================
def main():
    p = argparse.ArgumentParser(description="FDNS filtered + dealiased spectrum + uncertainties (robust IO)")
    p.add_argument("--folder", required=True)
    p.add_argument("--base-dir", default="/storage/home/hcoda1/4/mugliotti3/scratch/forced_lowfreq")
    p.add_argument("--ids", type=int, nargs="+", required=True)
    p.add_argument("--spectrum-type", choices=["energy", "enstrophy"], default="energy")
    p.add_argument("--delta", type=float, default=DELTA_DEFAULT)
    p.add_argument("--start-file", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-name", default="FDNS_filtered_dealiased_spectrum_with_uncertainties.txt")

    # robustness toggles
    p.add_argument("--prescan", action="store_true", help="Pre-scan and keep only readable netCDF files")
    p.add_argument("--retries", type=int, default=1, help="Retry count for netCDF open (transient IO)")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_means = []
    id_time_std = []
    id_time_sem = []
    nsamps = []
    k_ref = None

    for nid in args.ids:
        k, mean_t, std_t, sem_t, nsamp = compute_spectrum_for_id(
            args.base_dir,
            args.folder,
            nid,
            args.spectrum_type,
            args.delta,
            args.start_file,
            prescan=args.prescan,
            retries=args.retries,
            log_dir=out_dir,
        )

        if k_ref is None:
            k_ref = k
        else:
            n = min(len(k_ref), len(k))
            k_ref = k_ref[:n]
            mean_t = mean_t[:n]
            std_t = std_t[:n]
            sem_t = sem_t[:n]

        id_means.append(mean_t)
        id_time_std.append(std_t)
        id_time_sem.append(sem_t)
        nsamps.append(nsamp)

    id_means = np.stack(id_means)
    id_time_std = np.stack(id_time_std)
    id_time_sem = np.stack(id_time_sem)

    n_ids = id_means.shape[0]
    mean_over_ids = id_means.mean(axis=0)

    if n_ids >= 2:
        std_across_ids = id_means.std(axis=0, ddof=1)
    else:
        std_across_ids = np.zeros_like(mean_over_ids)
    sem_across_ids = std_across_ids / np.sqrt(n_ids)

    mean_time_std = id_time_std.mean(axis=0)
    mean_time_sem = id_time_sem.mean(axis=0)

    out_file = out_dir / args.out_name
    out_mat = np.column_stack((
        k_ref,
        mean_over_ids,
        std_across_ids,
        sem_across_ids,
        mean_time_std,
        mean_time_sem
    ))

    header = (
        "k  spectrum_mean_over_ids  std_across_ids  sem_across_ids  "
        "mean_time_std_within_id  mean_time_sem_within_id  "
        f"(n_ids={n_ids}, nsamp_per_id={nsamps}, prescan={args.prescan}, retries={args.retries})"
    )

    np.savetxt(out_file, out_mat, header=header)
    print(f"\nSaved spectrum → {out_file}")


if __name__ == "__main__":
    main()
