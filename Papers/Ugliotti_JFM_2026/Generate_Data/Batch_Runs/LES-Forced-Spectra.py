#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


# ---------------------------------------------------------
# Helper: spectrum from a single vorticity snapshot
# ---------------------------------------------------------
def spectrum_from_vorticity_snapshot(w, Lx=2.0 * np.pi, Ly=2.0 * np.pi, shell_average=False):
    """
    Energy spectrum E(k) from vorticity w(x,y) on periodic domain.

    Uses norm="forward" so Parseval is consistent.
    """
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D array, got {w.shape}")

    nx, ny = w.shape
    dx = Lx / nx
    dy = Ly / ny

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing="ij")
    ksq = kx2d**2 + ky2d**2

    w_hat = np.fft.fft2(w, norm="forward")

    psi_hat = np.zeros_like(w_hat, dtype=np.complex128)
    mask = ksq > 0.0
    psi_hat[mask] = -w_hat[mask] / ksq[mask]

    u_hat = 1j * ky2d * psi_hat
    v_hat = -1j * kx2d * psi_hat

    E2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    k_mag = np.sqrt(ksq)
    k_shell = np.rint(k_mag).astype(int)
    k_max = int(k_shell.max())

    flat_shell = k_shell.ravel()
    flat_E = E2d.ravel()

    Ek_shell = np.bincount(flat_shell, weights=flat_E, minlength=k_max + 1)

    if shell_average:
        counts = np.bincount(flat_shell, minlength=k_max + 1)
        nz = counts > 0
        Ek_shell[nz] /= counts[nz]

    k_vals = np.arange(k_max + 1, dtype=int)
    return k_vals, Ek_shell


# ---------------------------------------------------------
# File discovery (ROBUST)
# ---------------------------------------------------------
def _glob_files_for_id(base_dir: Path, node_id: int, prefixes):
    node_str = f"{node_id:03d}"
    hits = []
    for pref in prefixes:
        hits.extend(base_dir.glob(f"{pref}{node_str}_*.nc"))
    return sorted(set(hits))


def discover_ids(base_dir: Path, prefixes, max_id=999):
    ids = []
    for i in range(1, max_id + 1):
        if _glob_files_for_id(base_dir, i, prefixes):
            ids.append(i)
    return ids


# ---------------------------------------------------------
# Average spectrum over files for one ID
# ---------------------------------------------------------
def compute_average_spectrum_one_id(
    base_dir: Path,
    node_id: int,
    spectrum_type: str = "energy",
    start_snapshot: int = 0,
    prefixes=("data_comp_id", "data_id"),
    verbose=True,
):
    files = _glob_files_for_id(base_dir, node_id, prefixes)
    if not files:
        raise RuntimeError(
            f"No files for ID {node_id:03d} in {base_dir}. Prefixes tried: {list(prefixes)}"
        )

    if verbose:
        print(f"ID {node_id:03d}: found {len(files)} files. First={files[0].name} Last={files[-1].name}")

    # metadata
    Lx = Ly = 2.0 * np.pi
    sim_nu = np.nan
    sim_res = -1
    try:
        with xr.open_dataset(files[0]) as ds_meta:
            sim_nu = ds_meta.attrs.get("nu", np.nan)
            if "x" in ds_meta.dims:
                sim_res = int(ds_meta.dims["x"])
            elif "w" in ds_meta:
                sim_res = int(ds_meta["w"].shape[-2])
            Lx = float(ds_meta.attrs.get("Lx", 2.0 * np.pi))
            Ly = float(ds_meta.attrs.get("Ly", 2.0 * np.pi))
    except Exception as e:
        if verbose:
            print(f"[WARN] Could not read metadata from {files[0]}: {e}")

    k_ref = None
    accum = None
    num_snapshots = 0

    for fpath in tqdm(files, desc=f"ID {node_id:03d}: spectra", leave=False):
        try:
            with xr.open_dataset(fpath) as ds:
                if "w" not in ds:
                    continue

                w = ds["w"].values

                if w.ndim == 2:
                    snapshots = [w]
                elif w.ndim == 3:
                    t0 = max(int(start_snapshot), 0)
                    if t0 >= w.shape[0]:
                        continue
                    snapshots = [w[i, ...] for i in range(t0, w.shape[0])]
                else:
                    continue

                for w_snap in snapshots:
                    if not np.isfinite(w_snap).all():
                        # hard skip NaN snapshots to avoid poisoning the average
                        continue

                    k_vals, Ek = spectrum_from_vorticity_snapshot(w_snap, Lx=Lx, Ly=Ly)

                    if spectrum_type == "enstrophy":
                        Ek = (k_vals**2) * Ek

                    if k_ref is None:
                        k_ref = k_vals
                        accum = np.zeros_like(Ek, dtype=float)
                    else:
                        if len(k_vals) != len(k_ref):
                            m = min(len(k_vals), len(k_ref))
                            k_ref = k_ref[:m]
                            accum = accum[:m]
                            Ek = Ek[:m]

                    accum += Ek
                    num_snapshots += 1

        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping file {fpath.name}: {e}")

    if num_snapshots == 0:
        raise RuntimeError(f"No snapshots processed for ID {node_id:03d} (all skipped or empty).")

    avg = accum / float(num_snapshots)
    meta = {
        "node_id": int(node_id),
        "resolution": int(sim_res),
        "viscosity": float(sim_nu) if np.isfinite(sim_nu) else float("nan"),
        "domain_Lx": float(Lx),
        "domain_Ly": float(Ly),
        "num_snapshots": int(num_snapshots),
        "spectrum_type": spectrum_type,
    }
    return np.asarray(k_ref), np.asarray(avg), meta


# ---------------------------------------------------------
# Average spectrum across multiple IDs
# ---------------------------------------------------------
def compute_average_spectrum_all_ids(
    base_dir: Path,
    id_list=None,
    spectrum_type: str = "energy",
    start_snapshot: int = 0,
    prefixes=("data_comp_id", "data_id"),
):
    if id_list is None:
        id_list = discover_ids(base_dir, prefixes=prefixes)

    if not id_list:
        raise RuntimeError(f"No IDs found in {base_dir} (prefixes tried: {list(prefixes)})")

    k_ref = None
    accum = None
    ids_used = []

    print(f"Base dir: {base_dir}")
    print(f"Prefixes tried: {list(prefixes)}")
    print(f"IDs requested: {id_list}")

    for node_id in id_list:
        print(f"\n--- Processing ID {node_id:03d} ---")
        try:
            k, spec, _meta = compute_average_spectrum_one_id(
                base_dir=base_dir,
                node_id=node_id,
                spectrum_type=spectrum_type,
                start_snapshot=start_snapshot,
                prefixes=prefixes,
                verbose=True,
            )
        except Exception as e:
            print(f"[WARN] Skipping ID {node_id:03d}: {e}")
            continue

        if k_ref is None:
            k_ref = k
            accum = np.zeros_like(spec, dtype=float)
        else:
            if len(k) != len(k_ref):
                m = min(len(k), len(k_ref))
                k_ref = k_ref[:m]
                accum = accum[:m]
                spec = spec[:m]

        accum += spec
        ids_used.append(node_id)

    if not ids_used:
        # make this error maximally informative
        discovered = discover_ids(base_dir, prefixes=prefixes)
        raise RuntimeError(
            "No valid IDs processed.\n"
            f"- IDs requested: {id_list}\n"
            f"- IDs discoverable with prefixes {list(prefixes)}: {discovered}\n"
            "Likely cause: wrong --folder or wrong file prefix pattern."
        )

    avg = accum / float(len(ids_used))
    meta = {
        "num_ids": int(len(ids_used)),
        "ids_used": ids_used,
        "spectrum_type": spectrum_type,
    }
    return np.asarray(k_ref), np.asarray(avg), meta


def save_spectrum_txt(k, spec, out_file: Path, header: str):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((k, spec))
    np.savetxt(out_file, data, fmt="%.10e", header=header)
    print(f"Saved: {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Compute 1D isotropic spectra from vorticity NetCDF files.")
    p.add_argument("--folder", type=str, required=True)
    p.add_argument("--base-dir", type=str, default="/storage/home/hcoda1/5/mreynoso6/scratch/data")
    p.add_argument(
        "--pattern-prefix",
        type=str,
        nargs="*",
        default=["data_comp_id", "data_id"],
        help="Filename prefixes to try (default: data_comp_id data_id)",
    )

    p.add_argument("--spectrum-type", type=str, default="energy", choices=["energy", "enstrophy"])
    p.add_argument("--start-snapshot", type=int, default=0)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--node-id", type=int, default=None)
    g.add_argument("--all-ids", action="store_true")
    g.add_argument("--ids", type=int, nargs="+", default=None)

    p.add_argument(
        "--out-dir",
        type=str,
        default="/storage/home/hcoda1/4/mugliotti3/r-rg119-0/Paper2/Spectra",
    )
    p.add_argument("--out-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir) / args.folder
    if not base_dir.exists():
        raise FileNotFoundError(f"Folder not found: {base_dir}")

    prefixes = tuple(args.pattern_prefix)

    out_dir = Path(args.out_dir)

    if args.all_ids:
        id_list = None  # auto-discover inside
        k, spec, meta = compute_average_spectrum_all_ids(
            base_dir=base_dir,
            id_list=id_list,
            spectrum_type=args.spectrum_type,
            start_snapshot=args.start_snapshot,
            prefixes=prefixes,
        )
        out_name = args.out_name or f"{args.folder}_ALL_{args.spectrum_type}_spectrum.txt"
        out_file = out_dir / out_name
        header = (
            f"k   spectrum   (avg over {meta['num_ids']} ids; "
            f"spectrum_type={args.spectrum_type}; folder={args.folder}; ids_used={meta['ids_used']})"
        )
        save_spectrum_txt(k, spec, out_file, header)

    elif args.ids is not None:
        k, spec, meta = compute_average_spectrum_all_ids(
            base_dir=base_dir,
            id_list=args.ids,
            spectrum_type=args.spectrum_type,
            start_snapshot=args.start_snapshot,
            prefixes=prefixes,
        )
        out_name = args.out_name or f"{args.folder}_IDS_{args.spectrum_type}_spectrum.txt"
        out_file = out_dir / out_name
        header = (
            f"k   spectrum   (avg over {meta['num_ids']} ids; "
            f"spectrum_type={args.spectrum_type}; folder={args.folder}; ids_used={meta['ids_used']})"
        )
        save_spectrum_txt(k, spec, out_file, header)

    else:
        k, spec, meta = compute_average_spectrum_one_id(
            base_dir=base_dir,
            node_id=args.node_id,
            spectrum_type=args.spectrum_type,
            start_snapshot=args.start_snapshot,
            prefixes=prefixes,
            verbose=True,
        )
        out_name = args.out_name or f"{args.folder}_id{args.node_id:03d}_{args.spectrum_type}_spectrum.txt"
        out_file = out_dir / out_name
        header = (
            f"k   spectrum   (ID={meta['node_id']:03d}, num_snapshots={meta['num_snapshots']}, "
            f"spectrum_type={args.spectrum_type}, folder={args.folder})"
        )
        save_spectrum_txt(k, spec, out_file, header)


if __name__ == "__main__":
    main()
