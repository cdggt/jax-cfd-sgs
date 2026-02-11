#!/usr/bin/env python3
import jax
jax.config.update("jax_enable_x64", True)

import argparse
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import xarray as xr
from tqdm import tqdm


# =========================================================
# Gaussian filter (Fourier, periodic domain)
# =========================================================
def gaussian_filter_2d_fft_jax(w, delta, L=2.0 * np.pi):
    Nx, Ny = w.shape
    dx = L / Nx
    dy = L / Ny

    kx = 2.0 * np.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * jnp.fft.fftfreq(Ny, d=dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    Ghat = jnp.exp(-K2 * delta**2 / 24.0)
    w_hat = jnp.fft.fftn(w)
    return jnp.fft.ifftn(Ghat * w_hat).real


def snapshot_index(p: Path) -> int:
    # works for data_id001_00080.nc
    return int(p.stem.split("_")[-1])


def find_files(data_dir: Path, node_str: str):
    # support both naming conventions
    files = sorted(data_dir.glob(f"data_comp_id{node_str}_*.nc"))
    if not files:
        files = sorted(data_dir.glob(f"data_id{node_str}_*.nc"))
    return files


# =========================================================
# Quantiles + central probability intervals from a binned PDF
# =========================================================
def quantiles_and_central_intervals_from_pdf(
    bin_edges: np.ndarray,
    pdf: np.ndarray,
    masses=(0.50, 0.68, 0.90, 0.95, 0.99),
    q_levels=(0.005, 0.025, 0.16, 0.50, 0.84, 0.975, 0.995),
):
    """
    Given:
      bin_edges: (bins+1,)
      pdf:       (bins,) defined on [bin_edges[i], bin_edges[i+1]]
    Returns:
      q_levels, omega_q: quantiles
      masses, omega_lo, omega_hi: central intervals containing 'masses' probability

    Assumes pdf is normalized on the provided finite range:
      sum(pdf)*dω ≈ 1
    """
    pdf = np.asarray(pdf, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)

    if pdf.ndim != 1 or bin_edges.ndim != 1 or bin_edges.size != pdf.size + 1:
        raise ValueError("Shape mismatch: need bin_edges (bins+1) and pdf (bins)")

    dω = bin_edges[1] - bin_edges[0]
    prob = pdf * dω  # per-bin probability mass

    cdf_right = np.cumsum(prob)
    cdf_right = np.clip(cdf_right, 0.0, 1.0)
    if cdf_right[-1] > 0:
        cdf_right = cdf_right / cdf_right[-1]
    cdf_right[-1] = 1.0

    x_right = bin_edges[1:]  # same length as cdf_right

    q_levels = np.asarray(q_levels, dtype=np.float64)
    q_levels = np.clip(q_levels, 0.0, 1.0)
    omega_q = np.interp(q_levels, cdf_right, x_right)

    masses = np.asarray(masses, dtype=np.float64)
    masses = np.clip(masses, 0.0, 1.0)
    lo_ps = (1.0 - masses) / 2.0
    hi_ps = 1.0 - lo_ps

    omega_lo = np.interp(lo_ps, cdf_right, x_right)
    omega_hi = np.interp(hi_ps, cdf_right, x_right)

    return q_levels, omega_q, masses, omega_lo, omega_hi


# =========================================================
# PDF for ONE ID: fixed omega range [-omega_max, omega_max]
# =========================================================
def compute_fdns_pdf_for_id(args, node_id: int):
    node_str = f"{node_id:03d}"
    data_dir = Path(args.base_dir) / args.folder

    files = find_files(data_dir, node_str)
    files = [f for f in files if snapshot_index(f) >= args.snapshot_min]
    if not files:
        raise RuntimeError(f"No files for folder={args.folder} id={node_str} snapshot>={args.snapshot_min}")

    # Fixed bins for ALL IDs/models by construction
    bin_edges = np.linspace(-args.omega_max, args.omega_max, args.bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(args.bins, dtype=np.int64)

    # diagnostics
    out_lo = 0
    out_hi = 0
    total_pts = 0

    # metadata (optional)
    sim_nu = np.nan
    sim_res = -1
    try:
        with xr.open_dataset(files[0]) as ds0:
            sim_nu = ds0.attrs.get("nu", np.nan)
            sim_res = ds0.sizes.get("x", ds0["w"].shape[-2])
    except Exception:
        pass

    # (optional) sigma of FILTERED field, just for reporting
    # do it on up to 20 files (cheap-ish)
    if args.report_sigma:
        if len(files) > 20:
            idx = np.linspace(0, len(files) - 1, 20, dtype=int)
            subset = [files[i] for i in idx]
        else:
            subset = files

        sum_w = 0.0
        sum_w2 = 0.0
        count = 0
        for fpath in tqdm(subset, desc=f"Sigma (filtered) ID {node_str}", leave=False):
            with xr.open_dataset(fpath) as ds:
                w = jnp.asarray(ds["w"].values)
                w_f = gaussian_filter_2d_fft_jax(w, args.delta).reshape(-1)
                sum_w += float(jnp.sum(w_f))
                sum_w2 += float(jnp.sum(w_f * w_f))
                count += w_f.size
        mean = sum_w / count
        var = (sum_w2 / count) - mean**2
        sigma_filt = float(np.sqrt(max(var, 0.0)))
    else:
        sigma_filt = np.nan

    # Histogram pass
    for fpath in tqdm(files, desc=f"Binning (filtered) ID {node_str}", leave=False):
        with xr.open_dataset(fpath) as ds:
            w = jnp.asarray(ds["w"].values)
            w_f = gaussian_filter_2d_fft_jax(w, args.delta).reshape(-1)
            w_np = np.asarray(w_f)

        total_pts += w_np.size
        out_lo += int(np.sum(w_np < -args.omega_max))
        out_hi += int(np.sum(w_np >  args.omega_max))

        hist, _ = np.histogram(w_np, bins=bin_edges)
        counts += hist

    total_in_range = int(counts.sum())
    dω = bin_edges[1] - bin_edges[0]

    # normalize by points IN RANGE so integral over [-omega_max, omega_max] is 1
    pdf = counts / (total_in_range * dω) if total_in_range > 0 else np.zeros_like(counts, float)

    meta = {
        "folder": args.folder,
        "node_id": node_id,
        "n_files": len(files),
        "snapshot_min": args.snapshot_min,
        "bins": args.bins,
        "omega_max": args.omega_max,
        "delta": args.delta,
        "resolution": sim_res,
        "viscosity": float(sim_nu) if np.isfinite(sim_nu) else np.nan,
        "sigma_filtered_est": sigma_filt,
        "frac_outside": float((out_lo + out_hi) / total_pts) if total_pts else np.nan,
        "out_lo": out_lo,
        "out_hi": out_hi,
        "total_pts": total_pts,
        "total_in_range": total_in_range,
    }

    return bin_centers, bin_edges, pdf, meta


# =========================================================
# Save txt in your robust format (now with uncertainty + percentiles)
# =========================================================
def save_pdf_txt(
    out_path: Path,
    bin_centers: np.ndarray,
    pdf_mean: np.ndarray,
    meta: dict,
    pdf_std: np.ndarray | None = None,
    pdf_sem: np.ndarray | None = None,
    percentile_levels: np.ndarray | None = None,
    omega_percentiles: np.ndarray | None = None,
    central_mass_levels: np.ndarray | None = None,
    central_interval_lo: np.ndarray | None = None,
    central_interval_hi: np.ndarray | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# FDNS filtered vorticity PDF on fixed omega-range\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write("\n")

        def dump(name, arr):
            f.write(f"=== {name} (len={len(arr)}) ===\n")
            f.write(" ".join(f"{x:.8e}" for x in np.asarray(arr)) + "\n\n")

        dump("bin_centers", bin_centers)
        dump("pdf_mean", pdf_mean)

        if pdf_std is not None:
            dump("pdf_std", pdf_std)
        if pdf_sem is not None:
            dump("pdf_sem", pdf_sem)

        # percentiles (quantiles)
        if percentile_levels is not None and omega_percentiles is not None:
            dump("percentile_levels", percentile_levels)
            dump("omega_percentiles", omega_percentiles)

        # central mass intervals (e.g. 68% in [lo, hi])
        if (central_mass_levels is not None and
            central_interval_lo is not None and
            central_interval_hi is not None):
            dump("central_mass_levels", central_mass_levels)
            dump("central_interval_lo", central_interval_lo)
            dump("central_interval_hi", central_interval_hi)

    print(f"[saved] {out_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    p = argparse.ArgumentParser(description="FDNS filtered PDFs on fixed omega range [-omega-max, omega-max]")
    p.add_argument("--base-dir", default="/storage/home/hcoda1/5/mreynoso6/scratch/data")
    p.add_argument("--folder", required=True)  # single folder here (dns_forced_*)
    p.add_argument("--ids", type=int, nargs="+", required=True)
    p.add_argument("--bins", type=int, default=1000)
    p.add_argument("--omega-max", type=float, default=4.0)
    p.add_argument("--snapshot-min", type=int, default=80)
    p.add_argument("--delta", type=float, default=6.0 * np.pi / 256.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-name", default="FDNS_RAW_PDF_omegaMax4.txt")
    p.add_argument(
        "--report-sigma",
        action="store_true",
        help="Optional: estimate sigma of filtered field for diagnostics (does NOT affect binning).",
    )

    # percentile/band defaults that "make sense for PDFs"
    p.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0.5, 2.5, 16.0, 50.0, 84.0, 97.5, 99.5],
        help="Percentiles (in %) to save from the PDF CDF (e.g. 0.5 2.5 16 50 84 97.5 99.5).",
    )
    p.add_argument(
        "--central-bands",
        type=float,
        nargs="+",
        default=[50.0, 68.0, 90.0, 95.0, 99.0],
        help="Central probability mass bands (in %) to save as [lo, hi] intervals.",
    )

    args = p.parse_args()

    per_id = []
    frac_out = []
    sigma_est = []

    print(f"\n=== FDNS folder: {args.folder} ===")
    print(f"Fixed omega-range: [-{args.omega_max}, +{args.omega_max}]  bins={args.bins}")
    print(f"Filter delta = {args.delta}   snapshot_min = {args.snapshot_min}")

    bin_centers_ref = None
    bin_edges_ref = None

    for nid in args.ids:
        bin_centers, bin_edges, pdf, meta = compute_fdns_pdf_for_id(args, nid)

        if bin_centers_ref is None:
            bin_centers_ref = bin_centers
            bin_edges_ref = bin_edges
        else:
            if bin_centers.shape != bin_centers_ref.shape or not np.allclose(bin_centers, bin_centers_ref):
                raise RuntimeError("bin_centers mismatch across IDs (should not happen with fixed bins).")
            if bin_edges.shape != bin_edges_ref.shape or not np.allclose(bin_edges, bin_edges_ref):
                raise RuntimeError("bin_edges mismatch across IDs (should not happen with fixed bins).")

        per_id.append(pdf)
        frac_out.append(meta["frac_outside"])
        sigma_est.append(meta["sigma_filtered_est"])

    per_id = np.stack(per_id, axis=0)  # (n_ids, bins)
    n_ids = per_id.shape[0]

    pdf_mean = per_id.mean(axis=0)

    # --- uncertainties ACROSS IDs (ensemble uncertainty)
    if n_ids >= 2:
        pdf_std = per_id.std(axis=0, ddof=1)   # sample std across IDs
    else:
        pdf_std = np.zeros_like(pdf_mean)
    pdf_sem = pdf_std / np.sqrt(n_ids)

    # --- percentiles + central mass bands computed from pdf_mean (on the truncated range)
    q_levels = np.array(args.percentiles, dtype=float) / 100.0
    masses = np.array(args.central_bands, dtype=float) / 100.0

    percentile_levels, omega_percentiles, central_mass_levels, central_lo, central_hi = (
        quantiles_and_central_intervals_from_pdf(
            bin_edges_ref,
            pdf_mean,
            masses=tuple(masses.tolist()),
            q_levels=tuple(q_levels.tolist()),
        )
    )

    # print bands in a human-readable way
    for m, lo, hi in zip(central_mass_levels, central_lo, central_hi):
        print(f"Central {100*m:5.1f}% interval: [{lo: .4g}, {hi: .4g}]")

    meta_out = {
        "folder": args.folder,
        "ids": args.ids,
        "n_ids": int(n_ids),
        "bins": args.bins,
        "omega_max": args.omega_max,
        "snapshot_min": args.snapshot_min,
        "delta": args.delta,
        "frac_outside_mean": float(np.mean(frac_out)),
        "uncertainty_across": "ids",
        "pdf_std_ddof": 1,
        "pdf_sem_definition": "std_across_ids/sqrt(n_ids)",
        "percentiles_percent": args.percentiles,
        "central_bands_percent": args.central_bands,
        "note": "Percentiles/bands computed from pdf_mean normalized on [-omega_max, omega_max].",
    }
    if args.report_sigma:
        meta_out["sigma_filtered_est_mean"] = float(np.nanmean(sigma_est))

    out_dir = Path(args.out_dir)
    out_path = out_dir / args.out_name
    save_pdf_txt(
        out_path,
        bin_centers_ref,
        pdf_mean,
        meta_out,
        pdf_std=pdf_std,
        pdf_sem=pdf_sem,
        percentile_levels=percentile_levels,
        omega_percentiles=omega_percentiles,
        central_mass_levels=central_mass_levels,
        central_interval_lo=central_lo,
        central_interval_hi=central_hi,
    )

    print(f"\navg frac outside range = {meta_out['frac_outside_mean']:.3e}")
    print("If that’s not tiny, increase --omega-max (or you’re chopping tails).")


if __name__ == "__main__":
    main()
