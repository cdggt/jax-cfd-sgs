#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================================================
# Compute RAW vorticity PDF p(ω) on a FIXED ω-range for ONE ID
# =========================================================
def compute_pdf_fixed_range_for_id(base_dir: Path, folder: str, node_id: int,
                                  bins: int, omega_max: float):
    node_str = f"{node_id:03d}"
    data_dir = base_dir / folder
    file_list = sorted(data_dir.glob(f"data_comp_id{node_str}_*.nc"))
    if not file_list:
        raise RuntimeError(f"No files found: {data_dir}/data_comp_id{node_str}_*.nc")

    bin_edges = np.linspace(-omega_max, omega_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(bins, dtype=np.int64)

    out_lo = 0
    out_hi = 0
    total_pts = 0

    sim_nu = np.nan
    sim_res = -1
    try:
        with xr.open_dataset(file_list[0]) as ds0:
            sim_nu = ds0.attrs.get("nu", np.nan)
            sim_res = ds0.sizes.get("x", ds0["w"].shape[-2])
    except Exception:
        pass

    for fpath in tqdm(file_list, desc=f"Binning ID {node_str}", leave=False):
        with xr.open_dataset(fpath) as ds:
            w = np.asarray(ds["w"].values).reshape(-1)
            total_pts += w.size
            out_lo += int(np.sum(w < -omega_max))
            out_hi += int(np.sum(w >  omega_max))

            hist, _ = np.histogram(w, bins=bin_edges)
            counts += hist

    total_in_range = counts.sum()
    dω = bin_edges[1] - bin_edges[0]
    pdf = counts / (total_in_range * dω) if total_in_range > 0 else np.zeros_like(counts)

    return {
        "bin_centers": bin_centers,
        "pdf": pdf,
        "meta": {
            "node_id": node_id,
            "resolution": sim_res,
            "viscosity": float(sim_nu) if np.isfinite(sim_nu) else np.nan,
            "n_files": len(file_list),
            "omega_max": omega_max,
            "bins": bins,
            "frac_outside": float((out_lo + out_hi) / total_pts) if total_pts > 0 else np.nan,
        },
    }


# =========================================================
# Save TXT (mean + uncertainties)
# =========================================================
def save_pdf_txt(out_path: Path, meta: dict, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# RAW vorticity PDF p(omega) on fixed omega-range\n")
        for k, v in meta.items():
            f.write(f"# {k} = {v}\n")
        f.write("\n")

        def dump(name, arr):
            arr = np.asarray(arr)
            f.write(f"=== {name} (len={len(arr)}) ===\n")
            f.write(" ".join(f"{x:.8e}" for x in arr) + "\n\n")

        for name, arr in arrays.items():
            dump(name, arr)

    print(f"[saved] {out_path}")


# =========================================================
# Loader
# =========================================================
def load_pdf_txt(path: Path):
    data = {}
    key, buf = None, []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s:
                if key and buf:
                    data[key] = np.fromstring(" ".join(buf), sep=" ")
                key, buf = None, []
                continue
            if s.startswith("==="):
                if key and buf:
                    data[key] = np.fromstring(" ".join(buf), sep=" ")
                key = s.strip("= ").split(" (")[0].strip()
                buf = []
            elif s.startswith("#"):
                continue
            else:
                buf.append(s)
    if key and buf:
        data[key] = np.fromstring(" ".join(buf), sep=" ")
    return data


# =========================================================
# MAIN
# =========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="/storage/home/hcoda1/5/mreynoso6/scratch/data")
    p.add_argument("--folders", nargs="+", required=True)
    p.add_argument("--ids", type=int, nargs="+", required=True)
    p.add_argument("--bins", type=int, default=1000)
    p.add_argument("--omega-max", type=float, default=4.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-band", action="store_true",
                   help="Plot mean ± SEM band")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    # ---------- Compute ----------
    for folder in args.folders:
        per_id = []
        fracs = []

        print(f"\n=== MODEL: {folder} ===")
        for node_id in args.ids:
            res = compute_pdf_fixed_range_for_id(
                base_dir, folder, node_id, args.bins, args.omega_max
            )
            per_id.append(res["pdf"])
            fracs.append(res["meta"]["frac_outside"])

        bin_edges = np.linspace(-args.omega_max, args.omega_max, args.bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        pdf_stack = np.stack(per_id, axis=0)
        pdf_mean = pdf_stack.mean(axis=0)

        n_ids = pdf_stack.shape[0]
        pdf_std = pdf_stack.std(axis=0, ddof=1) if n_ids > 1 else np.zeros_like(pdf_mean)
        pdf_sem = pdf_std / np.sqrt(max(n_ids, 1))

        meta = {
            "model_folder": folder,
            "ids": args.ids,
            "n_ids": int(n_ids),
            "bins": args.bins,
            "omega_max": args.omega_max,
            "frac_outside_mean": float(np.mean(fracs)),
            "uncertainty": "std & sem across IDs (bin-wise)",
        }

        out_path = out_dir / f"{folder}_RAW_PDF_omegaMax{args.omega_max:g}.txt"
        save_pdf_txt(
            out_path,
            meta,
            bin_centers=bin_centers,
            pdf_mean=pdf_mean,
            pdf_std=pdf_std,
            pdf_sem=pdf_sem,
        )

    # ---------- Plot ----------
    if args.plot:
        plt.figure(figsize=(9, 6))
        for f in sorted(out_dir.glob(f"*_RAW_PDF_omegaMax{args.omega_max:g}.txt")):
            d = load_pdf_txt(f)
            w = d["bin_centers"]
            pmean = d["pdf_mean"]
            i = np.argsort(w)

            plt.semilogy(w[i], pmean[i], label=f.stem)

            if args.plot_band and "pdf_sem" in d:
                sem = d["pdf_sem"][i]
                lo = np.clip(pmean[i] - sem, 1e-300, None)
                hi = np.clip(pmean[i] + sem, 1e-300, None)
                plt.fill_between(w[i], lo, hi, alpha=0.15)

        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$p(\omega)$")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
