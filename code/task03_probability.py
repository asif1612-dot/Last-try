"""
Task 03 — Probability with NDVI
Creates the following outputs (in outputs/):
 - task03_p_events.csv
 - task03_event_prob_plot.png
 - task03_p_events_by_region.csv
 - task03_region_bar.png
 - task03_running_mean_samples.csv
 - task03_running_mean.png

Author: Asif (template by assistant)
"""
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
TIF_PATH = "data/assignment_ndvi_s2_2024_growing_season.tif"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# thresholds to evaluate (you can change these)
thresholds = np.linspace(-0.2, 0.8, 21)  # from -0.2 to 0.8 in 0.05 steps

# region grid (rows x cols)
REG_ROWS = 2
REG_COLS = 2

# sampling for LLN/CLT demonstration
SAMPLE_SIZES = [10, 50, 100, 500, 1000, 5000]
TRIALS_PER_SIZE = 500  # number of repetitions per sample size
RANDOM_SEED = 42

# ---------- Helpers ----------
def load_ndvi(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        crs = src.crs
        transform = src.transform
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, crs, transform

def fraction_geq(values, thresh):
    # fraction of valid pixels >= thresh
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.nan, 0, 0
    total = valid.sum()
    count = np.count_nonzero(values[valid] >= thresh)
    return count / total, int(count), int(total)

# ---------- Main ----------
def main():
    # load
    assert os.path.exists(TIF_PATH), f"NDVI file not found: {TIF_PATH}"
    ndvi, crs, transform = load_ndvi(TIF_PATH)
    flat = ndvi.flatten()
    valid_mask = np.isfinite(flat)
    vals = flat[valid_mask]

    print(f"Loaded NDVI: shape={ndvi.shape}, CRS={crs}, valid_pixels={vals.size}")

    # 1) P(NDVI >= t) for whole image
    rows = []
    for t in thresholds:
        p, count, total = fraction_geq(vals, t)
        rows.append({"threshold": float(t), "p_geq": float(p) if not np.isnan(p) else np.nan,
                     "count_geq": count, "total_valid": total})
    df_events = pd.DataFrame(rows)
    df_events.to_csv(os.path.join(OUT_DIR, "task03_p_events.csv"), index=False)
    print("Wrote:", os.path.join(OUT_DIR, "task03_p_events.csv"))

    # plot event probability curve
    plt.figure(figsize=(6,4))
    plt.plot(df_events["threshold"], df_events["p_geq"], marker="o")
    plt.xlabel("NDVI threshold (t)")
    plt.ylabel("P(NDVI ≥ t)")
    plt.title("Event probability vs threshold (whole image)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task03_event_prob_plot.png"), dpi=200)
    plt.close()
    print("Wrote:", os.path.join(OUT_DIR, "task03_event_prob_plot.png"))

    # 2) Regional comparisons: divide image into REG_ROWS x REG_COLS grid
    h, w = ndvi.shape
    r_heights = np.linspace(0, h, REG_ROWS+1, dtype=int)
    c_widths = np.linspace(0, w, REG_COLS+1, dtype=int)

    region_rows = []
    for i in range(REG_ROWS):
        for j in range(REG_COLS):
            r0, r1 = r_heights[i], r_heights[i+1]
            c0, c1 = c_widths[j], c_widths[j+1]
            sub = ndvi[r0:r1, c0:c1].flatten()
            valid = np.isfinite(sub)
            total = int(valid.sum())
            # compute probability for a chosen threshold set (e.g., t = 0.3)
            # we'll compute probabilities across the thresholds and also report P>=0.3
            p_table = []
            for t in thresholds:
                p, count, _ = fraction_geq(sub[valid], t)
                p_table.append(p)
            region_label = f"R{i+1}C{j+1}"
            region_rows.append({
                "region": region_label,
                "row_idx": i,
                "col_idx": j,
                "total_valid": total,
                "p_geq_0.3": float(fraction_geq(sub[valid], 0.3)[0]) if total>0 else np.nan
            })
    df_regions = pd.DataFrame(region_rows)
    df_regions.to_csv(os.path.join(OUT_DIR, "task03_p_events_by_region.csv"), index=False)
    print("Wrote:", os.path.join(OUT_DIR, "task03_p_events_by_region.csv"))

    # bar plot of P(NDVI>=0.3) by region
    plt.figure(figsize=(6,4))
    plt.bar(df_regions["region"], df_regions["p_geq_0.3"], color="tab:green")
    plt.xlabel("Region")
    plt.ylabel("P(NDVI ≥ 0.3)")
    plt.title("Regional comparison (P >= 0.3)")
    plt.ylim(0,1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task03_region_bar.png"), dpi=200)
    plt.close()
    print("Wrote:", os.path.join(OUT_DIR, "task03_region_bar.png"))

    # 3) Running mean / LLN & CLT demonstration (sampling)
    rng = np.random.default_rng(RANDOM_SEED)
    n_vals = vals.size
    if n_vals == 0:
        raise RuntimeError("No valid NDVI pixels found.")

    sample_records = []
    for n in SAMPLE_SIZES:
        n_eff = min(n, n_vals)
        for trial in range(TRIALS_PER_SIZE):
            # sample without replacement if n < total, else with replacement
            if n_eff < n_vals:
                sample = rng.choice(vals, size=n_eff, replace=False)
            else:
                sample = rng.choice(vals, size=n_eff, replace=True)
            sample_mean = float(np.mean(sample))
            sample_std = float(np.std(sample, ddof=1)) if n_eff>1 else 0.0
            sample_records.append({
                "sample_size": int(n_eff),
                "trial": int(trial),
                "sample_mean": sample_mean,
                "sample_std": sample_std
            })
    df_samples = pd.DataFrame(sample_records)
    df_samples.to_csv(os.path.join(OUT_DIR, "task03_running_mean_samples.csv"), index=False)
    print("Wrote:", os.path.join(OUT_DIR, "task03_running_mean_samples.csv"))

    # plot running mean behaviour: show distribution of sample_means per sample_size (boxplot)
    plt.figure(figsize=(8,5))
    order = SAMPLE_SIZES
    df_plot = df_samples.copy()
    df_plot["sample_size"] = df_plot["sample_size"].astype(int)
    box_data = [df_plot[df_plot["sample_size"]==s]["sample_mean"].values for s in order]
    plt.boxplot(box_data, labels=order, showfliers=False)
    plt.xlabel("Sample size")
    plt.ylabel("Sample mean of NDVI")
    plt.title("Distribution of sample means (LLN demonstration)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task03_running_mean.png"), dpi=200)
    plt.close()
    print("Wrote:", os.path.join(OUT_DIR, "task03_running_mean.png"))

    # 3b) single-run running mean (optional): accumulate random draw and show convergence
    # take a single large random shuffle of indices and compute cumulative mean
    idxs = rng.permutation(n_vals)
    seq = vals[idxs]
    max_run = min(5000, n_vals)  # limit to 5000 for plotting speed
    cum_means = np.cumsum(seq[:max_run]) / (np.arange(max_run) + 1)
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(1, max_run+1), cum_means, lw=1)
    plt.hlines(np.mean(vals), 1, max_run, colors="red", linestyles="dashed", label="Population mean")
    plt.xscale('log')
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Cumulative mean of NDVI")
    plt.title("Running mean (single random draw) — convergence to population mean")
    plt.legend()
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task03_running_mean_single.png"), dpi=200)
    plt.close()
    print("Wrote:", os.path.join(OUT_DIR, "task03_running_mean_single.png"))

    print("\nTask 03 complete. Summary files created in", OUT_DIR)

if __name__ == "__main__":
    main()
