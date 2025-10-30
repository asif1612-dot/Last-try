import os, math, json
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
TIF_PATH = "/workspaces/Last-try/data/assignment_ndvi_s2_2024_growing_season.tif"
OUT_DIR  = "/workspaces/Last-try/outputs"
MAP_DIR  = os.path.join(OUT_DIR, "task08_map_pack")
os.makedirs(MAP_DIR, exist_ok=True)

# Geometrical probability grid (approx cell size in pixels)
# If Sentinel-2 at 10 m, 25x25 px ~ 250 m cell
CELL_PX   = 25                 # grid size (square cells, in pixels)
T_THRESH  = 0.50               # crisp vegetation threshold for "NDVI >= t"
# Fuzzy membership μ_veg: 0 at <=A, 1 at >=B, linear between
FUZZY_A, FUZZY_B = 0.20, 0.70

# CLT sampling for uncertainty on mean NDVI
RANDOM_SEED = 7
MEAN_SAMPLE_N = 1000           # total random samples to estimate mean & CI
CLT_SIZES = [10, 30, 100]      # sample sizes for the sample-mean overlay

# ----------------------------
# LOAD NDVI
# ----------------------------
assert os.path.exists(TIF_PATH), f"Not found: {TIF_PATH}"
with rio.open(TIF_PATH) as src:
    ndvi = src.read(1).astype("float32")
    nodata = src.nodata
    profile = src.profile.copy()

if nodata is not None:
    ndvi = np.where(ndvi == nodata, np.nan, ndvi)

valid = np.isfinite(ndvi)
vals  = ndvi[valid]
assert vals.size > 0, "No valid NDVI pixels."

H, W = ndvi.shape

# ----------------------------
# GEOMETRICAL PROBABILITY per cell
# ----------------------------
def block_iter(h, w, cell):
    for r0 in range(0, h, cell):
        for c0 in range(0, w, cell):
            r1 = min(h, r0 + cell)
            c1 = min(w, c0 + cell)
            yield r0, r1, c0, c1

grid_records = []
geom_map = np.full_like(ndvi, np.nan, dtype="float32")

for r0, r1, c0, c1 in block_iter(H, W, CELL_PX):
    block = ndvi[r0:r1, c0:c1]
    m = np.isfinite(block)
    n_valid = int(m.sum())
    if n_valid == 0:
        frac = np.nan
    else:
        frac = float((block[m] >= T_THRESH).sum() / n_valid)

    grid_records.append({
        "r0": r0, "r1": r1, "c0": c0, "c1": c1,
        "valid": n_valid,
        "p_ge_t": frac
    })
    if n_valid > 0:
        geom_map[r0:r1, c0:c1] = frac

df_geom = pd.DataFrame(grid_records)
df_geom.to_csv(os.path.join(OUT_DIR, "task08_geom_cell_probs.csv"), index=False)

# ----------------------------
# FUZZY MEMBERSHIP map μ_veg
# ----------------------------
def mu_veg(x, a=FUZZY_A, b=FUZZY_B):
    # piecewise linear 0..1
    return np.where(~np.isfinite(x), np.nan,
            np.where(x <= a, 0.0,
            np.where(x >= b, 1.0,
            (x - a) / max(1e-6, (b - a)) )))

fuzzy_map = mu_veg(ndvi, FUZZY_A, FUZZY_B)
fuzzy_summary = {
    "mu_mean": float(np.nanmean(fuzzy_map)),
    "mu_std": float(np.nanstd(fuzzy_map)),
    "mu_p05": float(np.nanpercentile(fuzzy_map, 5)),
    "mu_p50": float(np.nanpercentile(fuzzy_map, 50)),
    "mu_p95": float(np.nanpercentile(fuzzy_map, 95)),
}
pd.DataFrame([fuzzy_summary]).to_csv(os.path.join(OUT_DIR, "task08_fuzzy_summary.csv"), index=False)

# ----------------------------
# DISTRIBUTION SNAPSHOT (Task 5–6 context)
# ----------------------------
hist_counts, hist_edges = np.histogram(vals, bins=60, range=(-0.2, 0.9))
cdf_x = np.sort(vals)
cdf_y = np.linspace(0, 1, cdf_x.size, endpoint=True)

pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts})\
  .to_csv(os.path.join(OUT_DIR, "task08_hist_counts.csv"), index=False)
pd.DataFrame({"ndvi": cdf_x, "cdf": cdf_y}).to_csv(os.path.join(OUT_DIR, "task08_empirical_cdf.csv"), index=False)

# ----------------------------
# CLT insight for uncertainty on mean NDVI
# ----------------------------
rng = np.random.default_rng(RANDOM_SEED)
draws = rng.choice(vals, size=MEAN_SAMPLE_N, replace=True)
hat_mu = float(np.mean(draws))
hat_sigma = float(np.std(draws, ddof=1))

clt_table = []
for n in CLT_SIZES:
    se = hat_sigma / math.sqrt(n)
    # approx 95% CI via normal
    ci_lo = hat_mu - 1.96 * se
    ci_hi = hat_mu + 1.96 * se
    clt_table.append({"n": n, "mean_hat": hat_mu, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi})
pd.DataFrame(clt_table).to_csv(os.path.join(OUT_DIR, "task08_mean_uncertainty.csv"), index=False)

# ----------------------------
# A SIMPLE PRIORITY SCORE (transparent!)
# Combine crisp area-fraction (geom_map) and fuzzy mean in each cell
# ----------------------------
priority_map = np.full_like(ndvi, np.nan, dtype="float32")
# compute cell-level fuzzy mean, then score = 0.6*P(NDVI>=t) + 0.4*mean(μ_veg)
cell_scores = []

for r0, r1, c0, c1 in block_iter(H, W, CELL_PX):
    gm_block = geom_map[r0:r1, c0:c1]
    mu_block = fuzzy_map[r0:r1, c0:c1]
    vmask = np.isfinite(gm_block) & np.isfinite(mu_block)
    if vmask.sum() == 0:
        score = np.nan
    else:
        score = 0.6 * float(np.nanmean(gm_block)) + 0.4 * float(np.nanmean(mu_block))
    cell_scores.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "score": score})
    if np.isfinite(score):
        priority_map[r0:r1, c0:c1] = score

df_scores = pd.DataFrame(cell_scores)
df_scores.to_csv(os.path.join(OUT_DIR, "task08_priority_scores.csv"), index=False)

# ----------------------------
# PLOTTING: 1) Geometrical probability map
# ----------------------------
plt.figure(figsize=(7.5,5.5))
plt.imshow(geom_map, vmin=0, vmax=1)
plt.colorbar(label=f"P(NDVI ≥ {T_THRESH})")
plt.title("Geometrical Probability by Cell")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "geom_prob_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2) Fuzzy μ_veg map
plt.figure(figsize=(7.5,5.5))
plt.imshow(fuzzy_map, vmin=0, vmax=1)
plt.colorbar(label="μ_veg (0..1)")
plt.title(f"Fuzzy Membership (A={FUZZY_A}, B={FUZZY_B})")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "fuzzy_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3) Priority map
plt.figure(figsize=(7.5,5.5))
plt.imshow(priority_map, vmin=0, vmax=1)
plt.colorbar(label="Priority score (0..1)")
plt.title("Priority Map (0.6·P + 0.4·μ)")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "priority_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4) Distribution snapshot + CLT CIs (small panel)
fig, ax = plt.subplots(figsize=(7.5,4.5))
ax.hist(vals, bins=60, alpha=0.7)
ax.set_title("NDVI Histogram (snapshot)")
ax.set_xlabel("NDVI"); ax.set_ylabel("Frequency")
txt = "\n".join([f"n={row['n']}: mean≈{hat_mu:.3f}, 95% CI [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
                  for row in clt_table])
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
plt.savefig(os.path.join(MAP_DIR, "distribution_and_CLT.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# Console summary
# ----------------------------
print("\n=== TASK 8 SYNTHESIS ===")
print(f"Raster shape: {H}x{W}, valid={vals.size}")
print(f"Fuzzy μ summary: {json.dumps(fuzzy_summary, indent=2)}")
print("Mean uncertainty (CLT overlay):")
for r in clt_table:
    print(f" n={r['n']:>3d}  mean≈{r['mean_hat']:.3f}  95%CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
print("\nWrote:")
for p in [
    os.path.join(MAP_DIR, "geom_prob_map.png"),
    os.path.join(MAP_DIR, "fuzzy_map.png"),
    os.path.join(MAP_DIR, "priority_map.png"),
    os.path.join(MAP_DIR, "distribution_and_CLT.png"),
]:
    print(" -", p)
print(" CSVs: task08_geom_cell_probs.csv, task08_fuzzy_summary.csv, task08_mean_uncertainty.csv, task08_priority_scores.csv, task08_hist_counts.csv, task08_empirical_cdf.csv")
