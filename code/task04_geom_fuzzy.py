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
REPORTS  = "/workspaces/Last-try/reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# Geometrical probability parameters
TARGET_CELL_M = 250      # target grid cell size in meters (approx)
THRESHOLD_T   = 0.50     # crisp threshold for "vegetated": NDVI >= t

# Fuzzy membership parameters: μ_veg(ndvi)
FUZZY_A = 0.20           # μ=0 at ndvi <= A
FUZZY_B = 0.70           # μ=1 at ndvi >= B

# ----------------------------
# Helpers
# ----------------------------
def meters_per_degree_lat(phi_rad: float) -> float:
    # Reasonable approximation
    return 111320.0

def meters_per_degree_lon(phi_rad: float) -> float:
    return 111320.0 * math.cos(phi_rad)

def pixel_size_meters(src):
    """
    Returns (px_w_m, px_h_m, lat_center_deg). Works for projected (meter) and geographic (degree) CRSs.
    """
    transform = src.transform
    h, w = src.height, src.width
    crs = src.crs

    px_w = abs(transform.a)
    px_h = abs(transform.e)

    # center latitude (for geographic CRS)
    cx = w / 2.0
    cy = h / 2.0
    # x = a*col + c ; y = e*row + f (assuming north-up, b=d=0)
    lat_center = transform.f + transform.e * cy

    if crs is not None and not crs.is_geographic:
        # projected in meters
        return px_w, px_h, lat_center
    else:
        # geographic degrees → meters using center latitude
        phi = math.radians(lat_center)
        mx = meters_per_degree_lon(phi)
        my = meters_per_degree_lat(phi)
        return px_w * mx, px_h * my, lat_center

def block_fraction_event(ndvi, cell_h, cell_w, thresh):
    """
    Compute per-block fraction of NDVI >= thresh (ignoring NaN).
    Returns (frac_map[nrows,ncols], valid_count_map).
    """
    H, W = ndvi.shape
    Hc = (H // cell_h) * cell_h
    Wc = (W // cell_w) * cell_w
    if Hc == 0 or Wc == 0:
        raise ValueError("Grid cell size too large for raster dimensions.")

    sub = ndvi[:Hc, :Wc]
    valid = np.isfinite(sub)
    event = (sub >= thresh) & valid

    nrows = Hc // cell_h
    ncols = Wc // cell_w

    # reshape to 4D blocks: (nrows, cell_h, ncols, cell_w)
    valid_counts = valid.reshape(nrows, cell_h, ncols, cell_w).sum(axis=(1,3)).astype(float)
    event_counts = event.reshape(nrows, cell_h, ncols, cell_w).sum(axis=(1,3)).astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        frac = event_counts / valid_counts
        frac[valid_counts == 0] = np.nan

    return frac, valid_counts

def fuzzy_membership(ndvi, a, b):
    """
    μ_veg(ndvi): 0 at <=a ; 1 at >=b ; linear in between.
    Returns μ with NaN where ndvi is NaN.
    """
    mu = np.full_like(ndvi, np.nan, dtype="float32")
    valid = np.isfinite(ndvi)
    x = ndvi[valid]

    mu_valid = np.zeros_like(x, dtype="float32")
    # in-between linear ramp
    mid = (x > a) & (x < b)
    mu_valid[mid] = (x[mid] - a) / (b - a)
    mu_valid[x >= b] = 1.0
    # x <= a already 0.0

    mu[valid] = mu_valid
    # clamp (safety)
    mu[valid] = np.clip(mu[valid], 0.0, 1.0)
    return mu

# ----------------------------
# Load NDVI
# ----------------------------
assert os.path.exists(TIF_PATH), f"File not found: {TIF_PATH}"
with rio.open(TIF_PATH) as src:
    profile = src.profile.copy()
    ndvi = src.read(1).astype("float32")
    nodata = src.nodata
    px_w_m, px_h_m, lat_c = pixel_size_meters(src)

if nodata is not None:
    ndvi = np.where(ndvi == nodata, np.nan, ndvi)

valid_mask = np.isfinite(ndvi)
assert valid_mask.any(), "No valid NDVI pixels found."

# Determine cell size in pixels to approximate TARGET_CELL_M
cell_w_px = max(1, int(round(TARGET_CELL_M / px_w_m)))
cell_h_px = max(1, int(round(TARGET_CELL_M / px_h_m)))

# ----------------------------
# Geometrical probability: per-cell P(NDVI >= t)
# ----------------------------
geom_frac, valid_counts = block_fraction_event(ndvi, cell_h_px, cell_w_px, THRESHOLD_T)
geom_stats = {
    "cell_size_target_m": TARGET_CELL_M,
    "pixel_size_m_x": px_w_m,
    "pixel_size_m_y": px_h_m,
    "approx_cell_w_px": cell_w_px,
    "approx_cell_h_px": cell_h_px,
    "mean_cell_prob": float(np.nanmean(geom_frac)),
    "median_cell_prob": float(np.nanmedian(geom_frac)),
    "min_cell_prob": float(np.nanmin(geom_frac)),
    "max_cell_prob": float(np.nanmax(geom_frac)),
    "threshold_t": THRESHOLD_T,
    "lat_center_deg": float(lat_c),
}
with open(os.path.join(OUT_DIR, "task04_geom_stats.json"), "w") as f:
    json.dump(geom_stats, f, indent=2)

plt.figure()
plt.imshow(geom_frac, vmin=0, vmax=1)
plt.colorbar(label=f"P(NDVI ≥ {THRESHOLD_T}) per ~{TARGET_CELL_M} m cell")
plt.title("Geometrical Probability Map")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR, "task04_geom_prob_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# Fuzzy membership: μ_veg(ndvi)
# ----------------------------
mu = fuzzy_membership(ndvi, FUZZY_A, FUZZY_B)
mu_stats = {
    "fuzzy_a": FUZZY_A,
    "fuzzy_b": FUZZY_B,
    "mean_mu": float(np.nanmean(mu)),
    "median_mu": float(np.nanmedian(mu)),
    "min_mu": float(np.nanmin(mu)),
    "max_mu": float(np.nanmax(mu)),
}
with open(os.path.join(OUT_DIR, "task04_fuzzy_stats.json"), "w") as f:
    json.dump(mu_stats, f, indent=2)

plt.figure()
plt.imshow(mu, vmin=0, vmax=1)
plt.colorbar(label="μ_veg (0–1)")
plt.title("Fuzzy Vegetation Membership Map")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR, "task04_fuzzy_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# Console summary
# ----------------------------
print("\n=== TASK 4 SUMMARY ===")
print("Geometrical probability (per-cell):")
for k, v in geom_stats.items():
    print(f"  {k}: {v}")
print("\nFuzzy membership:")
for k, v in mu_stats.items():
    print(f"  {k}: {v}")

print("\nWrote:")
print(f"  - {os.path.join(OUT_DIR, 'task04_geom_prob_map.png')}")
print(f"  - {os.path.join(OUT_DIR, 'task04_fuzzy_map.png')}")
print(f"  - {os.path.join(OUT_DIR, 'task04_geom_stats.json')}")
print(f"  - {os.path.join(OUT_DIR, 'task04_fuzzy_stats.json')}")
