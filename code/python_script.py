import os, json, warnings
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt

# ---- Paths (relative, not absolute) ----
tif_path = "data/assignment_ndvi_s2_2024_growing_season.tif"
out_dir  = "outputs"
os.makedirs(out_dir, exist_ok=True)

# ---- Check that the file exists ----
assert os.path.exists(tif_path), f"❌ File not found: {tif_path}"
print(f"✅ Found NDVI file: {tif_path}")

# ---- Open raster and read metadata ----
with rio.open(tif_path) as src:
    profile = src.profile.copy()
    band1 = src.read(1).astype("float32")
    nodata = src.nodata

# ---- Mask NoData values ----
if nodata is not None:
    band1 = np.where(band1 == nodata, np.nan, band1)

valid = np.isfinite(band1)
vals  = band1[valid]

# ---- Compute summary statistics ----
summary = {
    "path": tif_path,
    "driver": profile.get("driver"),
    "width": profile.get("width"),
    "height": profile.get("height"),
    "count_bands": profile.get("count"),
    "dtype": str(profile.get("dtype")),
    "crs": str(profile.get("crs")),
    "transform": str(profile.get("transform")),
    "valid_pixels": int(valid.sum()),
    "total_pixels": int(band1.size),
    "pct_valid": float(100.0 * valid.sum() / band1.size),
    "min": float(np.nanmin(vals)) if vals.size else np.nan,
    "p01": float(np.nanpercentile(vals, 1)) if vals.size else np.nan,
    "p05": float(np.nanpercentile(vals, 5)) if vals.size else np.nan,
    "median": float(np.nanmedian(vals)) if vals.size else np.nan,
    "mean": float(np.nanmean(vals)) if vals.size else np.nan,
    "std": float(np.nanstd(vals)) if vals.size else np.nan,
    "p95": float(np.nanpercentile(vals, 95)) if vals.size else np.nan,
    "p99": float(np.nanpercentile(vals, 99)) if vals.size else np.nan,
    "max": float(np.nanmax(vals)) if vals.size else np.nan,
}

# ---- Save summary as CSV and JSON ----
pd.DataFrame([summary]).to_csv(f"{out_dir}/task02_summary.csv", index=False)
with open(f"{out_dir}/task02_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---- Basic validation checks ----
warnings_list = []
if summary["count_bands"] != 1:
    warnings_list.append("Expected a single-band NDVI raster.")
if summary["min"] < -1.2 or summary["max"] > 1.2:
    warnings_list.append("NDVI range should be ~[-1, 1]; values outside may indicate scaling issues.")
if summary["pct_valid"] < 30:
    warnings_list.append("Very low % of valid pixels; cloud mask or export region might be off.")
if "EPSG:4326" not in summary["crs"] and "4326" not in summary["crs"]:
    warnings_list.append(f"CRS is {summary['crs']} (not EPSG:4326). This is OK, just be aware of units.")

# ---- Histogram ----
if vals.size:
    plt.figure()
    plt.hist(vals, bins=60, color="green", alpha=0.7)
    plt.title("NDVI Histogram")
    plt.xlabel("NDVI"); plt.ylabel("Frequency")
    plt.savefig(f"{out_dir}/task02_hist.png", dpi=150)
    plt.close()

# ---- Quicklook image ----
if vals.size:
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = -0.2, 0.9

    plt.figure()
    plt.imshow(band1, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(label="NDVI")
    plt.title("NDVI Quicklook (Robust 2–98% Stretch)")
    plt.axis("off")
    plt.savefig(f"{out_dir}/task02_quicklook.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- Print summary to console ----
print("\n=== NDVI VALIDATION SUMMARY ===")
for k, v in summary.items():
    print(f"{k:>14}: {v}")

if warnings_list:
    print("\n⚠️ WARNINGS:")
    for w in warnings_list:
        print(f" - {w}")
else:
    print("\n✅ Looks like a valid NDVI float raster.")

print(f"\nWrote outputs to: {out_dir}/")
print(" - task02_summary.csv")
print(" - task02_summary.json")
print(" - task02_quicklook.png")
print(" - task02_hist.png")
