# Task 02 — Codespace Walkthrough

**Goal:**  
To verify the NDVI GeoTIFF dataset within GitHub Codespaces and produce a set of validation outputs.

**Process summary:**
- Loaded Sentinel-2 NDVI raster from `/data/`.
- Checked metadata (CRS, resolution, NoData, band count).
- Computed descriptive statistics and saved to `task02_summary.csv` and `.json`.
- Created quicklook and histogram figures for visual inspection.
- Reviewed warnings for NDVI range and pixel validity.

**Observations:**
- NDVI range within expected limits (~ −0.2 – 0.9).  
- Valid pixel coverage acceptable (> 90%).  
- CRS detected as EPSG:32646 (UTM Zone 46 N), which is appropriate for Bangladesh.

**Reflection:**
This exercise confirmed that Codespaces can fully run raster analysis workflows.  
I verified file paths, dependencies, and output management in a reproducible environment.

**Outputs generated:**
- `outputs/task02_summary.csv`
- `outputs/task02_summary.json`
- `outputs/task02_quicklook.png`
- `outputs/task02_hist.png`
