# Task 03 — Probability with NDVI

## C1. P(NDVI ≥ t) — whole-image results
Attach `outputs/task03_p_events.csv` and `outputs/task03_event_prob_plot.png`.
Interpretation: describe how P changes with threshold; indicate a practical threshold (e.g., 0.3) and its meaning.

## C2. Regional comparison
Describe how regions were defined (2×2 grid). Attach `outputs/task03_p_events_by_region.csv` and `outputs/task03_region_bar.png`.
Discuss which regions show higher vegetation probability and possible reasons (e.g., built-up area, parks, water).

## C3. LLN demonstration (empirical)
Attach `outputs/task03_running_mean_samples.csv` and `outputs/task03_running_mean.png`.
Explain how the spread of sample means decreases as sample size increases (Law of Large Numbers).

## C4. CLT demonstration
Using sample means for a given sample size (e.g., n=100 or 1000), check approximate normality (histogram or QQ) and report sample mean ± std/sqrt(n). Discuss if distribution is approximately Gaussian.

## C5. Statistical interpretation and limitations
Discuss sources of bias: cloud masking, export scaling, spatial autocorrelation (pixels are not independent), edge effects, region definitions, and choice of threshold.

## C6. Conclusions
Short summary of main findings, recommended threshold for vegetation, and suggestions for future work (use stratified regions, bootstrapping, spatial smoothing).

