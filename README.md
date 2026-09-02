# algal-bloom-forecast

Research prototype for forecasting cyanobacterial harmful algal bloom intensity in western Lake Erie at 1-, 3-, 7-, and 14-day horizons.

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data and geospatial | pandas, NumPy, xarray, GeoPandas, Rasterio |
| Machine learning | scikit-learn, XGBoost/LightGBM, PyTorch |
| Visualization | Matplotlib, Plotly |
| Development | pytest, Ruff |

## Current release

The project includes a leakage-safe regional data pipeline, temporal evaluation splits, baseline and candidate model results, and a validated descriptive spatial extension. The current target is a historical 10-day composite series; production alerts, exact daily labels, and calibrated event probabilities are not included.

See the [research release manifest](data/manifests/algal_bloom_project_release_20260902T000951Z.json) for the validated artifact index.
