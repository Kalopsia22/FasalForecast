# Crop Yield Monitor (satellite + ML)

District-level crop health, mid-season yield forecasts with uncertainty, season
comparison and state rollups for wheat, chickpea, rice and maize across 17 Indian states.

## Run order
| Step | Command | Needs |
|---|---|---|
| 0 | `python 00_prepare_yield.py` | `../icrisat_district_level_raw.csv` (done; `data/yield_long.csv` included) |
| 1 | `python 01_gee_extract_features.py --project YOUR_GCP_PROJECT --crop all` | Earth Engine login |
| 1b | `python 01_gee_extract_features.py --project P --crop all --years 2025` | current season (refresh weekly) |
| 2 | `python 02_join_features_and_yield.py` | step 1 output |
| 3 | `python 03_train_models.py` | step 2 output (~10-20 min; `--fast` for a quick test) |
| 4 | `python 04_export_boundaries.py --project YOUR_GCP_PROJECT` | for the map |
| 5 | `streamlit run app.py` | |

Check `unmatched_districts.csv` after step 2: split districts need a manual decision.

## Try it without Earth Engine (DEMO, synthetic features)
Demo data and models are already in `data_demo/` and `models_demo/`:

    CROP_DATA_DIR=data_demo CROP_MODEL_DIR=models_demo streamlit run app.py

Yields are real; satellite/weather features are fabricated, so the accuracy shown is meaningless.
Regenerate with `python make_demo_data.py` then steps 2-3 with `--data-dir data_demo --model-dir models_demo`.

## Deploy
Push to GitHub (keep `data/`, `models/` small) and deploy on Streamlit Community Cloud.
The app only reads saved files, so it never needs Earth Engine credentials.

## Assumptions to verify
- ICRISAT year Y = agricultural year Y to Y+1 (season windows in `common.py` depend on it).
- GAUL state names (`01 ... --list-states`); Telangana may sit under Andhra Pradesh.
- Yield plausibility caps in `common.py` (`max_yield`) and the 500 ha minimum area in step 0.
