# Solar Generation Forecasting Model

Forecasts PV generation at 15-min / 1-hr / 24-hr / 7-day horizons with
P10/P50/P90 confidence intervals. Data: DKASC Alice Springs generation +
Open-Meteo historical forecast weather.

## Folder structure
```
Solar_Forecasting_Model/
├── dkasc_raw/                  # raw DKASC CSV downloads go here
├── output/                     # generated: training table, results, plots
├── fetch_dkasc_openmeteo.py    # Phase 1: build training table
├── trim_dkasc.py               # helper: trim full download to 2020-2023
├── phase2_baselines.py         # Phase 2: baselines + frozen eval harness
├── requirements.txt
└── README.md
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Workflow
1. Download a DKASC system CSV (5-min, fixed-tilt system, e.g. #13 Trina)
   from https://dkasolarcentre.com.au/download?location=alice-springs
   into `dkasc_raw/`. If you downloaded the full history, trim it:
   `python trim_dkasc.py`
2. `python fetch_dkasc_openmeteo.py`  -> output/training_table.parquet
3. `python phase2_baselines.py`       -> output/baseline_results.csv
4. Phase 3 (XGBoost) and Phase 4 (LSTM/TFT) scripts: coming next.

## Evaluation rules (frozen)
- Test year: 2023 (train on 2020-2022)
- Daytime only: solar elevation > 5 deg
- Metrics: nMAE %, nRMSE %, bias % (normalised by capacity),
  skill % vs smart persistence
