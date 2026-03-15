# Bristol Stock Exchange – MSc FinTech Project
**University of Bristol | 2025–2026**
**Author: Ajeet Kumar**

## Project Overview
This repository contains the full implementation and analysis for the Bristol Stock Exchange (BSE) coursework, covering four experimental questions on automated trading agents and market microstructure.

## Structure
```
├── BSE.py                  # Core BSE exchange engine (AY25/26)
├── bse_analysis.py         # Consolidated Q1–Q4 analysis code
├── bse_data_combined.csv   # Combined experimental output data
│   └── sources: tape, avg_balance, blotters, LOB_frames
└── README.md
```

## Questions Covered

| # | Topic |
|---|-------|
| Q1 | ZIP vs ZIC trader comparison – statistical analysis |
| Q2 | Reproducing Vernon Smith's Chart 5 (1962) with BSE |
| Q3 | MMM01 parameter optimisation (grid search, robustness) |
| Q4 | MMM02 performance comparison vs optimised MMM01* |

## Key Technologies
- **Python** – core simulation and analysis
- **BSE** – Bristol Stock Exchange engine
- **Matplotlib / Seaborn** – visualisation
- **SciPy / Statsmodels** – statistical testing
- **Pandas / NumPy** – data wrangling

## Running the Code
```bash
# Ensure BSE.py is in the same directory
python bse_analysis.py
```
