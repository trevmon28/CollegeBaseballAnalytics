# Quickstart: Ecliptic Works — Phase 1

**Project root**: `C:\Users\trevm\Projects\macro-dashboard\Ecliptic-works\`
**Environment**: `macro_env` conda env

---

## First-time setup

```powershell
# Activate environment
conda activate macro_env

# Install new dependencies
pip install duka ib_insync pyarrow scipy scikit-learn

# Create data directories
cd C:\Users\trevm\Projects\macro-dashboard\Ecliptic-works
python -c "
import os
for d in ['data/raw/fx','data/raw/macro','data/processed',
          'data/signals','data/trials','data/execution']:
    os.makedirs(d, exist_ok=True)
print('Directories created.')
"
```

---

## Pull historical FX data (one-time, ~10–30 min)

```powershell
python -m ecliptic.data.ingest_fx --pairs EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD USDSEK USDNOK --start 2000-01-01
```

Writes to `data/raw/fx/{PAIR}/{YEAR}.parquet`.

---

## Pull historical macro data (one-time)

Requires `FRED_API_KEY` environment variable (same key used by the macro-dashboard).

```powershell
$env:FRED_API_KEY = "your_key_here"
python -m ecliptic.data.ingest_macro --start 2000-01-01
```

Writes to `data/raw/macro/{SERIES_ID}.parquet` using ALFRED vintages.

---

## Build the feature matrix

```powershell
python -m ecliptic.data.build_features --start 2000-01-01 --end 2026-06-01
```

Writes `data/processed/feature_matrix.parquet`. Prints a look-ahead audit summary on completion.

---

## Run factor replication

```powershell
python -m ecliptic.research.replicate --factor carry --start 2000-01 --end 2020-12
python -m ecliptic.research.replicate --factor momentum --start 2000-01 --end 2020-12
```

Prints replication report comparing summary statistics to published benchmarks.

---

## Run a backtested research loop

```powershell
python run_research.py --config configs/baseline.yml
```

Runs the full research loop (data → harness → carry+momentum → conditioning), writes trial records to `data/trials/trial_registry.parquet`, and prints a DSR report.

---

## Daily forward-test run (paper trading)

Prerequisites: TWS or IB Gateway must be running on port 7497.

```powershell
python run_daily.py
```

Sequence:
1. Refreshes FX data for yesterday's close
2. Reloads macro risk score from macro-dashboard output
3. Computes conditioned signal → writes `data/signals/signal_{today}.json`
4. Submits orders to IBKR paper account
5. Runs reconciliation at end of session

---

## Check trial registry and DSR status

```powershell
python -m ecliptic.harness.dsr --report
```

Prints current DSR for the best strategy variant evaluated, number of trials, and pass/fail status.

---

## Key environment variables

| Variable | Description |
|----------|-------------|
| `FRED_API_KEY` | FRED/ALFRED API key (same as macro-dashboard) |
| `ECLIPTIC_PORTFOLIO_SIZE` | Total notional in USD (default: 100000) |
| `ECLIPTIC_DSR_THRESHOLD` | Minimum DSR to approve a strategy (default: 0.5) |
| `ECLIPTIC_IBKR_PORT` | TWS port (default: 7497 for paper, 7496 for live) |
| `ECLIPTIC_RISK_STALENESS_DAYS` | Days before risk score is considered stale (default: 7) |
