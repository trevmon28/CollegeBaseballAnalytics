# Data Model: Ecliptic Works — Phase 1 FX Strategy

**Branch**: `002-ecliptic-works-fx` | **Date**: 2026-06-07

---

## Entities

### FXPriceBar

One day's price data for a single currency pair.

| Field | Type | Notes |
|-------|------|-------|
| date | date | Trading date (NY close convention) |
| pair | str | e.g. `EURUSD`, `GBPUSD` |
| open | float | Log price |
| high | float | Log price |
| low | float | Log price |
| close | float | Log price (primary field) |
| volume | float | Tick volume from Dukascopy (informational) |
| log_return | float | `close_t - close_{t-1}` |

**Source**: Dukascopy tick data aggregated to daily.
**Storage**: `data/raw/fx/{pair}/{year}.parquet`

---

### MacroVintage

A point-in-time observation of a macro series — the value that was available on a specific date, not the subsequently revised figure.

| Field | Type | Notes |
|-------|------|-------|
| series_id | str | FRED/ALFRED series identifier |
| vintage_date | date | Date on which this value was available |
| reference_period | date | The period the value describes (e.g., end of quarter) |
| value | float | The reported value as of vintage_date |
| currency | str | ISO currency code this series belongs to (nullable for global series) |
| lag_days | int | Observed publication lag in days (reference_period → vintage_date) |

**Source**: ALFRED API (FRED vintage endpoint) for most series; fixed-lag rules for currencies without ALFRED coverage.
**Storage**: `data/raw/macro/{series_id}.parquet`

---

### FeatureRow

One row in the point-in-time feature matrix — all inputs available for a given date and currency.

| Field | Type | Notes |
|-------|------|-------|
| date | date | Backtest date |
| currency | str | ISO currency code (e.g., `EUR`, `GBP`) |
| log_spot | float | Log end-of-day spot rate vs USD |
| log_return_1m | float | 1-month log return (for momentum signal) |
| rate_differential | float | Annualized short rate (foreign - USD) — carry signal input |
| current_account_gdp | float | Current account as % GDP, latest vintage |
| inflation_yoy | float | CPI YoY, latest vintage |
| risk_score | float | Macro-dashboard composite risk score (0=risk-off, 1=risk-on) |
| risk_score_date | date | Date of the risk score used (must be ≤ feature date) |
| data_complete | bool | True if all required fields are populated; False flags incomplete rows |

**Source**: Derived by joining FXPriceBar + MacroVintage + risk score from macro-dashboard.
**Storage**: `data/processed/feature_matrix.parquet`

---

### CarryPortfolio

The LRV carry portfolio for one rebalancing date: currency weights and realized excess return over the holding period.

| Field | Type | Notes |
|-------|------|-------|
| rebalance_date | date | Date portfolio was formed |
| holding_end_date | date | End of holding period |
| currency | str | Currency in the portfolio |
| portfolio_bin | int | 1 (lowest carry) to 5 (highest carry) |
| weight | float | ±1 normalized weight (+ = long, - = short) |
| rate_differential | float | Forward discount used for sorting |
| excess_return | float | Realized log excess return over holding period |

**Storage**: `data/processed/carry_portfolio.parquet`

---

### MomentumPortfolio

The Menkhoff et al. cross-sectional momentum portfolio for one rebalancing period.

| Field | Type | Notes |
|-------|------|-------|
| rebalance_date | date | Date portfolio was formed |
| holding_end_date | date | End of holding period |
| currency | str | Currency in the portfolio |
| momentum_rank | int | Rank by prior 1-month return (1 = worst, 9 = best for G10) |
| weight | float | ±1 normalized weight |
| formation_return | float | Prior 1-month return used for ranking |
| excess_return | float | Realized log excess return over holding period |

**Storage**: `data/processed/momentum_portfolio.parquet`

---

### RiskScore

Daily composite risk score from the macro-dashboard pipeline.

| Field | Type | Notes |
|-------|------|-------|
| date | date | Score date |
| score | float | 0.0 (maximum risk-off) to 1.0 (maximum risk-on) |
| components | dict | Sub-scores by component (yield curve, credit, VIX, etc.) |
| generated_at | datetime | When the macro-dashboard produced this score |
| is_stale | bool | True if score is older than 7 calendar days |

**Source**: Loaded from macro-dashboard output (`data/outputs/risk_score.parquet` or equivalent).
**Storage**: Read-only from macro-dashboard; cached in `data/processed/risk_score_aligned.parquet` after date alignment.

---

### ConditionedSignal

The final daily strategy signal: carry weights scaled by the risk overlay, ready for execution.

| Field | Type | Notes |
|-------|------|-------|
| signal_date | date | Date signal is computed |
| currency | str | Target currency |
| base_carry_weight | float | Unconditioned carry weight (from CarryPortfolio) |
| risk_scalar | float | Exposure multiplier from conditioning overlay (0–1) |
| conditioned_weight | float | `base_carry_weight * risk_scalar` — final position weight |
| target_notional_usd | float | Dollar notional for this position given portfolio size |

**Storage**: `data/signals/signal_{YYYY-MM-DD}.json` (one file per day, append-only history)

---

### TrialRecord

One evaluated parameter configuration and its results — the input to the DSR gate.

| Field | Type | Notes |
|-------|------|-------|
| trial_id | int | Auto-incrementing across the research session |
| session_id | str | UUID for the research session (reset per new strategy idea, not per run) |
| timestamp | datetime | When this trial was evaluated |
| parameters | dict | Full parameter set evaluated (carry sort bins, momentum window, etc.) |
| in_sample_sharpe | float | Observed annualized Sharpe over training period |
| in_sample_return | float | Annualized return |
| return_skew | float | Return skewness (input to DSR) |
| return_kurtosis | float | Excess kurtosis (input to DSR) |
| n_observations | int | Number of return observations |
| dsr | float | Deflated Sharpe Ratio |
| dsr_passes | bool | True if DSR clears the minimum threshold (default 0.5) |

**Storage**: `data/trials/trial_registry.parquet` (append-only)

---

### Order

An FX order submitted to the IBKR paper account.

| Field | Type | Notes |
|-------|------|-------|
| order_id | str | IBKR-assigned order ID |
| signal_date | date | Signal date that generated this order |
| submitted_at | datetime | Submission timestamp |
| pair | str | FX pair (e.g., `EURUSD`) |
| direction | str | `BUY` or `SELL` |
| quantity | float | Order quantity in base currency units |
| order_type | str | `MKT` or `LMT` |
| limit_price | float | Nullable; populated for LMT orders |
| status | str | `SUBMITTED`, `FILLED`, `CANCELLED`, `FAILED` |

---

### Fill

Execution fill record from IBKR.

| Field | Type | Notes |
|-------|------|-------|
| fill_id | str | IBKR execution ID |
| order_id | str | FK → Order.order_id |
| filled_at | datetime | Fill timestamp |
| pair | str | FX pair |
| quantity_filled | float | |
| fill_price | float | |
| commission | float | Estimated commission |

---

### ReconciliationRecord

Daily reconciliation result: actual paper position vs. target.

| Field | Type | Notes |
|-------|------|-------|
| recon_date | date | |
| currency | str | |
| target_notional_usd | float | From ConditionedSignal |
| actual_notional_usd | float | From IBKR position query |
| discrepancy_usd | float | `actual - target` |
| within_tolerance | bool | True if `abs(discrepancy) < tolerance` (default $100) |
| notes | str | Nullable; populated if within_tolerance is False |

**Storage**: `data/execution/reconciliation.parquet` (append-only)

---

## State Transitions

### Order lifecycle
```
SUBMITTED → FILLED       (normal path)
SUBMITTED → CANCELLED    (manual cancel or session end)
SUBMITTED → FAILED       (connection error or IBKR rejection)
FAILED    → [retry next session]  (not auto-retried same session)
```

### Data pipeline states
```
RAW (Dukascopy/ALFRED) → PROCESSED (feature_matrix) → SIGNAL → EXECUTED → RECONCILED
```

Each stage writes to disk before the next stage begins (artifact-first pattern).

---

## Storage Layout

```text
macro-dashboard/Ecliptic-works/
├── data/
│   ├── raw/
│   │   ├── fx/
│   │   │   └── {PAIR}/          # EURUSD/, GBPUSD/, etc.
│   │   │       └── {YEAR}.parquet
│   │   └── macro/
│   │       └── {SERIES_ID}.parquet
│   ├── processed/
│   │   ├── feature_matrix.parquet
│   │   ├── carry_portfolio.parquet
│   │   ├── momentum_portfolio.parquet
│   │   └── risk_score_aligned.parquet
│   ├── signals/
│   │   └── signal_{YYYY-MM-DD}.json
│   ├── trials/
│   │   └── trial_registry.parquet
│   └── execution/
│       ├── orders.parquet
│       ├── fills.parquet
│       └── reconciliation.parquet
```
