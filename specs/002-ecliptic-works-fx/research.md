# Research: Ecliptic Works — Phase 1 FX Strategy

**Branch**: `002-ecliptic-works-fx` | **Date**: 2026-06-07

---

## 1. Dukascopy FX Data Ingestion

**Decision**: Use the `duka` Python library to pull Dukascopy bi5 tick data and resample to daily OHLCV. Fall back to direct HTTP fetch of their public archive if `duka` is unavailable or unmaintained.

**Rationale**: Dukascopy publishes tick data as LZMA-compressed bi5 files at a predictable URL pattern (`https://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5`). The `duka` library wraps this, handles decompression, and exposes a pandas DataFrame. The data is free, requires no account, and covers G10 pairs back to ~2003 with good completeness.

**Alternatives considered**:
- OANDA REST API (free practice account): Good quality, simpler API, but only ~5 years of history — insufficient for the 2000-present backtest requirement.
- Alpha Vantage (free tier): FX daily data available but limited history and API call caps make bulk historical pull impractical.
- Bloomberg / Refinitiv: Institutional quality but not accessible at retail without a subscription.

**Implementation note**: Dukascopy stores data by hour in UTC. Resample to daily using 17:00 NY close convention (23:00 UTC in winter, 22:00 UTC in summer) to match standard FX closing practice. Store as Parquet per pair per year.

---

## 2. ALFRED Point-in-Time Macro Data

**Decision**: Use the FRED API with the `vintage_dates` parameter (ALFRED endpoint) to retrieve the value of each macro series as it was reported on a given date, not the current revised value.

**Rationale**: The standard FRED API (`fredapi` Python library) returns the most recent revised values — silently introducing look-ahead bias into any macro feature. ALFRED (Archival FRED) tracks every revision with a `realtime_start` / `realtime_end` window. Querying `realtime_start=date, realtime_end=date` returns the vintage available on that exact date.

**Key series and their publication lags**:

| Series | FRED ID | Typical Lag |
|--------|---------|------------|
| Fed Funds Rate | FEDFUNDS | ~5 days (monthly) |
| ECB policy rate | ECBMLFR | ~5 days |
| BoE base rate | BOERUKM | ~5 days |
| BoJ policy rate | IRSTJPN | ~30 days |
| US Current Account (% GDP) | CURRENTACCOUNT | ~90 days (quarterly) |
| Euro Area Current Account | B9B1Q027S1 | ~90 days |
| US CPI YoY | CPIAUCSL | ~14 days |
| Global Risk Aversion (VIX) | VIXCLS | ~1 day |

**For currencies lacking FRED coverage** (AUD, NZD, NOK, SEK policy rates): supplement with World Bank API or IMF IFS, applying a conservative fixed lag of 30 days.

**Alternatives considered**:
- Manual download from central bank websites: Covers gaps but not scriptable for vintage history.
- Quandl/Nasdaq Data Link: Has some macro vintages but coverage is inconsistent across currencies.

---

## 3. Forward Discount / Carry Signal Construction

**Decision**: Approximate the forward discount (the carry signal) using the short-term interest rate differential between the foreign currency and USD, following the Lustig-Roussanov-Verdelhan (2011) methodology. Use 1-month interbank rates (LIBOR successors / risk-free rates) where available; overnight policy rates as fallback.

**Rationale**: The true forward discount is `f - s` (log forward minus log spot), which equals the interest rate differential by covered interest parity (CIP). In practice, retail data sources provide rates rather than forwards directly. The CIP relationship holds closely for G10 at daily horizon after 2010; pre-2010, small deviations exist but are not material for monthly-rebalancing carry portfolios.

**Carry portfolio construction** (Lustig et al. spec):
1. Sort G10 currencies monthly by annualized forward discount (interest differential vs USD).
2. Assign to 3–5 portfolios (low to high forward discount).
3. Carry factor = return of highest-discount portfolio minus return of lowest-discount portfolio (HML_FX).
4. Compute log excess return: `rx_{t+1} = Δs_{t+1} + (i* - i)_t` where `Δs` is log spot change.

**Minimum valid currencies per period**: 6 (if fewer G10 currencies have valid data, skip that rebalancing period rather than form a degenerate portfolio).

---

## 4. Cross-Sectional FX Momentum (Menkhoff et al.)

**Decision**: Implement the Menkhoff, Sarno, Schmeling & Schrimpf (2012) specification: rank G10 currencies by their excess return over the past 1 month, go long top-3 / short bottom-3, hold for 1 month.

**Rationale**: The 1-month formation / 1-month holding specification is the one that showed the strongest published results and is most widely replicated. Longer formation windows (3-month, 12-month) show decay post-publication. Dollar-neutral by construction.

**Implementation note**: Skip-month between formation and holding period is standard in equity momentum to avoid reversal; Menkhoff et al. do not require it for FX momentum, so it is not applied here.

---

## 5. Deflated Sharpe Ratio

**Decision**: Implement the Bailey & López de Prado (2014) DSR formula directly. The DSR adjusts the observed Sharpe ratio for (a) number of trials, (b) return skewness and excess kurtosis, (c) sample length.

**Formula** (for implementation reference):
```
DSR = SR_hat * sqrt((T-1)/T) / sqrt(1 - γ₃*SR_hat + ((γ₄-1)/4)*SR_hat²)
```
Where `SR_hat` is the observed annualized Sharpe, `T` is the number of observations, `γ₃` is return skewness, `γ₄` is excess kurtosis. The trial-count adjustment multiplies the effective Sharpe by `1 / sqrt(ln(N))` where `N` is the number of independent trials (using the Bonferroni-style deflation from their paper).

**Minimum sample length**: At minimum 60 monthly observations (~5 years) before any DSR report is considered meaningful.

**Alternatives considered**:
- Simple Sharpe with t-stat threshold (Harvey et al. 2016 suggest t > 3.0): Simpler but doesn't adjust for distribution moments.
- Bootstrap confidence intervals: More robust for fat-tailed returns but computationally heavier; deferred to Phase 2.

---

## 6. Purged + Embargoed Cross-Validation

**Decision**: Implement the López de Prado (2018) combinatorial purged cross-validation approach. Use walk-forward (expanding window) as the primary CV scheme; add an embargo of `embargo_pct = 0.01` (1% of the sample on each side of each test fold boundary).

**Rationale**: Standard k-fold CV on financial time series leaks information because adjacent observations are correlated. Purging removes training observations that overlap with test observations in time. Embargoing adds a buffer zone to handle autocorrelation that extends beyond the immediate overlap. Walk-forward (expanding window) is preferred over rolling window for strategy development because it uses more data and avoids the "forgetting" of early history.

**Embargo period rule**: Embargo = max(1 trading day, `ceil(autocorrelation_halflife_of_signal * 2)`) on each side of the test fold. For daily carry/momentum signals, typical autocorrelation halflife is ~5 days → embargo of 10 trading days is the default.

---

## 7. IBKR TWS API via ib_insync

**Decision**: Use `ib_insync` (Python asyncio wrapper for the IBKR TWS API) for order submission and fill logging. Connect to TWS on port 7497 (paper trading).

**Rationale**: `ib_insync` is the most actively maintained Python client for IBKR. It handles reconnection, request throttling, and provides a clean async interface. For daily-frequency rebalancing, performance overhead of the async model is irrelevant.

**FX instrument representation in ib_insync**:
```python
from ib_insync import Forex
eurusd = Forex('EURUSD')  # IBKR's FX pair format
```
Orders are submitted as `MarketOrder` at daily close, or as `LimitOrder` with a mid-price limit for tighter fills.

**Connection requirement**: TWS or IB Gateway must be running on the local machine with API access enabled. The execution layer handles `ConnectionError` gracefully by logging and retrying once; if the second attempt fails, it logs the unfilled target and exits cleanly.

---

## 8. Project Location and Environment

**Decision**: Ecliptic Works source code lives at `C:\Users\trevm\Projects\macro-dashboard\Ecliptic-works\`. The existing `macro_env` conda environment is extended with new dependencies rather than creating a new environment.

**New dependencies to add to macro_env**:
- `duka` — Dukascopy data ingestion
- `ib_insync` — IBKR TWS API
- `pyarrow` — Parquet I/O (likely already present)
- `scipy` — DSR formula (skewness/kurtosis)
- `scikit-learn` — purged CV utilities

**Rationale**: The macro-dashboard pipeline is the upstream data source (risk score). Co-locating in the same environment avoids dependency management overhead and allows direct import of dashboard outputs.

---

## 9. Daily Signal Format (Internal Contract)

**Decision**: The strategy's daily output is a JSON file written to `data/signals/signal_YYYY-MM-DD.json` with a defined schema (see contracts/signal-schema.md). This file is the hand-off point between research/backtesting and execution.

**Rationale**: File-based hand-off decouples the signal computation from execution, enables debugging by inspecting historical signal files, and mirrors the artifact-first pattern from the baseball analytics project.
