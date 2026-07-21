# Contract: Daily Signal Schema

**File pattern**: `data/signals/signal_{YYYY-MM-DD}.json`
**Producer**: `ecliptic/conditioning/overlay.py` (via `run_research.py` or `run_daily.py`)
**Consumer**: `ecliptic/execution/ibkr.py` (execution layer)

---

## Schema

```json
{
  "signal_date": "2026-06-07",
  "generated_at": "2026-06-07T17:05:32Z",
  "risk_score": 0.72,
  "risk_score_date": "2026-06-05",
  "risk_score_stale": false,
  "portfolio_size_usd": 100000,
  "positions": [
    {
      "currency": "EUR",
      "pair": "EURUSD",
      "base_carry_weight": 0.25,
      "momentum_weight": 0.10,
      "combined_weight": 0.35,
      "risk_scalar": 0.72,
      "conditioned_weight": 0.252,
      "target_notional_usd": 25200,
      "direction": "LONG"
    }
  ],
  "carry_factor_return_1d": 0.0012,
  "momentum_factor_return_1d": 0.0008,
  "n_currencies_active": 7,
  "dsr_passes": true,
  "dsr_value": 0.84,
  "notes": ""
}
```

---

## Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| signal_date | ISO date string | Yes | The date this signal applies to (next trading day) |
| generated_at | ISO datetime (UTC) | Yes | Timestamp of signal computation |
| risk_score | float [0–1] | Yes | Current macro risk score (1 = fully risk-on) |
| risk_score_date | ISO date string | Yes | Date the risk score was last computed |
| risk_score_stale | bool | Yes | True if risk_score_date is > 7 calendar days ago |
| portfolio_size_usd | float | Yes | Total notional in USD for sizing calculations |
| positions | array | Yes | One entry per active currency position |
| positions[].currency | str (ISO 4217) | Yes | 3-letter currency code |
| positions[].pair | str | Yes | FX pair as traded (always vs USD) |
| positions[].base_carry_weight | float | Yes | Raw carry weight before conditioning |
| positions[].momentum_weight | float | Yes | Raw momentum weight before conditioning |
| positions[].combined_weight | float | Yes | `base_carry + momentum` before risk scaling |
| positions[].risk_scalar | float [0–1] | Yes | Conditioning multiplier applied |
| positions[].conditioned_weight | float | Yes | `combined_weight * risk_scalar` |
| positions[].target_notional_usd | float | Yes | `conditioned_weight * portfolio_size_usd` |
| positions[].direction | str (`LONG`/`SHORT`) | Yes | Derived from sign of conditioned_weight |
| carry_factor_return_1d | float | Yes | Yesterday's carry factor return (diagnostic) |
| momentum_factor_return_1d | float | Yes | Yesterday's momentum factor return (diagnostic) |
| n_currencies_active | int | Yes | Number of currencies with valid data on this date |
| dsr_passes | bool | Yes | Whether the strategy has passed the DSR gate |
| dsr_value | float | Yes | Current DSR value (from trial registry) |
| notes | str | No | Any manual override or diagnostic note |

---

## Invariants

- `conditioned_weight = combined_weight * risk_scalar` (always)
- `target_notional_usd = conditioned_weight * portfolio_size_usd` (always)
- If `risk_score_stale == true`, all `risk_scalar` values MUST be ≤ 0.5 (conservative default)
- If `dsr_passes == false`, the execution layer MUST log a warning and skip order submission
- `n_currencies_active` MUST be ≥ 4; if fewer, signal is invalid and `positions` array is empty
- The sum of absolute `conditioned_weight` values MUST be ≤ 1.5 (leverage cap for Phase 1)

---

## Versioning

Schema version is implicit in the file format. Breaking changes require a new filename pattern (e.g., `signal_v2_{YYYY-MM-DD}.json`). Non-breaking additions (new diagnostic fields) are allowed without a version bump.
