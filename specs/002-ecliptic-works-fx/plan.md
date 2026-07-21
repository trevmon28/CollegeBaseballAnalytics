# Implementation Plan: Ecliptic Works — Phase 1 FX Strategy

**Branch**: `002-ecliptic-works-fx` | **Date**: 2026-06-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-ecliptic-works-fx/spec.md`

---

## Summary

Build a systematic, market-neutral G10 FX strategy in four layers: a point-in-time data pipeline (Dukascopy + ALFRED), a methods harness enforcing the Deflated Sharpe Ratio gate and purged cross-validation, replicated carry and momentum factors validated against published benchmarks, and a conditioning overlay that scales carry exposure using the macro-dashboard's risk score. The execution layer wires the conditioned daily signal to IBKR paper trading via the TWS API. The project lives at `C:\Users\trevm\Projects\macro-dashboard\Ecliptic-works\` and runs in the existing `macro_env` conda environment.

---

## Technical Context

**Language/Version**: Python 3.11+ (existing `macro_env`)

**Primary Dependencies**:
- `duka` — Dukascopy tick data ingestion
- `fredapi` + ALFRED endpoint — point-in-time macro series
- `pandas`, `numpy`, `scipy` — data processing, DSR formula
- `scikit-learn` — walk-forward CV utilities
- `ib_insync` — IBKR TWS API (paper trading)
- `pyarrow` — Parquet I/O
- Existing macro-dashboard pipeline — risk score input

**Storage**: Parquet files for all time-series data; JSON for daily signal files; append-only Parquet for trial registry and execution logs.

**Testing**: pytest; unit tests for DSR formula and carry construction; integration tests for data pipeline against known fixture data.

**Target Platform**: Windows workstation (primary); code must be POSIX-compatible for potential future VPS deployment.

**Project Type**: Research pipeline + daily execution system (quant finance).

**Performance Goals**: Full daily pipeline (data refresh → signal → IBKR submission) in < 10 minutes (SC-008).

**Constraints**: Daily-to-weekly rebalancing only; no intraday; IBKR TWS must be running locally for execution.

**Scale/Scope**: G10 FX (9 pairs), daily data 2000-present (~26 years × 9 pairs × ~250 days = ~58,500 rows per pair).

---

## Constitution Check

*The CFBBaseballAnalytics constitution (v1.0.0) governs the baseball analytics project. Ecliptic Works is a separate project and is not bound by those specific API and pipeline rules. However, the following analogous principles are adopted:*

| Constitution Principle | Ecliptic Works Analogue | Status |
|----------------------|------------------------|--------|
| Artifact-first pipeline | All pipeline stages write to disk before downstream stages read | Adopted |
| Schema-first contracts | Daily signal JSON schema defined before execution layer | Adopted |
| Graceful degradation | Connection failures logged cleanly; no silent skips | Adopted |
| Freshness transparency | Signal JSON includes `generated_at` and `risk_score_stale` | Adopted |
| Read-only separation | Research layer never mutates execution logs; execution never re-runs research | Adopted |

No constitution violations. Ecliptic Works should define its own constitution before Phase 2 begins.

---

## Project Structure

### Documentation (this feature)

```text
specs/002-ecliptic-works-fx/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/
│   └── signal-schema.md # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
macro-dashboard/Ecliptic-works/
├── ecliptic/
│   ├── data/
│   │   ├── ingest_fx.py          # Dukascopy pull → data/raw/fx/
│   │   ├── ingest_macro.py       # ALFRED pull → data/raw/macro/
│   │   └── build_features.py     # Join → data/processed/feature_matrix.parquet
│   ├── harness/
│   │   ├── trial_registry.py     # Append-only trial log
│   │   ├── dsr.py                # Deflated Sharpe Ratio computation
│   │   └── cv.py                 # Purged + embargoed walk-forward CV
│   ├── factors/
│   │   ├── carry.py              # LRV carry portfolio construction
│   │   └── momentum.py           # Menkhoff momentum portfolio construction
│   ├── conditioning/
│   │   └── overlay.py            # Risk score → exposure scalar → ConditionedSignal
│   ├── execution/
│   │   ├── ibkr.py               # ib_insync connection + order submission
│   │   └── reconcile.py          # Daily position reconciliation
│   ├── backtest/
│   │   ├── engine.py             # Vectorized backtest runner
│   │   └── report.py             # Performance report + replication comparison
│   └── research/
│       └── replicate.py          # Factor replication CLI
├── data/                         # gitignored (all artifacts)
│   ├── raw/
│   │   ├── fx/
│   │   └── macro/
│   ├── processed/
│   ├── signals/
│   ├── trials/
│   └── execution/
├── tests/
│   ├── unit/
│   │   ├── test_dsr.py
│   │   ├── test_carry.py
│   │   ├── test_momentum.py
│   │   └── test_overlay.py
│   └── integration/
│       ├── test_feature_pipeline.py
│       └── test_signal_schema.py
├── configs/
│   └── baseline.yml              # Default research config
├── run_research.py               # Full research pipeline entrypoint
├── run_daily.py                  # Daily paper-trading entrypoint
└── requirements_ecliptic.txt     # Additional deps beyond macro_env base
```

**Structure Decision**: Single project layout under `Ecliptic-works/`. The `ecliptic/` package contains all importable modules; `run_*.py` scripts are the CLI entrypoints. `data/` is gitignored — artifacts are not committed. The execution layer (`ibkr.py`) imports only from `data/signals/` — never from `factors/` or `conditioning/` directly.

---

## Phase 0: Research — Complete

All decisions resolved. See [research.md](research.md) for full rationale.

**Key decisions**:

| Decision | Choice |
|----------|--------|
| FX data source | Dukascopy `duka` library; daily OHLCV at 17:00 NY close |
| Macro vintage source | FRED ALFRED API (`realtime_start=date` parameter) |
| Carry signal | Interest rate differential (CIP proxy); 1–3 portfolio bins; monthly rebalance |
| Momentum spec | Menkhoff 1m formation / 1m hold; no skip-month; G10 top-3 vs bottom-3 |
| DSR implementation | Bailey & López de Prado (2014) formula; min 60 monthly observations |
| CV scheme | Walk-forward expanding window; embargo = max(1d, 2× autocorrelation halflife); default 10 days |
| IBKR connection | `ib_insync` port 7497 (paper); single retry on connection failure |
| Python environment | `macro_env` extended; no new environment |

---

## Phase 1: Design — Complete

### Data Flow

```
Dukascopy tick data ─────────────────────────────────────────────────────┐
                                                                          ├── feature_matrix.parquet
ALFRED macro vintages (FRED API, realtime_start) ────────────────────────┘
                                                                          │
                                              carry_portfolio.parquet ────┤
                                           momentum_portfolio.parquet ────┤
                                                                          ↓
macro-dashboard risk_score.parquet ─────────────────────────── ConditionedSignal
                                                                          │
                                                         signal_{date}.json
                                                                          │
                                                    IBKR Paper (TWS API, port 7497)
                                                                          │
                                                         reconciliation.parquet
```

### Research Loop (backtesting)

```
feature_matrix → carry_portfolio → trial_registry (append N trials)
                                             │
                                     DSR gate check
                                             │
                                    passes → conditioning overlay
                                             │
                                      backtest report (conditioned vs unconditioned)
```

### Key Invariants

1. **No look-ahead**: `build_features.py` asserts every macro value on date `t` has `vintage_date ≤ t`. Verified on a random 5% row sample.
2. **Trial counter is mandatory**: `run_research.py` refuses to run without an initialized `trial_registry.parquet`.
3. **DSR gate blocks execution**: `run_daily.py` reads `dsr_passes` from the signal JSON and skips order submission if False.
4. **Signal is the only hand-off point**: `ecliptic/execution/ibkr.py` reads only from `data/signals/`; no imports from research modules.
5. **Execution logs are append-only**: No row in `orders.parquet`, `fills.parquet`, or `reconciliation.parquet` is ever mutated.

### Artifacts generated (Phase 1)

- [data-model.md](data-model.md) — all entities, fields, storage layout
- [contracts/signal-schema.md](contracts/signal-schema.md) — daily signal JSON schema with invariants
- [quickstart.md](quickstart.md) — setup, data pull, and daily run instructions

---

## Complexity Tracking

> No constitution violations requiring justification.
