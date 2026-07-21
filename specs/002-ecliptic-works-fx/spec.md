# Feature Specification: Ecliptic Works — Phase 1 FX Strategy

**Feature Branch**: `002-ecliptic-works-fx`

**Created**: 2026-06-07

**Status**: Draft

**Input**: Systematic market-neutral FX carry/momentum strategy with conditioning layer, built on top of the existing macro-dashboard.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Assemble a Trustworthy Point-in-Time Dataset (Priority: P1)

Trevor needs to pull together daily G10 FX prices and macro series into a single feature matrix where every row reflects only what was knowable on that date — no revised values, no forward-looking releases. He runs the data pipeline once and gets back a labeled dataset he can hand to any model with confidence that the numbers are honest.

**Why this priority**: Every downstream backtest, replication, and conditioning result is only as trustworthy as this foundation. Look-ahead bias here manufactures fake alpha silently. Nothing else can be validated until this layer is clean.

**Independent Test**: Can be fully tested by ingesting two years of G10 FX and macro data, then verifying that macro values on a given backtest date match what ALFRED reported as of that date (not the later revised value) — before any model or signal is built.

**Acceptance Scenarios**:

1. **Given** a specified date range and currency universe, **When** the data pipeline runs, **Then** it produces a daily feature matrix covering all G10 pairs with FX prices, interest rate differentials, current-account balances, and the macro-dashboard risk score — each lagged to its true publication date.
2. **Given** a macro series with a historical revision, **When** the data pipeline queries that series for a past date, **Then** it returns the vintage value available on that date, not the subsequently revised figure.
3. **Given** a macro release with a known publication lag (e.g., current-account data released 60+ days after the reference period), **When** the feature matrix is constructed, **Then** that series is absent from rows that predate its release, not forward-filled with future values.
4. **Given** Dukascopy tick data for a G10 pair, **When** the pipeline aggregates to daily frequency, **Then** it produces a clean end-of-day mid-price and a forward discount (interest differential) aligned by currency and date.

---

### User Story 2 — Run a Signal Search with Honest Overfitting Controls (Priority: P2)

After designing candidate parameter variations for the carry or momentum signal, Trevor runs multiple configurations through the backtest. The methods harness tracks every trial, computes the Deflated Sharpe Ratio for the best performer, and reports whether that result is statistically credible — given how many configurations were tried.

**Why this priority**: Without this gate, the research loop will inevitably produce a beautifully overfit backtest. The harness is the mechanism that enforces honesty before any signal is declared viable.

**Independent Test**: Can be fully tested by running a known number of parameter combinations against historical data and verifying that the harness (a) counts every trial, (b) computes the DSR correctly for the best result, and (c) flags results below the credibility threshold — before any trading signal is deployed.

**Acceptance Scenarios**:

1. **Given** N parameter combinations have been evaluated, **When** the best in-sample Sharpe ratio is reported, **Then** the harness also reports the corresponding Deflated Sharpe Ratio, the trial count used in the adjustment, and whether the DSR clears a minimum credibility threshold.
2. **Given** a result that looks strong in-sample, **When** the DSR adjustment is applied, **Then** results that benefited heavily from search (high N, short sample, or fat-tailed returns) are clearly flagged as not credible.
3. **Given** a backtest validation run, **When** time-series cross-validation is performed, **Then** training folds are separated from test folds by an embargo gap proportional to the signal autocorrelation, preventing any information from the test period leaking into training.
4. **Given** no trial counter is active, **When** a new research session starts, **Then** the harness refuses to report a "final" DSR until the trial counter has been initialized and at least one trial recorded.

---

### User Story 3 — Replicate the Carry and Momentum Factors (Priority: P3)

Trevor constructs the Lustig-Roussanov-Verdelhan G10 carry portfolio and the Menkhoff et al. cross-sectional momentum portfolio on his clean dataset, then compares the return series to published benchmarks. Matching the published magnitudes within reasonable sampling tolerance confirms the data layer is correct and gives him a known-good baseline before he innovates.

**Why this priority**: Replication before innovation is the key discipline from the literature review. If Trevor can't reproduce known results, he can't trust novel ones. This also stress-tests the data layer end-to-end.

**Independent Test**: Can be fully tested by running the replication notebooks and comparing summary statistics (annualized return, Sharpe ratio, max drawdown) to published figures from the original papers — independently of any conditioning or execution layer.

**Acceptance Scenarios**:

1. **Given** clean G10 FX data from 2000 onward, **When** the carry factor is constructed using the standard high-minus-low forward-discount sort, **Then** the annualized excess return and Sharpe ratio fall within ±30% of the Lustig et al. published figures for a comparable sample period.
2. **Given** the same dataset, **When** the cross-sectional momentum factor is constructed using the standard 1-month formation / 1-month holding approach (Menkhoff et al. specification), **Then** the return characteristics (sign of mean return, rough Sharpe magnitude) are directionally consistent with the published results.
3. **Given** a replication run, **When** results deviate materially from published benchmarks, **Then** the system surfaces a diagnostic report identifying which data series or construction step is the likely source of divergence.
4. **Given** successful replication, **When** factor returns are computed out-of-sample (post-publication period), **Then** the system reports whether the factor has remained positive in the post-publication window — with no assertion of statistical significance unless the DSR gate has been passed.

---

### User Story 4 — Apply the Conditioning Overlay and Measure Its Effect (Priority: P4)

Trevor connects the macro-dashboard's composite risk score to the carry strategy: when the risk score signals a risk-off regime, carry exposure is reduced proportionally; when risk-on, full exposure is restored. He then compares conditioned vs. unconditioned performance across the backtest period to see whether the overlay improves the risk-adjusted return or reduces drawdown.

**Why this priority**: Carry's central weakness is crash risk in risk-off episodes (the "nickels in front of a steamroller" property). The conditioning overlay is the primary mechanism that addresses this, and it is the main original contribution of Phase 1 relative to a vanilla carry replication.

**Independent Test**: Can be fully tested by running two backtest variants — one with the overlay applied, one without — and producing a side-by-side comparison of Sharpe ratio, max drawdown, and Calmar ratio, before any execution layer is connected.

**Acceptance Scenarios**:

1. **Given** the macro-dashboard risk score is available as a daily series, **When** the conditioning overlay is applied, **Then** carry position size is scaled between 0% (full risk-off) and 100% (full risk-on) as a continuous function of the risk score, with the mapping rule documented and reproducible.
2. **Given** a known historical risk-off episode (e.g., 2008–09, 2020 COVID), **When** the overlay is evaluated against that period, **Then** the conditioned strategy has materially lower drawdown than the unconditioned carry strategy over that window.
3. **Given** a full backtest comparing conditioned and unconditioned carry, **When** summary statistics are produced, **Then** the output includes annualized return, annualized volatility, Sharpe ratio, max drawdown, and Calmar ratio for both variants, plus the DSR for the conditioned result.
4. **Given** the risk score is unavailable or stale for a given date, **When** the overlay is queried, **Then** it defaults to a conservative (reduced exposure) stance rather than assuming risk-on.

---

### User Story 5 — Paper Trade the Strategy via IBKR (Priority: P5)

Once the backtested signal passes the DSR gate, Trevor runs the strategy in forward paper-trading mode: the daily signal fires, generates target FX positions, submits orders to the IBKR paper account via the TWS API, and logs every fill. At end of day, the reconciliation step confirms actual positions match targets.

**Why this priority**: Validating the execution plumbing on paper before deploying real capital is a non-negotiable step. Order routing, fill logging, and position reconciliation all need to be confirmed to work before live trading begins.

**Independent Test**: Can be fully tested by submitting a known set of synthetic FX orders to the IBKR paper account, verifying fills are received and logged, and running reconciliation to confirm the reported paper position matches the order — independently of any backtest or research layer.

**Acceptance Scenarios**:

1. **Given** a daily signal producing target FX positions, **When** the execution layer runs, **Then** it translates target positions into FX orders, submits them to the IBKR paper account via the TWS connection, and records the order IDs.
2. **Given** submitted orders, **When** fills are received from IBKR, **Then** each fill is logged with timestamp, instrument, quantity, fill price, and order ID.
3. **Given** end-of-day fills, **When** reconciliation runs, **Then** the system confirms that actual paper positions match target positions within a specified tolerance, and flags any discrepancy for review.
4. **Given** the TWS connection is unavailable, **When** the execution layer attempts to submit orders, **Then** it logs the connection failure, retains the unfilled target positions, and does not silently skip or duplicate orders on reconnection.

---

### Edge Cases

- What happens when Dukascopy data has a gap for a specific currency pair on a given date?
- How does the carry sort behave when fewer than 4 G10 currencies have valid forward discounts on a given date?
- What happens when the macro-dashboard pipeline has not yet run for the current week and the risk score is stale?
- How does the methods harness handle a research session where the trial counter was not initialized (e.g., inherited a prior backtest result)?
- What happens if the IBKR paper account has insufficient margin for the target position size?
- How are corporate or central-bank holiday gaps in FX data (e.g., JPY on Japanese holidays) handled in the feature matrix?

---

## Requirements *(mandatory)*

### Functional Requirements

**Data Layer**

- **FR-001**: The system MUST ingest daily G10 FX end-of-day mid-prices from Dukascopy covering at least 2000–present.
- **FR-002**: The system MUST ingest macro series (policy rates, current-account balances, inflation, risk score) from ALFRED (point-in-time vintages) and the macro-dashboard pipeline, lagging each series to its true publication date.
- **FR-003**: The system MUST produce a daily feature matrix where each row contains only information knowable on that date — no look-ahead from revised macro data or unreleased series.
- **FR-004**: The system MUST flag and handle gaps in FX data (holidays, missing feeds) with a documented fill rule rather than silently forward-filling or dropping rows.

**Methods Harness**

- **FR-005**: The system MUST maintain a trial counter that increments each time a distinct parameter configuration is evaluated against the historical dataset.
- **FR-006**: The system MUST compute the Deflated Sharpe Ratio for any reported result, using the trial count, sample length, and return distribution moments as inputs.
- **FR-007**: The system MUST perform time-series cross-validation with an embargo gap between training and test folds; the embargo length MUST be at least as long as the signal's estimated autocorrelation decay period.
- **FR-008**: The system MUST refuse to certify a result as validated if the DSR falls below a documented minimum threshold.

**Factor Replication**

- **FR-009**: The system MUST implement the Lustig-Roussanov-Verdelhan carry factor construction: sort G10 currencies monthly by forward discount (interest rate differential), form a high-minus-low long-short portfolio, compute excess returns.
- **FR-010**: The system MUST implement the Menkhoff et al. cross-sectional FX momentum construction: rank currencies by prior 1-month return, form a high-minus-low portfolio, compute excess returns.
- **FR-011**: The system MUST produce a replication report comparing factor summary statistics to published benchmarks and surfacing any material divergence.

**Conditioning Layer**

- **FR-012**: The system MUST accept the macro-dashboard risk score as a daily input and map it to a continuous carry exposure scalar between 0 and 1.
- **FR-013**: The system MUST produce both conditioned and unconditioned carry return series over the backtest period, enabling direct comparison.
- **FR-014**: When the risk score is unavailable or stale beyond a defined threshold, the system MUST default to a conservative (reduced) exposure level.

**Execution Layer**

- **FR-015**: The system MUST connect to the IBKR paper account via the TWS API and submit FX orders derived from the daily signal.
- **FR-016**: The system MUST log every order submission and fill with timestamp, instrument, quantity, price, and order ID.
- **FR-017**: The system MUST run a daily reconciliation step confirming actual paper positions match target positions, and log any discrepancies.

### Key Entities

- **FX Feature Matrix**: Daily rows × feature columns; each cell holds the point-in-time value of an FX price, interest differential, or macro indicator as of that date.
- **Carry Factor Portfolio**: A daily long-short G10 FX portfolio sorted by forward discount; primary base signal.
- **Momentum Factor Portfolio**: A daily long-short G10 FX portfolio sorted by prior-period return; complementary signal.
- **Risk Score**: A scalar (0–1 or similar) produced by the macro-dashboard representing current global risk regime; drives the conditioning overlay.
- **Conditioned Signal**: The carry factor return scaled by the risk score overlay; the strategy's live output.
- **Trial Registry**: A persistent record of every parameter configuration evaluated, used as input to the DSR gate.
- **Execution Log**: An append-only record of every order, fill, and reconciliation result from the IBKR paper account.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The data pipeline produces a complete G10 FX feature matrix with no look-ahead bias, verified by confirming that any macro value on a past date matches its ALFRED vintage for that date — with zero violations in a spot-check of 50 randomly sampled date/series combinations.
- **SC-002**: The replication of the Lustig-Verdelhan carry factor produces an annualized excess return and Sharpe ratio within ±30% of the published figures for the 2000–2020 sample period.
- **SC-003**: The replication of the Menkhoff et al. momentum factor produces a positive mean excess return directionally consistent with the published results over the same sample.
- **SC-004**: The methods harness correctly computes the DSR for a known synthetic example (constructed with a known number of trials and return moments) — result matches the analytical formula within numerical precision.
- **SC-005**: The conditioned carry strategy demonstrates lower maximum drawdown than the unconditioned carry strategy in at least two of the three most severe historical risk-off episodes in the backtest sample.
- **SC-006**: The conditioned carry strategy's Sharpe ratio passes the DSR gate (credibility threshold) at the 95% confidence level with the trial count accumulated during Phase 1 research.
- **SC-007**: The execution layer successfully submits, fills, and reconciles a batch of 10 synthetic IBKR paper orders with zero unlogged fills and zero position discrepancies at end-of-day reconciliation.
- **SC-008**: The full daily pipeline — data refresh → signal computation → conditioning → order generation → IBKR submission — completes end-to-end in under 10 minutes on the primary workstation.

---

## Assumptions

- The existing macro-dashboard pipeline (FRED + World Bank + IMF, weekly cadence) continues to run and produces a current risk score that can be consumed by this system. The risk score series from the dashboard covers at least 2005–present in backtest history.
- Dukascopy's free tick data archive provides sufficient G10 FX coverage (EUR, GBP, JPY, CHF, AUD, CAD, NZD, SEK, NOK vs USD) from 2000 onward with acceptable data quality for daily-frequency research.
- IBKR TWS or IB Gateway must be running locally for the execution layer to connect; the strategy does not attempt to replace or bypass this requirement.
- The `macro_env` conda environment in `C:\Users\trevm\Projects\macro-dashboard\` is the target Python runtime for this project; no new environment will be created.
- Phase 1 targets a daily-to-weekly rebalancing horizon; intraday execution optimization is out of scope.
- Equity instruments are out of scope for Phase 1; the strategy is FX-only (G10 + select EM as a later extension).
- Fractional Kelly position sizing is targeted for Phase 2; Phase 1 uses equal-notional sizing within the carry/momentum portfolios to isolate signal quality from sizing effects.
- EM FX extension (beyond G10) is deferred to Phase 2 pending validation of the G10 pipeline.
- Live trading (real capital) is out of scope for Phase 1; all execution testing is on the IBKR paper account.
