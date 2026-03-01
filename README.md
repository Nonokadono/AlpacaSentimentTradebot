# Trade Bot Master — Alpaca Sentiment-Scaled Algo Trading Bot

A professional-grade algo-trading bot built on [Alpaca](https://alpaca.markets) that:

1. **Builds a portfolio** each loop cycle by scoring every whitelisted instrument with a composite signal (momentum/trend + mean-reversion + price-action), ranks candidates by `|signal_score|`, and selects trades greedily until gross exposure and position-count limits are hit.
2. **Executes each selected trade as a bracket order** (market entry + attached stop-loss + take-profit) via Alpaca's bracket order API.
3. **Scales position size** per trade using sentiment from Perplexity Sonar and a volatility-based stop distance.

---

## Key Features

- **`ENV_MODE` switch** (`PAPER` / `LIVE`) driven purely by environment variables — zero code changes to promote to live
- **AI-powered sentiment** via Perplexity Sonar (`sonar` model) scoring news `{-2, -1, 0, 1}` with confidence
- **Composite technical signal** — momentum/trend (50%), mean-reversion RSI+MA (30%), price-action breakout+candles (20%)
- **Portfolio builder** — ranks all feasible candidates by `|signal_score|`, adds greedily while respecting the 90% gross exposure cap and max open positions limit
- **Risk-based position sizing** — 0.5–1% equity at risk per trade, scaled by sentiment score
- **Optional Sonar portfolio veto** — a second AI pass that can block individual trades at the portfolio level
- **Bracket orders** — market entry with attached stop-loss and take-profit submitted as a single Alpaca bracket order
- **Kill-switch** — halts new trades on daily loss ≥ 4% or drawdown ≥ 9% from high watermark
- **Sentiment-triggered position close** — existing positions can be closed immediately if sentiment flips to strongly negative
- **Persistent equity state** — start-of-day equity and high watermark survive restarts via `data/equity_state.json`
- **Coloured structured logging** — per-instrument signal, sentiment, proposed trade, portfolio overview, and kill-switch events

---

## Requirements

Python **3.9+** recommended.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

All credentials and mode flags are injected via environment variables — **never hard-coded**.

### Alpaca credentials

| Variable | Description |
|---|---|
| `APCA_API_KEY_ID` | Your Alpaca API key ID |
| `APCA_API_SECRET_KEY` | Your Alpaca API secret key |
| `APCA_API_BASE_URL` | API gateway URL (see below) |
| `APCA_API_ENV` | `PAPER` (default) or `LIVE` |
| `LIVE_TRADING_ENABLED` | `true` to submit real orders in LIVE mode (default `false`) |

### AI / Sentiment

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Perplexity API key for the Sonar sentiment model |

### PAPER mode (default)

```bash
export APCA_API_ENV=PAPER
export APCA_API_BASE_URL=https://paper-api.alpaca.markets
export APCA_API_KEY_ID=your_paper_key
export APCA_API_SECRET_KEY=your_paper_secret
export GEMINI_API_KEY=your_gemini_key
```

### LIVE mode

```bash
export APCA_API_ENV=LIVE
export APCA_API_BASE_URL=https://api.alpaca.markets
export APCA_API_KEY_ID=your_live_key
export APCA_API_SECRET_KEY=your_live_secret
export GEMINI_API_KEY=your_gemini_key
export LIVE_TRADING_ENABLED=false   # keep false for dry-run; set true only after sign-off
```

### Smoke test

```bash
export MANUAL_APPROVE_SMOKE=true   # set to actually submit the smoke-test bracket order
```

---

## Project Structure

```
.
├── adapters/
│   └── alpaca_adapter.py          # Alpaca REST adapter (account, bars, news, bracket orders)
├── ai_client.py                   # Perplexity Sonar news sentiment scorer (NewsReasoner)
├── config/
│   ├── config.py                  # BotConfig dataclass + loader; risk, sentiment, technical configs
│   └── instrument_whitelist.yaml  # Whitelisted symbols with exchange, lot size, flags
├── core/
│   ├── portfolio_builder.py       # Portfolio-level candidate ranking and exposure management
│   ├── portfolio_veto.py          # Optional Sonar-based portfolio veto layer
│   ├── risk_engine.py             # Pre-trade risk checks, position sizing, sentiment scaling
│   ├── sentiment.py               # SentimentModule wrapping NewsReasoner → SentimentResult
│   └── signals.py                 # SignalEngine: composite momentum + mean-reversion + price-action
├── execution/
│   ├── order_executor.py          # Bracket order submission + sentiment-triggered position close
│   └── position_manager.py        # Normalises Alpaca positions into PositionInfo dicts
├── monitoring/
│   ├── kill_switch.py             # KillSwitch: daily loss and drawdown halt logic
│   └── monitor.py                 # Structured coloured logging for all bot events
├── tests/
│   ├── __init__.py
│   └── test_smoke_paper.py        # "Hello Alpaca" PAPER smoke test
├── data/
│   └── equity_state.json          # Persisted start-of-day equity + high watermark (auto-created)
├── main.py                        # Main trading loop (60-second poll)
├── requirements.txt
└── README.md
```

---

## How It Works

### Main loop (`main.py`)

Every **60 seconds**:

1. Fetch account, build `EquitySnapshot`, and normalise open positions from Alpaca
2. Check kill-switch (daily loss / drawdown) — halt the cycle if triggered
3. Check gross exposure — skip new trade evaluation if already at the 90% cap
4. Run `PortfolioBuilder` across all whitelisted symbols:
   - Generate a composite technical signal (`signal_score ∈ [-1, 1]`) per symbol
   - Fetch and score news sentiment via Perplexity Sonar
   - Run full pre-trade risk checks (whitelist, sentiment block, stop distance, sizing, exposure, limits)
   - Rank all feasible trades by `|signal_score|` descending
   - Greedily select trades while projected gross exposure ≤ 90% equity and open positions ≤ max
   - Optionally apply Sonar portfolio veto (`enable_portfolio_veto: true`)
5. Submit a **bracket order** for each selected trade via `OrderExecutor`
6. Optionally close existing positions where sentiment has flipped strongly negative

### Signal generation (`core/signals.py`)

Each instrument is scored on three components that are weighted and summed into a composite:

| Component | Weight | Inputs |
|---|---|---|
| Momentum / trend | 50% | Price momentum normalised by ATR, EMA trend direction |
| Mean reversion | 30% | RSI overbought/oversold, price distance from moving average |
| Price action | 20% | Breakout above/below N-bar high/low, candle pattern score |

A composite score ≥ `long_threshold` (0.2) → **BUY**; ≤ `short_threshold` (-0.2) → **SELL**; otherwise **skip**.

### Order execution (`execution/order_executor.py`)

For each approved `ProposedTrade`, `OrderExecutor.execute_proposed_trade()` calls `AlpacaAdapter.submit_bracket_order()`, which submits a single Alpaca bracket order:

```
Market entry  →  attached take-profit (limit)
             →  attached stop-loss (stop-limit)
```

In **LIVE** mode with `LIVE_TRADING_ENABLED=false`, all logic runs but order submission is skipped (dry-run / connectivity validation only).

---

## Configuration

### Risk limits (`config/config.py` — `RiskLimits`)

| Parameter | Default | Description |
|---|---|---|
| `max_risk_per_trade_pct` | 1.0% | Max equity risked at the stop per trade |
| `min_risk_per_trade_pct` | 0.5% | Min equity risked (floor after sentiment scaling) |
| `gross_exposure_cap_pct` | 90% | Portfolio gross notional cap |
| `daily_loss_limit_pct` | 4% | Daily loss that triggers kill-switch halt |
| `max_drawdown_pct` | 9% | Drawdown from high watermark that triggers halt |
| `max_open_positions` | 15 | Maximum simultaneous open positions |

### Sentiment config (`SentimentConfig`)

| Parameter | Default | Description |
|---|---|---|
| `neutral_band` | 0.1 | Scores within ±0.1 → scale factor = 1.0 |
| `min_scale` | 0.2 | Minimum size scale factor from sentiment |
| `max_scale` | 1.3 | Maximum size scale factor from sentiment |
| `no_trade_negative_threshold` | -0.4 | Scores below this → zero size (no trade) |

Discrete sentiment `-2` (unstable / extreme risk) **always hard-blocks** the trade regardless of these thresholds.

### Technical signal config (`TechnicalSignalConfig`)

| Parameter | Default | Description |
|---|---|---|
| `weight_momentum_trend` | 0.5 | Weight of momentum + trend component |
| `weight_mean_reversion` | 0.3 | Weight of RSI + MA-distance mean-reversion |
| `weight_price_action` | 0.2 | Weight of breakout + candle pattern score |
| `long_threshold` | 0.2 | Composite score ≥ this → BUY signal |
| `short_threshold` | -0.2 | Composite score ≤ this → SELL signal |
| `base_stop_vol_mult` | 1.5× | Stop distance = this × ATR-normalised volatility × price |
| `base_tp_vol_mult` | 3.0× | Take-profit distance base (scaled by signal strength) |

### Instrument whitelist (`config/instrument_whitelist.yaml`)

Add or uncomment symbols to expand the tradeable universe. Each entry requires:

```yaml
AAPL:
  exchange: NASDAQ
  lot_size: 1
  fractional: true
  shortable: true
  marginable: true
  trading_hours: "09:30-16:00"
  sector: TECH          # optional
```

Currently **active** symbols (uncommented): `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`, `AMD`, `MU`, `NFLX`, `SPY`, `QQQ`, `VOO`, `IWM`, `XLF`, `XLE`, `SLV`, `EEM`, `BAC`, `COST`

---

## Sentiment Engine

News is fetched from Alpaca's news API per symbol (up to 20 items, since last poll or last 6 hours). `NewsReasoner` (`ai_client.py`) sends headlines and summaries to the Perplexity `sonar` model, which returns a discrete score plus confidence:

| Discrete score | Meaning | Continuous mapping |
|---|---|---|
| `-2` | Extremely unstable / do not trade | Hard block — zero size regardless of confidence |
| `-1` | Clearly negative | `score = -1.0 × confidence` |
| `0` | Neutral / mixed | `score ≈ 0` |
| `1` | Clearly positive | `score = +1.0 × confidence` |

The continuous score `s ∈ [-1, 1]` feeds `RiskEngine.sentiment_scale()`, which outputs a multiplier applied to the per-trade risk budget (clamped between `min_scale` and `max_scale`).

---

## Kill-Switch Behaviour

| Trigger | Action |
|---|---|
| `daily_loss_pct ≤ -4%` | Halt new trades for the session; require manual reset |
| `drawdown_pct ≤ -9%` | Halt and require human review before restart |

Kill-switch state is evaluated every loop iteration **before** any signal generation or order placement. Behaviour is **identical in PAPER and LIVE modes**.

---

## Running the Smoke Test (PAPER only)

```bash
export APCA_API_ENV=PAPER
export APCA_API_BASE_URL=https://paper-api.alpaca.markets
export APCA_API_KEY_ID=your_paper_key
export APCA_API_SECRET_KEY=your_paper_secret
export AI_API_KEY=your_perplexity_key

# Read-only — no order placed
python -m tests.test_smoke_paper

# With order submission enabled
export MANUAL_APPROVE_SMOKE=true
python -m tests.test_smoke_paper
```

The smoke test:
1. Connects to Alpaca PAPER and fetches account summary + positions
2. Builds an `EquitySnapshot` and verifies the kill-switch is not triggered
3. Generates a composite technical signal for the first whitelisted symbol
4. Scores news sentiment via Perplexity Sonar
5. Runs full `pre_trade_checks` through the risk engine
6. Logs the proposed bracket order
7. Optionally submits it if `MANUAL_APPROVE_SMOKE=true` and the trade passes all checks

---

## Promoting from PAPER to LIVE

Switching takes **under 10 minutes** — only environment variables change:

```bash
# 1. Update env vars
export APCA_API_ENV=LIVE
export APCA_API_BASE_URL=https://api.alpaca.markets
export APCA_API_KEY_ID=your_live_key
export APCA_API_SECRET_KEY=your_live_secret

# 2. Start in dry-run (no orders placed)
export LIVE_TRADING_ENABLED=false
python main.py

# 3. After confirming connectivity and data, enable live orders
export LIVE_TRADING_ENABLED=true
python main.py   # restart required
```

> **Important:** The environment switch is logged with timestamp at startup. In LIVE mode with `LIVE_TRADING_ENABLED=false`, the full signal + risk + portfolio-build pipeline runs but order submission is skipped — use this for pre-flight validation.

---

## Safety Notes

- Extensively forward-test in PAPER before setting `LIVE_TRADING_ENABLED=true`
- `data/equity_state.json` persists the high watermark and start-of-day equity across restarts — back this up or migrate to a database for production
- Never hard-code API keys; always use environment variables or a secrets manager
- Review and tighten risk limits in `config/config.py` before going live
- `enable_portfolio_veto: false` by default — enable only after validating Sonar veto quality in PAPER
- Extend the instrument whitelist incrementally; validate each new symbol in PAPER first
