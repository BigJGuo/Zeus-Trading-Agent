# Trading principles (distilled from the research library)

These are the empirical regularities your decisions should rest on. Cite a
principle (not a paper) in every `propose_trade` rationale. When a setup
doesn't fit anything below, that is itself information — be skeptical.

## 1. Momentum

- **Cross-sectional momentum**: ranks stocks by their 3–12 month return,
  skip the last 1 month, and go long the top decile. Robust over 100+
  years and across asset classes (Jegadeesh & Titman; Moskowitz, Ooi &
  Pedersen; Hurst, Ooi & Pedersen).
- **Time-series momentum (TSMOM)**: a stock's own past 12-month return
  (skip 1m) predicts its next month — even controlling for cross-section.
- **Industry momentum drives most single-stock momentum** (Moskowitz &
  Grinblatt). If a name is up only because its industry is up, you don't
  have an *additional* signal — you have one signal in two clothes.
- **Factor momentum is real**: factors that outperformed last year tend
  to outperform next year (Ehsani & Linnainmaa). The factor itself is
  a tradable timing signal, not just stocks.
- **Momentum CRASHES asymmetrically.** When the market pivots from bear
  to bull, past losers (high-beta junk) violently outperform past winners.
  This is a left-tail risk specific to momentum. Defenses:
  (a) **vol-scale** momentum exposure (lower size when momentum's own
  realized vol is rising); (b) **regime filter** — cut momentum exposure
  near regime transitions; (c) accept that as long-only you get less of
  this protection than a long-short fund does.

## 2. Mean reversion

- **1-month reversal**: last month's biggest winners underperform next
  month; last month's biggest losers outperform (Jegadeesh 1990; Lehmann
  1990). Mostly a microstructure / liquidity-rebate effect — but it works.
- The danger zone is **the gap between reversal (≤1 month) and momentum
  (3–12 months)**: 1–3 month windows are noisy. Don't take swing-horizon
  trades on signals that live in this window unless you have a
  catalyst-driven reason.
- **Long-horizon mean reversion**: equity returns are less risky over
  10y+ than per-period vol implies (negative autocorrelation at long
  horizons). Translate this into: drawdowns mean-revert; don't panic-
  exit a high-quality long_term name just because it's drawn down 15%.

## 3. Quality / fundamentals (Fama–French 5-factor world)

- The Fama–French 5-factor model spans most US equity cross-section:
  market, size (SMB), value (HML), **profitability (RMW)**, **investment
  (CMA)**. Last two are the key extensions.
- **Gross profitability (gross profits / assets) is as strong as B/M**
  (Novy-Marx). Prefer high-gross-margin firms; "growth at quality" beats
  raw growth.
- **High asset growth → low future returns** (Cooper, Gulen, Schill).
  Empire-building destroys per-share value; M&A binges are a yellow flag.
- **High accruals → low future returns** (Sloan, the accrual anomaly).
  Earnings driven by accruals (vs cash flows) don't persist. Watch the
  CFO/NI ratio.
- **Avoid distress** (Campbell, Hilscher, Szilagyi). Distressed firms
  underperform their CAPM-implied return — the "distress anomaly" is
  asymmetric. If a long_term name's Altman-Z is deteriorating, exit.

## 4. Volatility, beta, sizing

- **Low-vol / low-beta anomaly**: high-idiosyncratic-vol stocks
  underperform (Ang, Hodrick, Xing, Zhang). "Betting against beta"
  works because leverage-constrained investors bid up high-beta names.
  Practical implication: **prefer the low-vol name within a sector**
  when you have a tie.
- **Vol-scale your sizing.** Target constant *volatility per name*, not
  constant *dollars per name*. A $10k position in a 60-vol biotech is
  not the same risk as $10k in WMT. The risk engine enforces dollar
  caps; you should propose sizes that already reflect realized vol.
- **Evaporating liquidity**: in stress, bid/ask widens and ADV drops
  before prices move. Notional caps relative to **20-day ADV** are the
  first line of defense — never propose a notional > 5% of 20d ADV
  for a swing/day name; long_term can go a bit higher with patience.

## 5. Pairs and relative-value

- **Cointegration-based pairs trading** has historical edge but Sharpe
  has decayed with HFT competition (Gatev, Goetzmann, Rouwenhorst).
  Today's edge is in **regime-aware pairs**: spreads widen unpredictably
  in stress regimes (regime-switching relative-value literature). Don't
  hold a stat-arb pair through a regime change without a tighter stop.
- This system is long-only, so pure pairs trades aren't allowed. But
  pairs **logic is useful** for selection: if you're long AAPL on a
  thesis, ask "what's the cheaper-relative analog inside the same
  industry-momentum cluster?"

## 6. Events and announcement drift

- **PEAD (post-earnings announcement drift)**: stocks drift in the
  direction of earnings surprises for 60+ trading days after the
  announcement (Bernard & Thomas). This is actionable on **swing**
  horizon — a positive surprise + positive guidance is a 1–2 week
  lean-in setup with a clear stop.
- **Post-FOMC drift**: bond markets drift after FOMC (Brusa, Savor,
  Wilson); equity vol regime shifts around FOMC days. Cut risk into
  FOMC; trade reactions, not anticipations.

## 7. Information / sentiment signals

- **Short interest + institutional ownership**: high SI paired with
  low institutional ownership is a bearish setup. As a long-only fund,
  treat this as an **avoid** signal, not a short signal.
- **Options info content**: unusual option volume and call/put OI
  imbalances precede stock moves. Spike in call OI + rising stock = a
  positive signal; spike in put OI + falling stock = warning.
- **Machine forecast disagreement is itself a risk premium**
  (Atmaz et al.; ML disagreement literature) — high disagreement
  across models means high expected return *and* high risk; don't
  size up just because your one model is bullish if other models split.

## 8. Risk management

- **Regime first, signal second.** Most factors are regime-dependent.
  Momentum loves trending markets, hates pivots. Value loves recoveries,
  hates panics. Always know what regime the overseer is reporting.
- **Tail risk dominates outcomes.** A handful of crash days dominate
  long-horizon Sharpe. Stops, vol scaling, and drawdown caps are not
  optional — they buy you the right to compound.
- **Capacity is a risk.** A strategy works until it's crowded.
  ADV-relative sizing prevents the worst version of this.
- **Don't rebuild a thesis to keep a position.** If your stop is hit,
  exit and write a `lesson` row. Re-entry requires a new brief.

## 9. Long-only adjustment factor

- All the long-short anomalies above have two legs. As a long-only
  fund, **you capture only the long leg** — historically about half
  the academic Sharpe of the long-short paper.
- Translate "short the losers" research findings into:
  (a) **avoid** them (negative-screening filter on the universe), and
  (b) **earlier exit** when a held name turns into a "loser" by the
  same anomaly's definition.
- Don't forecast double-Sharpe and be surprised when you get half.

## 10. ML / RL guidance (relevant for the predictor models + the bandit)

- **Stat-arb with RL** can find non-stationary alpha but only with
  careful reward shaping (risk-adjusted, not raw PnL) and explicit
  exploration controls. Online learning on live capital without a
  parallel sim is unsafe.
- **DRL for stock trading** (FinRL et al.): policy gradients on raw
  price overfit. Use **feature-engineered state** (the same factors
  above) and **risk-adjusted reward** (Sharpe, drawdown-penalized PnL),
  not naked PnL.
- **Limit-order-book deep models** (DCNNs, market-making with Hawkes
  processes) are not applicable to this paper-trading equity swing
  system — they require HFT infrastructure and microstructure access.
- **Universal features of price formation** (Sirignano-Cont):
  power-law market impact, fat tails, volatility clustering, cross-
  sectional return correlations. These are stylized facts to *model*,
  not to ignore. Sizing decisions that assume normal returns are
  systematically miscalibrated.

## The research library

Every principle above is grounded in a specific paper. The actual
PDF excerpts (title page + abstract + first body section + slice of
conclusion) are persisted in the journal under
`agent_id='research_library', kind='paper'`. You can pull them into
your reasoning when you need primary-source detail:

```
query_journal(agent_id='research_library', kind='paper', limit=5)
```

You can also filter by symbol if a paper happens to reference one
(rare). The library is read-only — you cannot write to it. Use it
when:
- You need to ground a thesis in primary research before sizing it big.
- You want to verify the empirical horizon, asset class, or sample
  period of a finding before extrapolating it.
- The model prediction disagrees with your factor intuition and you
  want to check which way the literature actually leans.

Don't dump the entire library into your context every run — that's
expensive and noisy. Pull the 1-3 papers most relevant to the
specific decision you're making.

## How to use this

1. **Anchor every entry rationale in a principle.** "Long XYZ on
   3–12 month cross-sectional momentum + gross profitability filter"
   is a defensible rationale. "I have a feeling about XYZ" is not.
2. **Don't double-count signals.** Single-stock momentum + industry
   momentum + factor momentum on the same name is *one* signal viewed
   from three angles, not three independent confirmations.
3. **Respect the asymmetry of long-only**: more conservative in late-
   cycle / regime-pivot windows since you don't have the short-leg
   hedge.
4. **Vol-scale.** Sizes you propose should already reflect realized
   vol; don't dump that on the risk engine.
5. **When in doubt, smaller.** Capacity, liquidity, and tail risk all
   penalize overconfidence in size more than in selection.
