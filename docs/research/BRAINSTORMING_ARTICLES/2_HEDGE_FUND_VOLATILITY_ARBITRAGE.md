# Volatility Arbitrage: How Funds Profited During the August 2024 VIX Spike | by Navnoor Bawa | Oct, 2025 | Medium

Member-only story

# Volatility Arbitrage: How Funds Profited During the August 2024 VIX Spike

[

![Navnoor Bawa](https://miro.medium.com/v2/da:true/resize:fill:48:48/0*yb5oLibQYMBeYFWY)





](/@navnoorbawa?source=post_page---byline--d31cf0da1949---------------------------------------)

[Navnoor Bawa](/@navnoorbawa?source=post_page---byline--d31cf0da1949---------------------------------------)

Following

9 min read

·

6 days ago

54

Listen

Share

More

## Deconstructing the Largest VIX Move in History — and the Trades That Made Millions

📖 Read this article FREE on Substack: [\[Volatility Arbitrage: How Funds Profited\]](https://open.substack.com/pub/navnoorbawa/p/volatility-arbitrage-how-funds-profited?r=2io3ue&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)

_A quantitative breakdown of P&L generation during market panic_

![](https://miro.medium.com/v2/resize:fit:1500/1*_o2ekzFwhtzUlhcYc0n3FA.jpeg)

## The Event

August 5, 2024 — starting at 3:15 AM Eastern Time and reaching its peak by 8:30 AM — the CBOE Volatility Index recorded its largest single-day spike in history: **180% surge to 65+**, exceeding peaks during both the 2008 financial crisis and March 2020. By market close, it had collapsed to 39.

**The catalyst:** Bank of Japan raised rates 15 basis points on July 31, triggering a violent unwind of yen carry trades. The TOPIX lost 12% on August 5. JP Morgan estimated 65–75% of global carry trade positions unwound by mid-August.

**The opportunity:** This wasn’t a fundamental volatility event. It was a technical dislocation in options market microstructure — and that’s where the money was made.

## Market Structure Breakdown: Understanding the Dislocation

## Why VIX Spiked to Levels Incompatible with Reality

VIX is calculated from mid-values of quoted option prices, not actual trades. When market makers face one-sided order flow in illiquid conditions, they widen bid-ask spreads asymmetrically — particularly for out-of-the-money puts.

**Pre-crisis positioning (per BIS Bulletin 95):**

-   Market makers held short positions in 300,000+ put contracts ($160B notional)
-   Pre-market trading volume: 80x lower than regular sessions
-   VIX methodology assigns larger weights to far OTM puts (inversely proportional to strike²)

**During the spike:**

-   Deep OTM put bid-ask spreads widened to >80% of mid-price (vs. 25% average)
-   These options contributed **86% of the VIX spike**
-   Trading volume remained low — the spike was driven by quote adjustments, not actual trades
-   Market depth evaporated: best bid/ask sizes collapsed to minimal levels

**Critical insight:** The gap between VIX and its nearest futures contract reached a record high. VIX futures remained below 35 while spot VIX calculated from illiquid pre-market SPX options surged above 65. This **31+ point basis blowout** was the primary arbitrage signal.

## Strategy #1: VIX Futures Basis Arbitrage

## Structure

**Long:** VIX futures (underpriced vs. spot)  
**Short:** SPX options portfolio replicating VIX calculation (overpriced)  
**Hedge:** Delta-neutral via S&P 500 futures

## P&L Driver

Pre-market VIX hit 65+ while front-month VIX futures remained below 35 — creating an unprecedented and unsustainable basis exceeding 31 points. Sophisticated vol desks recognized that liquidity would return at the 9:30 AM market open, mechanically collapsing the spread.

**Execution timeline:**

**3:15–8:30 AM:** VIX surge from 23.9 to 65+. Futures remain below 35. Basis exceeds 31 points — a structural anomaly.

**9:30 AM:** “Volume increased massively in regular trading during August 5 and bid-ask spreads came down sharply, indicating an influx of orders to take advantage of the large pre-market bid-ask spreads.” (BIS)

**4:15 PM Close:** VIX at 39. Basis compressed to normal levels (~2–3 points).

## P&L Example

The basis compression on VIX futures (worth $1,000 per point):

-   Spread entry: ~31 points (65 spot minus < 35 futures)
-   Spread exit: ~2 points (normalized)
-   Realized P&L: $29,000 per contract

Scale: 100-contract position = **$2.9M single-session profit**

## Strategy #2: Variance Swap Relative Value

## Opportunity

Variance swaps (contracts on realized volatility) exhibited different dislocation patterns than VIX derivatives. The trade exploited the gap between implied and expected realized volatility.

**Structure:**

**Leg 1:** Long 30-day variance swap at spike levels (50–55 vol)  
**Leg 2:** Short VIX futures  
**Rationale:** VIX implied 65%+ annualized vol for the next 30 days. Historical crisis realized vol rarely exceeds 40–50%.

## The Math

-   S&P 500 realized vol averages: 15–20% (normal), 40–50% (crisis)
-   August 5 VIX implied: 65%+ forward volatility
-   Bet: 30-day realized vol would undershoot dramatically

**Actual outcome:** “As remarkable as the spike itself was how fast it dissipated when markets opened: within few hours, VIX dropped rapidly and finished the trading day around 39.” (BIS)

Subsequent volatility normalized well below crisis levels, with August 2024 monthly realized volatility settling around 19–20% — nowhere near the 65%+ implied at the spike.

## P&L Calculation

Variance notional: $100,000 vega (pays $100K per vol point):

-   Entry: Long variance at 55, short VIX at 65
-   Settlement: Realized vol significantly lower than implied
-   Variance leg P&L: $100,000 × (55–20) = **$3.5M**
-   VIX short leg (collapsed to 15–20 range): Additional **$4–5M**

**Total strategy P&L: $7.5–8.5M on $100K vega position**

## Strategy #3: Dispersion Trading

## Mechanics

Dispersion trades exploit the relationship between index volatility and constituent stock volatility. During panics, correlation spikes temporarily but mean-reverts quickly. Index options become expensive versus single-stock options.

**Structure:**

-   **Sell:** SPX index options (elevated vol)
-   **Buy:** Weighted basket of options on SPX constituents (lower vol)
-   **Greeks:** Delta-neutral, vega-flat

## August 2024 Dynamics

“Implied correlations of the largest S&P 500 index constituents did increase on August 2 and August 5” (BIS), driven by systematic deleveraging forcing indiscriminate selling — not fundamental correlation changes.

**Entry point:**

-   VIX spike drove SPX ATM implied vol to 45–50%
-   Single-stock options (AAPL, MSFT, GOOGL): 35–40% implied vol
-   Index premium: 10–15 vol points

**Mean reversion:** Within 5 trading days, correlations normalized as yen carry trade stabilized.

## P&L Example

-   Notional: $50M
-   Vega exposure: $250K per vol point
-   Entry spread: 12 vol points (index expensive)
-   Exit spread: 4 vol points (normalized)
-   **Profit: $250K × 8 = $2M**

## Strategy #4: Tail Risk Funds — Structured Convexity

## Universa’s Blueprint

Tail-risk fund Universa Investments delivered **100% return in April 2025** and **4,144% YTD return through March 2020** during COVID volatility using systematic OTM put buying.

**Structure:**

-   Systematic purchase of 20–30% OTM SPX puts
-   3–6 month expiries (tail focus)
-   Allocate 2–5% of portfolio to premium
-   Target: Asymmetric payoff (small consistent losses, multiples during spikes)

## August 2024 Execution

**Pre-crisis cost:** 30-delta SPX puts cost ~1.5% of notional quarterly. $1B fund spends $15M/quarter.

**Payoff trigger:** VIX exceeded 65. S&P 500 dropped 4–6% intraday before recovering. Puts struck at 5,000 (when SPX at 5,500) moved from 0.30 delta to 0.70 delta.

**Mark-to-market gains:** 200–400% on position in hours.

## P&L Estimate

-   Allocation: $50M to tail puts (5% of $1B portfolio)
-   MTM gain: 3–5x capital
-   Portfolio-level return: **15–25% single-day**

This is the essence of convexity: small consistent drags, massive asymmetric upside during dislocations.

## The Other Side: Short Vol Destruction

## Who Lost

Funds managing **$21.5B+ in volatility strategies lost up to 40% in a single day** during the August spike.

**The doomed strategy:** Systematic short volatility via VIX ETPs and naked option selling.

**Pre-crisis state:** Traders sold VIX straddles/strangles collecting premium during mid-2024’s low-vol regime (VIX range-bound at 12–15).

Example position:

-   Short VIX 20 calls @ $0.50 premium
-   Short VIX 13 puts @ $0.40 premium
-   Collected: $900 per spread

**August 5 outcome:** VIX gapped to 65+. The 20 calls were 45+ points ITM.

-   MTM loss: $45,000+ per contract
-   **Return on margin: -4,500%+**

## The Feedback Loop

“Market makers had been selling over several previous trading sessions a lot of short-term options, predominantly puts, including those expiring on August 5.” (BIS)

Margin calls forced liquidations → additional option buying → further spread widening → amplified VIX spike.

“Sizes associated with best bid and ask quotes were much smaller than in comparable overnight periods, indicating deteriorating liquidity as market makers managed inventory in a one-sided market.” (BIS)

**Lesson:** Selling volatility without tail hedges or dynamic position sizing is not a strategy. It’s a guaranteed eventual ruin.

## The Yen Carry Trade Catalyst

## Macro Context

**July 31, 2024:** Bank of Japan raised rates to 0.25% from 0–0.1% — an unexpectedly hawkish move ending negative rates implemented in 2016.

**Carry trade mechanics:**

-   Borrow yen at ~0% (pre-hike)
-   Invest in USD assets yielding 5%+ (Treasuries, equities, HY credit)
-   Collect 5%+ net carry
-   Leverage 5–10x for 25–50% annual returns

**The unwind:**

-   BoJ hikes → yen strengthens rapidly (USD/JPY dropped from 149.94 on July 31 to 143.89 on August 5)
-   Leveraged positions face FX losses: 4%+ loss × 5–10x leverage = 20–40% drawdowns
-   Forced selling of U.S. equities/bonds to cover yen liabilities
-   Cross-market contagion: equities down, volatility up, credit spreads widen

**Why this mattered for vol trading:** Smart money recognized this wasn’t a U.S. credit event or earnings recession — it was a pure liquidity and positioning shock. That’s precisely when implied vol exceeds expected realized vol, creating mean-reversion trades.

## Key Principles: Extracting Tradeable Insights

## 1\. Methodology Matters More Than Headlines

VIX calculation is based on quotes, not trades, making it “vulnerable to bid-ask spread widening regardless of a fundamental rise in underlying volatility” (BIS). Pre-market readings are structurally less reliable due to thin liquidity (80x lower volume).

**Trade the dislocation, not the headline number.**

## 2\. Basis Trades Are Mechanical Mean Reversion

When spot VIX diverges from VIX futures by 30+ points, arbitrageurs enter. VIX futures remained below 35 while spot surged above 65 — “giving rise to a price dislocation between the less liquid S&P options markets, underlying the spot VIX calculation, and the more liquid VIX futures market.” (BIS)

The compression is fast and predictable. This is structural, not discretionary.

## 3\. Differentiate Technical from Fundamental Volatility

“The spike in VIX was not accompanied by a commensurate increase in other volatility measures. In particular, volatility of at-the-money options also did not spike as much.” (BIS)

When VIX diverges from ATM volatility and realized vol expectations, it signals technical anomaly, not fundamental repricing. This is the edge.

## 4\. Cross-Asset Contagion Creates Multi-Leg Alpha

The yen carry unwind generated opportunities across:

-   **FX options:** Long JPY vol
-   **Equity vol:** Long VIX, SPX puts
-   **Rates vol:** Short JGB vol (BoJ tightening)
-   **Credit:** Long credit spreads (deleveraging widens HY)

Single-asset strategies miss the interconnected playbook. Real edge comes from mapping contagion pathways.

## 5\. Market Maker Positioning Predicts Mean Reversion

“Market makers had positive options gamma exposure at the start of the regular trading session on August 5” (BIS), meaning they would dampen volatility by buying weakness and selling strength.

Dealer positioning (via options flow data and gamma exposure metrics) provides timing edge on mean reversion trades.

## 6\. Short Vol Without Hedges = Guaranteed Ruin

Short volatility generates positive carry until it doesn’t. “Funds managing over $21.5 billion lost up to 40% in a single day.”

Selling options without tail hedges, dynamic scaling, and stress-tested risk limits guarantees eventual annihilation. The math is undefeated.

## Conclusion: Alpha Lives in Market Plumbing

The August 2024 VIX spike wasn’t a Black Swan. It was a documented confluence of:

1.  **Macro trigger:** Yen carry unwind (BoJ rate hike)
2.  **Microstructure vulnerability:** VIX calculation from illiquid pre-market quotes
3.  **Positioning crowding:** Short vol, leveraged carry trades

Funds that profited understood three things:

**Where the dislocation occurred:** Spot VIX vs. futures (31+ point basis), index vol vs. single-stock vol, implied vol vs. expected realized vol

**Why it was unsustainable:** Liquidity would return at market open, spreads would compress mechanically

**How to size and time entry:** Pre-market chaos → regular session mean reversion with precise risk management

This is quantitative research: identifying structural inefficiencies, modeling expected outcomes, executing with precision. The money isn’t in predicting _if_ markets will panic — it’s in knowing _how_ they panic, and where technical dislocations create asymmetric opportunities.

When the next volatility event hits, winners won’t be those betting on direction. They’ll be those who understand options pricing formulas, market maker inventory dynamics, and cross-asset feedback loops that transform panic into profit.

## References & Data Sources

**Primary Technical Analysis:**

-   Bank for International Settlements. (2024). _BIS Bulletin No 95: Anatomy of the VIX spike in August 2024_. Todorov, K. & Vilkov, G. [https://www.bis.org/publ/bisbull95.pdf](https://www.bis.org/publ/bisbull95.pdf)
-   U.S. Securities and Exchange Commission. (2025). _DERA Working Paper: VIX Spike Analysis August 2024_. [https://www.sec.gov/files/dera-vix-working-paper-2504.pdf](https://www.sec.gov/files/dera-vix-working-paper-2504.pdf)
-   Bank for International Settlements. (2024). _BIS Bulletin No 90: The market turbulence and carry trade unwind of August 2024_. Aquilina, M. et al. [https://www.bis.org/publ/bisbull90.pdf](https://www.bis.org/publ/bisbull90.pdf)

**Market Data & Analysis:**

-   J.P. Morgan Private Bank. (2024). _Amid rate cuts, do carry trades still work_. August 2024.
-   Bloomberg. (2024). _JPMorgan Says Three Quarters of Global Carry Trades Now Unwound_. August 8, 2024.
-   Reuters. (2024). _Bank of Japan to outline bond taper plan, debate rate hike timing_. July 30, 2024.

**Fund Performance:**

-   Hedgeweek. (2025). _Black swan hedge fund Universa up 100% amid April volatility_. May 2025.
-   Bloomberg. (2020). _Taleb-Advised Universa Tail Fund Returned 3,600% in March_. April 8, 2020.
-   Bawa, N. (2025). _The $1 Billion VIX Gamble: Inside Hedge Fund Volatility Strategies_. Medium.

**Exchange Rate Data:**

-   Exchange-Rates.org. (2024). _USD to JPY Historical Exchange Rates_. August 2024 data.

**Methodology Reference:**

-   CBOE. (2022). _White Paper: Cboe Volatility Index_. Chicago Board Options Exchange.

_This article is part of a quantitative research series deconstructing real trades executed by hedge funds. Core focus: understanding P&L drivers and extracting actionable principles from market dislocations._

_For technical discussions on volatility trading, market microstructure, or quantitative strategy development, connect on_ [_LinkedIn._](https://www.linkedin.com/in/navnoorbawa/)

## Embedded Content

---