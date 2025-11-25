# Build an AI Stock Analysis Agent: A Step-by-Step Python Guide for 2025 | by Huzaifa Zahoor | Sep, 2025 | Meyka Finance Wire

Member-only story

# Build an AI Stock Analysis Agent: A Step-by-Step Python Guide for 2025

[

![Huzaifa Zahoor](https://miro.medium.com/v2/resize:fill:48:48/1*pVo4VTFsY8arJFPCAHGguA.png)





](/@huzaifazahoor654?source=post_page---byline--82f2bf4b9f66---------------------------------------)

[Huzaifa Zahoor](/@huzaifazahoor654?source=post_page---byline--82f2bf4b9f66---------------------------------------)

Follow

7 min read

·

Sep 10, 2025

91

Listen

Share

More

Fundamental data and charts are complex. What if an AI could analyze them for you and explain it in plain English?

In this guide, you’ll build a Python agent that takes a stock ticker, fetches real-time data and headlines, and asks a large language model (LLM) to generate a structured report. The output includes a **plain-English summary**, **bullish/bearish signals** tied to the data, and a **suggested price target** — all in clean JSON you can store, render, or send anywhere.

> And if you’d rather skip the coding and use a live version today, check out m[**eyka.com**](https://meyka.com/?utm_source=chatgpt.com) — it’s the production-ready platform powered by the same AI stock analysis agent we’re about to create.

**Tech stack preview:** Python, `openai` (LLM), `yfinance` (data), `requests`, and `pandas`. We’ll keep the code simple, extensible, and production-friendly.

![](https://miro.medium.com/v2/resize:fit:1050/0*_1Ew7PkYnf7MbboS)
Photo by Emiliano Vittoriosi on Unsplash

## What You’ll Build (At a Glance)

-   **Input:** stock ticker (e.g., `AAPL`)
-   **Data pulled:** price, day high/low, volume, 50/200-day SMAs, recent news headlines
-   **AI output (JSON):**

{   "summary": "…",   "bullish\_signals": \["…"\],   "bearish\_signals": \["…"\],   "price\_target": { "base": 205.0, "range": \[190.0, 220.0\], "time\_horizon\_days": 90 },   "confidence": 0.68,   "sources": \["…"\] }

## System Requirements & Tooling

-   **Python:** 3.10+ recommended
-   **API Keys:** OpenAI API key (environment variable `OPENAI_API_KEY`)
-   **Rate limits:** Yahoo Finance (`yfinance`) may throttle; add caching and retry logic where possible. (Community notes highlight occasional rate-limit spikes.) [GitHub](https://github.com/ranaroussi/yfinance/issues/2422?utm_source=chatgpt.com)

## Setup & Installation (pip/uv)

Install the essentials:

pip install openai requests pandas yfinance

> _You’ll also need an_ **_OpenAI API key_** _from your_ [_OpenAI dashboard_](https://platform.openai.com/?utm_source=chatgpt.com) _and to set it in your environment:_

\# macOS/Linux  
export OPENAI\_API\_KEY="sk-..."  
\# Windows PowerShell  
setx OPENAI\_API\_KEY "sk-..."

For the latest client usage patterns (Responses API and model selection), see OpenAI’s official docs. [OpenAI Platform+1](https://platform.openai.com/?utm_source=chatgpt.com)

## \[Focus\] Build an AI Stock Analysis Agent: A Step-by-Step Python Guide for 2025

At a high level:

1.  Pull data for a ticker with `yfinance`.
2.  Compute simple technical context (SMA50/200).
3.  Grab recent headlines to ground the LLM’s reasoning.
4.  Craft an **analyst-style prompt** with strict JSON output.
5.  Call the **OpenAI API** in JSON mode.
6.  Parse and render the result.

## Step 1 — Fetch Real-Time Market Data with `yfinance`

We’ll fetch **current price, day high/low, volume**, and a short **history** to compute SMA50/200.

import os, time, json, math  
import pandas as pd  
import yfinance as yf  
  
def fetch\_market\_snapshot(ticker: str, period="1y", interval="1d"):  
    tk = yf.Ticker(ticker)  
    info = tk.fast\_info  \# fast price/volume snapshot  
    hist = tk.history(period=period, interval=interval)  
  
    \# Compute SMA50 / SMA200  
    hist\["SMA50"\]  = hist\["Close"\].rolling(window=50).mean()  
    hist\["SMA200"\] = hist\["Close"\].rolling(window=200).mean()  
  
    snapshot = {  
        "ticker": ticker.upper(),  
        "price": float(info\["last\_price"\]) if "last\_price" in info else float(hist\["Close"\].iloc\[-1\]),  
        "day\_high": float(info.get("day\_high", float("nan"))),  
        "day\_low": float(info.get("day\_low", float("nan"))),  
        "volume": int(info.get("last\_volume", 0)),  
        "sma50": float(hist\["SMA50"\].iloc\[-1\]) if not math.isnan(hist\["SMA50"\].iloc\[-1\]) else None,  
        "sma200": float(hist\["SMA200"\].iloc\[-1\]) if not math.isnan(hist\["SMA200"\].iloc\[-1\]) else None,  
    }  
    return snapshot, hist

### Add Market News

`yfinance` exposes recent headlines via `Ticker.news` in modern versions. If it’s unavailable or limited in your environment, consider a **Market News API** (e.g., Alpha Vantage’s news endpoint) as a fallback. [PyPI](https://pypi.org/project/yfinance/?utm_source=chatgpt.com)[Alpha Vantage](https://www.alphavantage.co/?utm_source=chatgpt.com)

import datetime as dt  
  
def fetch\_news\_headlines(ticker: str, limit=6):  
    headlines = \[\]  
    try:  
        tk = yf.Ticker(ticker)  
        \# Newer yfinance versions often provide a list of dicts with 'title' and 'link'  
        for n in (tk.news or \[\])\[:limit\]:  
            headlines.append({  
                "title": n.get("title", "")\[:160\],  
                "url": n.get("link") or n.get("providerPublishTime") or "",  
            })  
    except Exception:  
        pass  \# fallback below  
  
    \# Optional: Alpha Vantage Market News (requires ALPHA\_VANTAGE\_API\_KEY)  
    if len(headlines) == 0 and os.getenv("ALPHA\_VANTAGE\_API\_KEY"):  
        import requests  
        params = {  
            "function": "NEWS\_SENTIMENT",  
            "tickers": ticker.upper(),  
            "apikey": os.getenv("ALPHA\_VANTAGE\_API\_KEY"),  
            "sort": "LATEST",  
            "limit": limit  
        }  
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)  
        if r.ok:  
            data = r.json().get("feed", \[\])\[:limit\]  
            for item in data:  
                headlines.append({  
                    "title": item.get("title", "")\[:160\],  
                    "url": item.get("url", "")  
                })  
    return headlines

> _Alpha Vantage provides real-time/historical data, 60+ technical indicators, and a Market News API — helpful when Yahoo headlines are sparse._ [_Alpha Vantage+1_](https://www.alphavantage.co/?utm_source=chatgpt.com)

## Step 2 — Design the AI Analyst Prompt

The LLM must behave like a disciplined **equity analyst**:

-   Use only provided data/news — no guessing.
-   Produce **JSON only**.
-   Clearly separate **bullish** vs **bearish** signals tied to data points (e.g., price vs SMA200, volume surges, headline catalysts).
-   Include a **base price target**, a **range**, and a **time horizon (days)**.
-   Add a **confidence** score (0–1).

### Prompt Template (Multi-Shot)

SYSTEM\_PROMPT = """You are a cautious equity research assistant.  
\- Use only the supplied market data and headlines as evidence.  
\- Cite evidence briefly in parentheses (e.g., "Price above SMA200 (152 > 148)").  
\- Output VALID JSON only that matches the provided schema.  
\- If data is insufficient, say so in 'summary' and set 'confidence' <= 0.4.  
\- You are NOT a financial advisor. This is for educational purposes only."""  
  
USER\_TEMPLATE = """Analyze the following context for ticker {ticker}:  
  
\[MARKET\_SNAPSHOT\]  
{snapshot\_json}  
  
\[HEADLINES\] (most recent first)  
{headlines\_bulleted}  
  
Return JSON with keys:  
summary (str),  
bullish\_signals (list\[str\]),  
bearish\_signals (list\[str\]),  
price\_target (object: base \[float\], range \[low, high\], time\_horizon\_days \[int\]),  
confidence (float 0-1),  
sources (list\[str\] of URLs used, if any)  
  
Constraints:  
\- JSON only, no markdown.  
\- Be concise but specific.  
"""  
  
\# Optional few-shot exemplars to steer style  
FEW\_SHOT\_USER = """Analyze:  
\[MARKET\_SNAPSHOT\]  
{"ticker":"MSFT","price":410.2,"day\_high":412.0,"day\_low":405.1,"volume":21834567,"sma50":402.1,"sma200":370.4}  
\[HEADLINES\]  
\- "Azure growth accelerates as AI demand surges" (https://example.com/msft1)  
\- "PC recovery lifts Windows OEM revenue" (https://example.com/msft2)  
"""  
FEW\_SHOT\_ASSISTANT = """{  
  "summary": "Momentum remains constructive: price above SMA50/200 (410>402>370) with AI tailwinds from Azure demand.",  
  "bullish\_signals": \[  
    "Price above SMA200 and SMA50 (410.2 > 402.1 > 370.4)",  
    "AI-related revenue drivers (Azure growth)"  
  \],  
  "bearish\_signals": \[  
    "Short-term overextension risk near day high (412.0)"  
  \],  
  "price\_target": {"base": 430.0, "range": \[400.0, 450.0\], "time\_horizon\_days": 90},  
  "confidence": 0.71,  
  "sources": \["https://example.com/msft1","https://example.com/msft2"\]  
}"""

## Step 3 — Call the OpenAI API (Python)

OpenAI’s Python SDK supports **JSON-mode** responses suitable for programmatic parsing. Below uses the modern client and Responses-style call pattern. (See OpenAI docs for up-to-date snippets and model selection guidance.) [OpenAI Platform+1](https://platform.openai.com/?utm_source=chatgpt.com)

from openai import OpenAI  
client = OpenAI()  \# expects OPENAI\_API\_KEY in env  
  
def analyze\_with\_llm(ticker: str, snapshot: dict, headlines: list, model="gpt-4o-mini"):  
    \# Build headlines block  
    if headlines:  
        head\_str = "\\n".join(\[f'- "{h\["title"\]}" ({h.get("url","")})' for h in headlines\])  
    else:  
        head\_str = "- (no recent headlines found)"  
  
    user\_prompt = USER\_TEMPLATE.format(  
        ticker=ticker.upper(),  
        snapshot\_json=json.dumps(snapshot),  
        headlines\_bulleted=head\_str  
    )  
  
    \# Responses API with JSON object output  
    resp = client.responses.create(  
        model=model,  
        input\=\[  
            {"role": "system", "content": \[{"type":"text", "text": SYSTEM\_PROMPT}\]},  
            {"role": "user", "content": \[{"type":"text", "text": FEW\_SHOT\_USER}\]},  
            {"role": "assistant", "content": \[{"type":"text", "text": FEW\_SHOT\_ASSISTANT}\]},  
            {"role": "user", "content": \[{"type":"text", "text": user\_prompt}\]}  
        \],  
        response\_format={"type": "json\_object"}  
    )  
    \# Extract text from the first output  
    raw = resp.output\[0\].content\[0\].text  \# structure per Responses API  
    return json.loads(raw)

> **_Note:_** _Model names and the_ `_responses.create_` _structure may evolve; check the_ **_API Reference_** _and_ **_model selection guide_** _before deploying._ [_OpenAI Platform+1_](https://platform.openai.com/?utm_source=chatgpt.com)

### Handling Timeouts & Retries

Wrap the call with exponential backoff and set per-request timeouts using your HTTP transport if needed. (See OpenAI client docs for transport configuration.) [OpenAI Platform](https://platform.openai.com/?utm_source=chatgpt.com)

## Step 4 — Parse & Display Results

Show a clean CLI printout and optionally save JSON for dashboards.

def render\_report(ticker: str, result: dict):  
    print("\\n=== AI Stock Analysis Report:", ticker.upper(), "===\\n")  
    print("Summary:")  
    print(result.get("summary", "(no summary)"), "\\n")  
  
    print("Bullish Signals:")  
    for s in result.get("bullish\_signals", \[\]):  
        print("  •", s)  
    print("\\nBearish Signals:")  
    for s in result.get("bearish\_signals", \[\]):  
        print("  •", s)  
  
    pt = result.get("price\_target", {}) or {}  
    base = pt.get("base")  
    rng = pt.get("range")  
    horizon = pt.get("time\_horizon\_days")  
    print("\\nPrice Target:")  
    print(f"  Base: {base}  Range: {rng}  Horizon (days): {horizon}")  
  
    conf = result.get("confidence")  
    print(f"\\nConfidence: {conf}")  
  
    srcs = result.get("sources", \[\])  
    if srcs:  
        print("\\nSources:")  
        for u in srcs:  
            print("  -", u)

### **_Docs & References:_**

-   OpenAI API docs & model selection (keep an eye on JSON/Responses patterns). [OpenAI Platform+1](https://platform.openai.com/?utm_source=chatgpt.com)
-   `yfinance` package (news/data features evolve). [PyPI](https://pypi.org/project/yfinance/?utm_source=chatgpt.com)
-   Alpha Vantage docs (Market News & technical indicators). [Alpha Vantage+1](https://www.alphavantage.co/documentation/?utm_source=chatgpt.com)

## Validation & Safety

-   **Not financial advice:** Make this explicit in the system prompt and your CLI output.
-   **Grounding:** The prompt instructs the model to rely **only** on supplied data/news.
-   **Sanity checks:** If SMA values are `None` (insufficient history), ensure the model explains reduced confidence.
-   **Rate limits:** If you hit Yahoo limits, backoff/retry and consider caching. [GitHub](https://github.com/ranaroussi/yfinance/issues/2422?utm_source=chatgpt.com)

## Performance & Cost Tips

-   **Token budget:** Truncate headlines to the newest 5–8 and keep prices compact (latest snapshot + minimal history summary).
-   **Batching:** Analyze multiple tickers by looping and reusing the same client; write each JSON to disk.
-   **Caching:** Cache `yfinance` responses locally to avoid frequent calls.
-   **Model choice:** Use a smaller, faster model for drafts; upgrade to a larger model for final output, as advised in model selection guidance. [OpenAI Platform](https://platform.openai.com/docs/guides/model-selection/1-focus-on-accuracy-first?utm_source=chatgpt.com)

## Extensions & Next Steps

-   **Streamlit UI:** Turn the JSON into a visual dashboard (gauges for confidence, chips for signals).
-   **Discord/Slack bot:** Post the report daily.
-   **Cron job:** Schedule on Linux/macOS using `crontab`.
-   **Add more data sources:** SEC filings, earnings transcripts (RAG with embeddings), technical indicators (RSI/MACD). Alpha Vantage exposes 50+ indicators you can append to the snapshot. [Alpha Vantage](https://www.alphavantage.co/best_stock_market_api_review/?utm_source=chatgpt.com)

## Conclusion

You just built a practical **AI Stock Analysis Agent** that fetches real-time market data, structures it for an LLM, and outputs a **clean JSON report** with **summary**, **bullish/bearish signals**, and a **price target**. From here, consider a Streamlit front-end, a Slack/Discord bot, or a nightly Cron job. You can also wire in **SEC filings** and earnings transcripts for deeper, fundamental context.

> If you’d like to see how this looks in a polished, production-grade environment, explore m[**eyka.com**](https://meyka.com/?utm_source=chatgpt.com). It’s the live version of this very project — offering AI-driven stock analysis, market insights, and a chatbot interface without needing to run any code yourself.

**External Resource:** Explore OpenAI’s developer docs for up-to-date client examples and model guidance.

## Embedded Content

---