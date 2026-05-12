# Signal Scanner Button Version for Vercel

This version does **not** require anything to run on your local computer.

The dashboard is a static `index.html` page. When you click **Run Scan**, the browser calls `/api/scan`, a Vercel Python Function. The function scans once, returns JSON to the page, and stops.

## What changed from the original files

- No forever loop on Vercel.
- No `scanner_state.json` polling.
- Added a **Run Scan** button.
- Added a Vercel Python API route at `api/scan.py`.
- Moved the scanner logic into `api/scanner_core.py`.
- Aligned the server-side scanner to the Pine Script v6 logic as closely as possible in a stateless web API:
  - Entry Style: Fast 5m, Balanced, Conservative
  - Session phase logic
  - HTF structure filter
  - Pullback, breakout, liquidity sweep, and second-entry logic
  - Failed breakout/fakeout cooldown
  - Signal cooldown and max signals per hour
  - Score system for Buy %, Sell %, Edge, Confidence
  - Virtual trade manager simulation across the loaded bars
  - Actions like BUY CONFIRMED, SELL CONFIRMED, HOLD LONG, HOLD SHORT, TRAIL LONG, TRAIL SHORT, SELL / EXIT LONG, BUY / EXIT SHORT, GET READY, LEAN, WAIT

## Important limitations

1. TradingView Pine runs on chart data inside TradingView. This Vercel version uses Yahoo Finance chart data, so values can differ slightly.
2. The Python scan uses the last closed bar by default to behave more like Pine's confirmed-bar signals.
3. Vercel Functions have time limits. If the full list times out, use the Optional Symbols box first, for example:
   `AAPL,NVDA,TSLA,BTC-USD,ETH-USD`
4. This is an informational scanner, not financial advice.

## Deploy using only browser clicks

1. Create a GitHub account or sign in.
2. Create a new repository.
3. Upload these files and folders into the repository:
   - `index.html`
   - `requirements.txt`
   - `vercel.json`
   - the whole `api` folder
4. Sign in to Vercel.
5. Click **Add New... > Project**.
6. Import the GitHub repository.
7. Click **Deploy**.
8. Open the Vercel URL and click **Run Scan**.

## File list

```text
index.html
requirements.txt
vercel.json
api/scan.py
api/scanner_core.py
```
