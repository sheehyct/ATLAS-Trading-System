Concurrent Requests
Many applications require fetching data for multiple symbols at once.
To speed up these workflows, the Theta Data API fully supports asynchronous clients such as httpx.AsyncClient.

The example below demonstrates how to:

Send multiple option snapshot requests concurrently
Respect a maximum concurrency limit (based on your subscription tier)
Use asyncio and a semaphore to control parallelism
Collect all responses efficiently into a single result list
This pattern is recommended for high-throughput data retrieval such as scanning large symbol lists or date ranges.

Example:

import asyncio
import httpx

BASE_URL = 'http://localhost:25503/v3'
SYMBOLS = ['AAPL', 'TSLA', 'META', 'SPY', 'NVDA', 'QQQ', 'AMC', 'SBUX']


# Semaphore is used to control concurrency limit
CONCURRENCY_LIMIT = 4
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ----------------------------------------------------
# Async Request Function
# ----------------------------------------------------
async def fetch(client: httpx.AsyncClient, symbol: str):
    async with semaphore:
        params = {'symbol': symbol, 'expiration': '*'}
        r = await client.get(BASE_URL + '/option/snapshot/ohlc', params=params)
        r.raise_for_status()
        return {"symbol": symbol, "data": r}


# ----------------------------------------------------
# Main Runner
# ----------------------------------------------------
async def run_requests():
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [fetch(client, sym) for sym in SYMBOLS]
        results = await asyncio.gather(*tasks)

    return results