Basic Pipeline
The example below features an example of combining Polars and asyncio to acheive exceptionally fast processing times for seconds-level data.


import asyncio
import httpx
import polars as pl 
import io

# ----------------------------------------------------
# Constant Variables
# ----------------------------------------------------

BASE_URL = 'http://localhost:25503/v3'
CONCURRENCY_LIMIT = 4
SEMAPHORE = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ----------------------------------------------------
# STEP 1: Get avaliable trading days
# ----------------------------------------------------
# Make the request (Using ndjson as an example)
r = httpx.get(BASE_URL + '/stock/list/dates/quote?symbol=AAPL&format=ndjson', timeout=60)
r.raise_for_status()

# Load response data into Polars
avaliable_dates = pl.read_ndjson(io.StringIO(r.text))
request_date_range = avaliable_dates['date'][-30:]

# ----------------------------------------------------
# STEP 2: Create async request function
# ----------------------------------------------------
async def fetch(client: httpx.AsyncClient, date: str):
    async with SEMAPHORE:
        params = {'symbol': 'AAPL', 'date': date, 'interval': '1s', 'format': 'ndjson'}
        r = await client.get(BASE_URL + '/stock/history/quote', params=params)
        r.raise_for_status()
        df = pl.read_ndjson(io.StringIO(r.text))
        return df 
    
# ----------------------------------------------------
# Main Runner
# ----------------------------------------------------
async def run_requests():
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [fetch(client, date) for date in request_date_range]
        results = await asyncio.gather(*tasks)

    # Combine all dataframes into one
    results = pl.concat(results)
    return results

# ----------------------------------------------------
# Running the script
# ----------------------------------------------------
if __name__ == "__main__":
    results = asyncio.run(run_requests())
    print(results)
In this workflow:

Step 1 retrieves a list of available trading days using an NDJSON response and loads it directly into Polars.

Step 2 defines an asynchronous fetch function that:

issues a request for a specific trading day
loads the returned NDJSON payload into a Polars DataFrame
Step 3 uses asyncio.gather to run many requests concurrently (bounded by a semaphore), and finally concatenates all returned DataFrames into a single result.

This pattern provides a clean, scalable structure for building high-performance data ingestion pipelines.