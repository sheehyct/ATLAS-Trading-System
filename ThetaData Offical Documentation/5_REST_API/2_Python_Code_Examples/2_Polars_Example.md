Using Polars
Polars is a high-performance DataFrame library built for speed, scalability, and efficient memory use.
It is designed around a columnar execution model and uses Apache Arrow memory formats internally, enabling fast analytical workloads and automatic parallelization.

Key Advantages
Fast columnar execution with automatic multithreading.
Lazy and eager APIs for optimized query plans or immediate results.
Expression-based syntax that eliminates Python loops and ensures vectorized operations.
Scales to large datasets with far less memory overhead than pandas.
Documentation
Official documentation and user guide:
👉 Polars User Guide

Example

import polars as pl
import httpx
import io

# Base API URL
BASE_URL = "http://localhost:25503/v3"
url = BASE_URL + "/option/history/ohlc"

# Request parameters
params = {
    "date": "2024-11-07",
    "symbol": "AAPL",
    "expiration": "2025-01-17",
    "interval": "5m",
    "format": "ndjson"
}

# Make the request
r = httpx.get(url, params=params, timeout=60)
r.raise_for_status()

# Load ndjson into Polars
df = pl.read_ndjson(io.StringIO(r.text))

print(df)
Polars makes it straightforward to switch between NDJSON and CSV responses.

To load NDJSON:


df = pl.read_ndjson(io.StringIO(r.text))
To switch to CSV, simply use:


df = pl.read_csv(io.StringIO(r.text))
Remember to also update the API request’s format parameter to match the file type you want to parse.