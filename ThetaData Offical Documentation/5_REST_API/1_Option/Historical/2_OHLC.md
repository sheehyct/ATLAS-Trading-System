Open High Low Close (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/option/history/ohlc
REQUIRED

The v3 Theta Terminal must be running to access data.

Aggregated OHLC bars that use SIP rules for each bar.
Time timestamp of the bar represents the opening time of the bar. For a trade to be part of the bar: bar timestamp <= trade time < bar timestamp + interval.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns OHLC for an option contract
http://localhost:25503/v3/option/history/ohlc?symbol=AAPL&expiration=20231103&strike=170.000&right=call&date=20231103&interval=1m
Query Parameters
date
Required
 -
The date to fetch data for.

Type:
string
symbol
Required
 -
The stock or index symbol, or underlying symbol for options.

Type:
string
expiration
Required
 -
The expiration of the contract in YYYY-MM-DD or YYYYMMDD format.

Type:
string
strike
 -
The strike price of the contract in dollars (ie 100.00 for $100.00), or * for all strikes.

Type:
string (Default: *)
right
 -
The right (call or put) of the contract.

Type:
string (Default: both)
Enum
call, put, both
interval
Required
 -
The size of the time interval must be one of the available options listed below.

Type:
string (Default: 1s)
Enum
tick, 10ms, 100ms, 500ms, 1s, 5s, 10s, 15s, 30s, 1m, 5m, 10m, 15m, 30m, 1h
start_time
 -
The start time (inclusive) in the specified day.

Type:
string (Default: 09:30:00)
end_time
 -
The end time (inclusive) in the specified day.

Type:
string (Default: 16:00:00)
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns OHLC for an option contract

Content-Type

application/json

application/json
Schema
JSON
array of:

symbol
string
The symbol of the contract, or stock / underlying asset / option / index.


expiration
string
Expiration date of the contract in YYYY-MM-DD format.

Format
date

strike
number
Strike price of the contract in dollars 180.00


right
string
Indicates whether the contract is a call or put option.


timestamp
string
The timestamp in YYYY-MM-DDTHH:mm:ss.SSS format.

Format
date-time

open
number
The opening trade price.


high
number
The highest traded price.


low
number
The lowest traded price.


close
number
The closing traded price.


volume
integer
The amount of contracts / shares traded.


count
integer
The amount of trades.


vwap
number
The volume weighted average price of the given interval.

Sample Code

Python

JavaScript

import httpx  # install via pip install httpx
import csv
import io

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
  'date': '2024-11-07',
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
  'interval': '1m',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/history/ohlc'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields