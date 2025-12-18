Open High Low Close (v3)
Standard
Pro
GET
https://localhost:25503/v3/index/history/ohlc
REQUIRED

The v3 Theta Terminal must be running to access data.

Aggregated OHLC bars that use SIP rules for each bar.
Time timestamp of the bar represents the opening time of the bar. For a trade to be part of the bar: bar timestamp <= trade time < bar timestamp + interval.
Exchanges typically generate a price report every second for popular indices like SPX.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns OHLC for a given symbol between specified dates (inclusive) with a one minute interval
http://localhost:25503/v3/index/history/ohlc?symbol=SPX&start_date=20241104&end_date=20241104&interval=1m
Query Parameters
symbol
Required
 -
The stock or index symbol, or underlying symbol for options.

Type:
string
start_date
Required
 -
The start date (inclusive).

Type:
string
end_date
Required
 -
The end date (inclusive).

Type:
string
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
Returns OHLC for a given symbol between specified dates (inclusive) with a one minute interval

Content-Type

application/json

application/json
Schema
JSON
array of:

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
from datetime import datetime, timedelta

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
RAW_PARAMS= {
  'symbol': 'SPX',
  'start_date': '2024-11-07',
  'end_date': '2024-11-07',
  'interval': '1m',
}

# define date range
start_date = datetime.strptime('2024-11-07', '%Y-%m-%d')
end_date = datetime.strptime('2024-11-07', '%Y-%m-%d')

dates_to_run = []
while start_date <= end_date:
    if start_date.weekday() < 5:  # skip Sat/Sun
        dates_to_run.append(start_date)
    start_date += timedelta(days=1)

print("Dates to request:", [d.strftime("%Y-%m-%d (%A)") for d in dates_to_run])

#
# This is the streaming version, and will read line-by-line
#
for day in dates_to_run:
    day_str = day.strftime("%Y%m%d")

    # set params
    params = RAW_PARAMS
    if 'start_date' in params:
        params['start_date'] = day_str
    if 'end_date' in params:
        params['end_date'] = day_str
    url = BASE_URL + '/index/history/ohlc'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields