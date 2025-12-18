Price (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/index/history/price
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieves historical indices price reports. Exchanges typically generate a price report every second for popular indices like SPX.
When the interval parameter is specified, the returned data represents the price at the exact time of each timestamp. If the timestamp in the response is 10:30:00, the price field represents the price at that exact time of the day.
A price update from the exchange is omitted if the price remained the same from the previous update.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns historical index price reports
http://localhost:25503/v3/index/history/price?symbol=SPX&date=20241104&interval=1m
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
interval
Required
 -
The size of the time interval must be one of the available options listed below.

Type:
string (Default: 1s)
Enum
tick, 10ms, 100ms, 500ms, 1s, 5s, 10s, 15s, 30s, 1m, 5m, 10m, 15m, 30m, 1h
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns historical index price reports

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

price
number
The trade price.

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
  'symbol': 'SPX',
  'interval': '1m',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/index/history/price'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields