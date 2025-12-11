Open High Low Close (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/option/snapshot/ohlc
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieve a real-time last ohlc of an option contract for the trading day.
You might need to change the default expiration date to a different date if it is past the current date.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns OHLC for a given option contract
http://localhost:25503/v3/option/snapshot/ohlc?symbol=AAPL&expiration=20260116&right=call&strike=275.000
Returns OHLC for all option contracts
http://localhost:25503/v3/option/snapshot/ohlc?symbol=AAPL&expiration=*
Query Parameters
symbol
Required
 -
The stock or index symbol, or underlying symbol for options.

Type:
string
expiration
Required
 -
The expiration of the contract in YYYY-MM-DD or YYYYMMDD format, or * for all expirations.

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
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns OHLC for a given option contract

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

Sample Code

Python

JavaScript

import httpx  # install via pip install httpx
import csv
import sys
import io
from datetime import datetime

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
}

# Weekend Check (Sat/Sun)
now = datetime.now()

if now.weekday() >= 5: # 5=Sat, 6=Sun
    print("Market is Closed snapshots may not work")
    sys.exit(0)

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/snapshot/ohlc'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields