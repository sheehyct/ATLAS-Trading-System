Quote (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/option/snapshot/quote
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieve a real-time last NBBO quote of an option contract.
You might need to change the default expiration date to a different date if it is past the current date.
This endpoint will return no data if the market was closed for the day. Theta Data resets the snapshot cache at midnight ET every night.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns last NBBO quote for an option contract
http://localhost:25503/v3/option/snapshot/quote?symbol=AAPL&expiration=20260116&right=call&strike=275.000
Returns last NBBO quote for all option contracts
http://localhost:25503/v3/option/snapshot/quote?symbol=AAPL&expiration=*
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
Returns last NBBO quote for an option contract

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


bid_size
integer
The last NBBO bid size.


bid_exchange
integer
The last NBBO bid exchange.


bid
number
The last NBBO bid price.


bid_condition
integer
The last NBBO bid condition.


ask_size
integer
The last NBBO ask size.


ask_exchange
integer
The last NBBO ask exchange.


ask
number
The last NBBO ask price.


ask_condition
integer
The last NBBO ask condition.

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
url = BASE_URL + '/option/snapshot/quote'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields