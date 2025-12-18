Price (v3)
Standard
Pro
GET
https://localhost:25503/v3/index/snapshot/price
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieves a real-time last index price.
Exchanges typically generate a price report every second for popular indices like SPX.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns last index price
http://localhost:25503/v3/index/snapshot/price?symbol=SPX
Query Parameters
symbol
Required
 -
The stock or index symbol, or underlying symbol for options. Specify '*' for all symbols or a comma separated list when appropriate.

Type:
array
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns last index price

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


price
number
The trade price.

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
  'symbol': 'SPX',
}

# Weekend Check (Sat/Sun)
now = datetime.now()

if now.weekday() >= 5: # 5=Sat, 6=Sun
    print("Market is Closed snapshots may not work")
    sys.exit(0)

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/index/snapshot/price'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields