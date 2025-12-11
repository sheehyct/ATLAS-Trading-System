Dates (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/option/list/dates/{request_type}
REQUIRED

The v3 Theta Terminal must be running to access data.

Lists all dates of data that are available for an option with a given symbol, request type, and expiration.
This endpoint is updated overnight.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

List all dates for an option quote for a given symbol and expiration date
http://localhost:25503/v3/option/list/dates/quote?symbol=AAPL&expiration=20220930
List all dates for an option trade for a given symbol with any expiration date
http://localhost:25503/v3/option/list/dates/trade?symbol=AAPL&expiration=20220930
Path Parameters
request_type
Required
 -
The request type.

Type:
string
Enum
trade, quote
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
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
List all dates for an option quote for a given symbol and expiration date

Content-Type

application/json

application/json
Schema
JSON
array of:

date
string
The date formated as YYYY-MM-DD.

Format
date
Sample Code

Python

JavaScript

import httpx  # install via pip install httpx
import csv
import io

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/list/dates/trade'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields