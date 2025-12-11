Contracts (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/option/list/contracts/{request_type}
REQUIRED

The v3 Theta Terminal must be running to access data.

Lists all contracts that were traded or quoted on a particular date.

If the symbol parameter is specified, the returned contracts will be filtered to match the symbol.
Multiple symbols can be specified by separating them with commas such as symbol=AAPL,SPY,AMD
This endpoint is updated real-time.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

List all contracts for an option trade with a given date
http://localhost:25503/v3/option/list/contracts/trade?date=20220930
List all contracts for an option quote with a given symbol and date
http://localhost:25503/v3/option/list/contracts/quote?symbol=AAPL&date=20220930
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
 -
The stock or index symbol, or underlying symbol for options.

Type:
array
date
Required
 -
The date to fetch data for.

Type:
string
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
List all contracts for an option trade with a given date

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
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/list/contracts/trade'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields