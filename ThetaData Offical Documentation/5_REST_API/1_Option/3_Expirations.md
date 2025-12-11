Expirations (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/option/list/expirations
REQUIRED

The v3 Theta Terminal must be running to access data.

Lists all dates of expirations that are available for an option with a given symbol.
This endpoint is updated overnight.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

List all expirations for an option with a given symbol
http://localhost:25503/v3/option/list/expirations?symbol=AAPL
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
List all expirations for an option with a given symbol

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
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/list/expirations'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields