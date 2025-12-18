Dates (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/index/list/dates
REQUIRED

The v3 Theta Terminal must be running to access data.

Lists all dates of data that are available for a index with a given request type and symbol. This endpoint is updated overnight.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

List all dates for a index for a given symbol
http://localhost:25503/v3/index/list/dates?symbol=SPX
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
List all dates for a index for a given symbol

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
  'symbol': 'SPX',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/index/list/dates'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields