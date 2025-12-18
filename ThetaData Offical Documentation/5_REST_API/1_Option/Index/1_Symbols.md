Symbols (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/index/list/symbols
REQUIRED

The v3 Theta Terminal must be running to access data.

A symbol can be defined as a unique identifier for a stock / underlying asset. Common terms also include: root, ticker, and underlying. This endpoint returns all traded symbols for options. This endpoint is updated overnight.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

List all symbols for indices
http://localhost:25503/v3/index/list/symbols
Query Parameters
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
List all symbols for indices

Content-Type

application/json

application/json
Schema
JSON
array of:

symbol
string
The symbol of the contract, or stock / underlying asset / option / index.

Sample Code

Python

JavaScript

import httpx  # install via pip install httpx
import csv
import io

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/index/list/symbols'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields