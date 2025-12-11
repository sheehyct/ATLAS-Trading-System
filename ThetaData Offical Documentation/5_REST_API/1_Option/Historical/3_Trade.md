Trade (v3)
Standard
Pro
GET
https://localhost:25503/v3/option/history/trade
REQUIRED

The v3 Theta Terminal must be running to access data.

Returns every trade reported by OPRA.
Trade condition mappings can be found here.
Extended trade conditions are not reported by OPRA for options, so they can be ignored.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns every trade for an option contract
http://localhost:25503/v3/option/history/trade?symbol=AAPL&expiration=20241108&strike=220.000&right=call&date=20241104
Returns every trade for all option contracts
http://localhost:25503/v3/option/history/trade?symbol=AAPL&expiration=*&date=20241104
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
Returns every trade for an option contract

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


timestamp
string
The timestamp in YYYY-MM-DDTHH:mm:ss.SSS format.

Format
date-time

sequence
integer
The exchange sequence.


ext_condition1
integer
Additional trade condition(s). These can be ignored for options.


ext_condition2
integer
Additional trade condition(s). These can be ignored for options.


ext_condition3
integer
Additional trade condition(s). These can be ignored for options.


ext_condition4
integer
Additional trade condition(s). These can be ignored for options.


condition
integer
The trade condition.


size
integer
The amount of contracts / shares traded.


exchange
integer
The exchange the trade was executed.


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
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/history/trade'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields