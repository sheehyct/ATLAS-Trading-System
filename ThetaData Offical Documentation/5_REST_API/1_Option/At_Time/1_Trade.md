Trade (v3)
Standard
Pro
GET
https://localhost:25503/v3/option/at_time/trade
REQUIRED

The v3 Theta Terminal must be running to access data.

Returns the last trade reported by OPRA at a specified millisecond of the day.
Trade condition mappings can be found here.
Extended trade conditions are not reported by OPRA for options, so they can be ignored.
The time_of_dayparameter represents the 00:00:00.000 ET that the trade should be provided for.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns the last trade for an option contract
http://localhost:25503/v3/option/at_time/trade?symbol=AAPL&expiration=20241108&strike=220.000&right=call&start_date=20241104&end_date=20241104&time_of_day=09:30:01.000
Query Parameters
symbol
Required
 -
The stock or index symbol, or underlying symbol for options.

Type:
string
start_date
Required
 -
The start date (inclusive).

Type:
string
end_date
Required
 -
The end date (inclusive).

Type:
string
time_of_day
Required
 -
The time of the day to fetch data for; assumed to be America/New_York.

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
Returns the last trade for an option contract

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
from datetime import datetime, timedelta

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
RAW_PARAMS= {
  'symbol': 'AAPL',
  'start_date': '2024-11-07',
  'end_date': '2024-11-07',
  'time_of_day': '09:31:00.000',
  'expiration': '2025-01-17',
}

# define date range
start_date = datetime.strptime('2024-11-07', '%Y-%m-%d')
end_date = datetime.strptime('2024-11-07', '%Y-%m-%d')

dates_to_run = []
while start_date <= end_date:
    if start_date.weekday() < 5:  # skip Sat/Sun
        dates_to_run.append(start_date)
    start_date += timedelta(days=1)

print("Dates to request:", [d.strftime("%Y-%m-%d (%A)") for d in dates_to_run])

#
# This is the streaming version, and will read line-by-line
#
for day in dates_to_run:
    day_str = day.strftime("%Y%m%d")

    # set params
    params = RAW_PARAMS
    if 'start_date' in params:
        params['start_date'] = day_str
    if 'end_date' in params:
        params['end_date'] = day_str
    url = BASE_URL + '/option/at_time/trade'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields