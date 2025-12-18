Price (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/index/at_time/price
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieves historical indices price reports. Exchanges typically generate a price report every second for popular indices like SPX.
The time_of_day parameter represents the 00:00:00.000 ET that the price should be provided for.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns specific at time historical index price reports
http://localhost:25503/v3/index/at_time/price?symbol=SPX&start_date=20241104&end_date=20241108&time_of_day=09:30:01.000
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
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns specific at time historical index price reports

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
  'symbol': 'SPX',
  'start_date': '2024-11-07',
  'end_date': '2024-11-07',
  'time_of_day': '09:31:00.000',
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
    url = BASE_URL + '/index/at_time/price'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields