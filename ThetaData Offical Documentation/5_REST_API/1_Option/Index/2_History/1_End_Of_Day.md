End of Day (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/index/history/eod
REQUIRED

The v3 Theta Terminal must be running to access data.

Since the indices feeds do not provide a national EOD report, Theta Data generates a national EOD report at 17:15 each day.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns EOD report for a given symbol between specified dates (inclusive)
http://localhost:25503/v3/index/history/eod?symbol=SPX&start_date=20241104&end_date=20241108
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
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns EOD report for a given symbol between specified dates (inclusive)

Content-Type

application/json

application/json
Schema
JSON
array of:

created
string
The date formated as YYYY-MM-DDTHH:mm:ss.SSS format.

Format
date-time

last_trade
string
The last trade date formated as YYYY-MM-DDTHH:mm:ss.SSS format.

Format
date-time

open
number
The opening trade price.


high
number
The highest traded price.


low
number
The lowest traded price.


close
number
The closing traded price.


volume
integer
The amount of contracts / shares traded.


count
integer
The amount of trades.


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
import io
from datetime import datetime, timedelta

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
RAW_PARAMS= {
  'symbol': 'SPX',
  'start_date': '2024-11-07',
  'end_date': '2024-11-07',
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
    url = BASE_URL + '/index/history/eod'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields