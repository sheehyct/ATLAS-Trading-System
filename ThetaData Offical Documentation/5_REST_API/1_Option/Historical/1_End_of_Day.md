End of Day (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/option/history/eod
REQUIRED

The v3 Theta Terminal must be running to access data.

Since OPRA does not provide a national EOD report for options, Theta Data generates a national EOD report at 17:15 ET each day.
created represents the datetime the report was generated and last_trade represents the datetime of the last trade.
The quote in the response represents the last NBBO reported by OPRA at the time of report generation.
You can read more about EOD & OHLC data here.
The quote fields (bid / ask info) may not be available prior to 2023-12-01. We will expose further history for the EOD quote in the near future.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns EOD report for an option contract
http://localhost:25503/v3/option/history/eod?symbol=AAPL&expiration=20241115&strike=170.000&right=call&start_date=20241104&end_date=20241104
Returns EOD report for all option contracts
http://localhost:25503/v3/option/history/eod?symbol=AAPL&expiration=*&start_date=20241104&end_date=20241104
Query Parameters
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
format
 -
The format of the data when returned to the user.

Type:
string (Default: csv)
Enum
csv, json, ndjson, html
Responses
200
Returns EOD report for an option contract

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
  'start_date': '2024-11-07',
  'end_date': '2024-11-07',
  'symbol': 'AAPL',
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
    url = BASE_URL + '/option/history/eod'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields