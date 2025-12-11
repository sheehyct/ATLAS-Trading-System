End of Day Greeks (v3)
Standard
Pro
GET
https://localhost:25503/v3/option/history/greeks/eod
REQUIRED

The v3 Theta Terminal must be running to access data.

Returns the data for all contracts that share the same provided symbol and expiration.
Uses Theta Data's EOD reports that get generated at 17:15 ET each day. The closing option price and closing underlying price are used for the greeks calculation.
Set expiration to * if you want to retrieve data for every option that shares the same symbol. (note: Any expiration=* must be requested day by day)
The quote fields (bid / ask info) may not be available prior to 2023-12-01. We are working to expose this over the coming months. Obtaining the quote at the end of the day requires much more processing than the trades, so we initially generated our history for trades.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns EOD report for an option contract
http://localhost:25503/v3/option/history/greeks/eod?symbol=AAPL&expiration=20241108&strike=220.000&right=call&start_date=20241104&end_date=20241104
Returns EOD report for all option contracts
http://localhost:25503/v3/option/history/greeks/eod?symbol=AAPL&expiration=*&start_date=20241104&end_date=20241104
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
annual_dividend
 -
The annualized expected dividend amount to be used in Greeks calculations.

Type:
number
rate_type
 -
The interest rate type to be used in a Greeks calculation.

Type:
string (Default: sofr)
Enum
sofr, treasury_m1, treasury_m3, treasury_m6, treasury_y1, treasury_y2, treasury_y3, treasury_y5, treasury_y7, treasury_y10, treasury_y20, treasury_y30
rate_value
 -
The interest rate, as a percent, to be used in a Greeks calculation.

Type:
number
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


timestamp
string
The timestamp in YYYY-MM-DDTHH:mm:ss.SSS format.

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


delta
number
The delta.


theta
string
The Theta.


vega
number
The vega.


rho
number
The rho.


epsilon
string
The epsilon.


lambda
number
The lambda.


gamma
number
The gamma.


vanna
string
The vanna.


charm
number
The charm.


vomma
number
The vomma.


veta
number
The veta.


vera
number
The vera.


speed
number
The speed.


zomma
number
The zomma.


color
string
The color.


ultima
string
The ultima.


d1
number
The d1.


d2
number
The d2.


dual_delta
string
The dual delta.


dual_gamma
number
The dual gamma.


implied_vol
number
The implied volatiltiy calculated using the trade price.


iv_error
number
IV Error: the value of the option calculated using the implied volatiltiy divided by the actual value reported in the quote. This value will increase as the strike price recedes from the underlying price.


underlying_timestamp
string
The underlying date formated as YYYY-MM-DDTHH:mm:ss.SSS format.

Format
date-time

underlying_price
number
The midpoint of the underlying at the time of the option trade.

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
  'expiration': '2025-01-17',
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
    url = BASE_URL + '/option/history/greeks/eod'

    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                print(row)  # Now you get a parsed list of fields