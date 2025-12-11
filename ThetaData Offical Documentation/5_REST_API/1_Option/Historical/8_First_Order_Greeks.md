First Order Greeks (v3)
Standard
Pro
GET
https://localhost:25503/v3/option/history/greeks/first_order
REQUIRED

The v3 Theta Terminal must be running to access data.

Returns the data for all contracts that share the same provided symbol and expiration.
Calculated using the option and underlying midpoint price. If an interval size is specified (highly recommended), the option quote used in the calculation follows the same rules as the quote endpoint.
The underlying price represents whatever the last underlying price was at the timestamp field. You can read more about how Theta Data calculates greeks here.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns first order greeks for an option contract
http://localhost:25503/v3/option/history/greeks/first_order?symbol=AAPL&expiration=20241108&date=20241104&interval=5m
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
The expiration of the contract in YYYY-MM-DD or YYYYMMDD format.

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
interval
Required
 -
The size of the time interval must be one of the available options listed below.

Type:
string (Default: 1s)
Enum
tick, 10ms, 100ms, 500ms, 1s, 5s, 10s, 15s, 30s, 1m, 5m, 10m, 15m, 30m, 1h
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
Returns first order greeks for an option contract

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

bid
number
The last NBBO bid price.


ask
number
The last NBBO ask price.


delta
number
The delta.


theta
number
The Theta.


vega
number
The vega.


rho
number
The rho.


epsilon
number
The epsilon.


lambda
number
The lambda.


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

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
  'date': '2024-11-07',
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
  'interval': '1m',
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/history/greeks/first_order'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields