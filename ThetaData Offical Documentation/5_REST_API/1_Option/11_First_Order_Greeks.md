First Order Greeks (v3)
Standard
Pro
GET
https://localhost:25503/v3/option/snapshot/greeks/first_order
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieve a real-time last greeks calculation for all option contracts that lie on a provided expiration.
You might need to change the default expiration date to a different date if it is past the current date. Some quotes are omitted in the example to reduce the space of the sample output.
Make expiration * if you want to get the snapshot for every expiration chain for the underlying.
This endpoint will return no data if the market was closed for the day. Theta Data resets the snapshot cache at midnight ET every night.

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns first order greeks for an option contract
http://localhost:25503/v3/option/snapshot/greeks/first_order?symbol=AAPL&expiration=20260116&strike=275.000&right=call
Returns first order greeks for all option contracts
http://localhost:25503/v3/option/snapshot/greeks/first_order?symbol=AAPL&expiration=*
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
stock_price
 -
The underlying stock price to be used in the Greeks calculation.

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


implied_vol
number
The implied volatiltiy calculated using the trade price.


iv_error
string
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
import sys
import io
from datetime import datetime

BASE_URL = "http://localhost:25503/v3"  # all endpoints use this URL base

# set params
params = {
  'symbol': 'AAPL',
  'expiration': '2025-01-17',
}

# Weekend Check (Sat/Sun)
now = datetime.now()

if now.weekday() >= 5: # 5=Sat, 6=Sun
    print("Market is Closed snapshots may not work")
    sys.exit(0)

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/option/snapshot/greeks/first_order'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields