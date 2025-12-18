On Date (v3)
Value
Standard
Pro
GET
https://localhost:25503/v3/calendar/on_date
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieves equity market schedule for a given date
Note: Holiday data is available 01/01/2012 through the end of the calendar year that immediately follows the current year
*On days when the market closes early at 1:00 PM ET; eligible options will trade until 1:15 PM.
**Some NYSE exchanges will continue late trading until 5:00 PM ET on early close days.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Returns equity market schedule for the requested date
http://localhost:25503/v3/calendar/on_date?date=20251225
Query Parameters
date
Required
 -
The date to fetch data for.

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
Returns requested day schedule

Content-Type

application/json

application/json
Schema
JSON
array of:

type
string
The schedule type (open, full_close, early_close, weekend).


open
string
Market open time in HH:mm:ss format.


close
string
Market close time in HH:mm:ss format.

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
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/calendar/on_date'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields