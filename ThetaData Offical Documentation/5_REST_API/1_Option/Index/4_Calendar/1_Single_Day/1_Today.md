Today (v3)
Free
Value
Standard
Pro
GET
https://localhost:25503/v3/calendar/today
REQUIRED

The v3 Theta Terminal must be running to access data.

Retrieves current day equity market schedule
*On days when the market closes early at 1:00 PM ET; eligible options will trade until 1:15 PM.
**Some NYSE exchanges will continue late trading until 5:00 PM ET on early close days.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

Current day equity market schedule
http://localhost:25503/v3/calendar/today
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
Returns current day schedule

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
}

#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + '/calendar/today'


with httpx.stream("GET", url, params=params, timeout=60) as response:
    response.raise_for_status()  # make sure the request worked
    for line in response.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)  # Now you get a parsed list of fields