There are various subscriptions you can purchase, for the various data types sold by Theta Data. This page describes what each subscription entails.

REQUIRED

To access any data, you must have the Theta Terminal running. You should have a terminal open that looks something similar to the image below.

image.png

When you start the terminal, it will display the level of access you have for each type of data.

Free Data
1 year of free historical EOD (End of Day) data for US stocks and options is provided for free. There is a 20-requests/minute rate limit imposed on free accounts.

Stock Data
Theta Data has full historical coverage for the UTP tape going back to 2012-06-01. For symbols only available on the CTA tape, the history is limited to 2020-01-01. This includes symbols like SPY and GE. Be sure to read our Making Requests before purchasing a subscription.

General Access
Tier	Granularity	First Access Date	Server Threads	Delay
FREE	EOD	2023-06-01	30 reqs/min	1 day
VALUE	1 Minute	2021-01-01	1	15-minute
STANDARD	1 Minute	2016-01-01	2	Real-time
PRO	Tick Level	2012-06-01	4	Real time
Historical Endpoint Access
Endpoint	FREE	VALUE	STANDARD	PRO
EOD Report	✔	✔	✔	✔
Quote		✔	✔	✔
OHLC		✔	✔	✔
[Splits]			✔	✔
Trades			✔	✔
Trade Quote			✔	✔
Real Time Endpoint Access
Endpoint	FREE	VALUE	STANDARD	PRO
Quote Snapshot		Delayed (15min)	Real Time	Real Time
OHLC Snapshot		Delayed (15min)	Real Time	Real Time
Trade Snapshot			Real Time	Real Time
Bulk Quote Snapshot			Real Time	Real Time
Options Data
General Access
Tier	Granularity	First Access Date	Server Threads	Delay
FREE	EOD	2023-06-01	30 reqs/min	1 day
VALUE	1 Minute	2020-01-01	1	Real time
STANDARD	Tick Level	2016-01-01	2	Real time
PRO	Tick Level	2012-06-01	4	Real time
Historical Endpoint Access
Endpoint	FREE	VALUE	STANDARD	PRO
EOD	✔	✔	✔	✔
Quote		✔	✔	✔
Open Interest		✔	✔	✔
OHLC		✔	✔	✔
Trade			✔	✔
Trade Quote			✔	✔
Implied Volatility			✔	✔
Greeks 1st Order			✔	✔
Greeks 2nd Order				✔
Greeks 3rd Order				✔
Trade Greeks 1st Order				✔
Trade Greeks 2nd Order				✔
Trade Greeks 3rd Order				✔
Real-Time Endpoint Access
Endpoint	FREE	VALUE	STANDARD	PRO
Quote		✔	✔	✔
Open Interest		✔	✔	✔
OHLC		✔	✔	✔
Trade			✔	✔
Index Data
The resolution of the data is entirely dependent on the reporting exchange. For instance CBOE reports SPX every second. Indices from the Nasdaq Indices Feed are currently not supported. This includes $NDX.

If the previous reported price has not changed, there will be no new tick reported by Theta Data. For instance, if the price of SPX is $4000 at 9:31:00 and the price has not changed at 9:31:01, a new price message will not be available historically and in real-time. This is easy to work around as any "missing" historical price tick can be interpreted as the price did not change from the previous tick.

Tier	Granularity	First Access Date	Delay	Server Threads
FREE	EOD	2024-01-01	NO ACCESS	NO ACCESS
VALUE	15-minute	2023-01-01	15-minute	1
STANDARD	Lowest reported by venues	2022-01-01	real-time	2
PRO	Lowest reported by venues	2017-01-01	real-time	4
Symbol Coverage
Real-time / ongoing updates is available for all indices reported on the CGIF. This includes SPX and VIX. Historic coverage for indices such as RUT and DJX is available between the first access date and 2024-07-01. There is no support for NDX or any symbols on the Nasdaq Indices feed. Our near term plans are to generate synthetic indices data that will match the officially reported prices with a 99% accuracy. This synthetic pricing data will be available to indices data subscribers once available.

Endpoint Access
Endpoint	FREE	VALUE	STANDARD	PRO
EOD Report	X	X	X	X
Price		X	X	X
Price Snapshot			X	X
OHLC Snapshot			X	X