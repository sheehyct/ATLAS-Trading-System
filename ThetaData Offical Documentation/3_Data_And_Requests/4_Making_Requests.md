Data Availability
TIP

Check our website as we are continually adding more data for customers!

Historical Greeks (options) & Equities Availability
Equities data is split up into 3 feeds (see The SIPs for more information):

CTA-A (Administered by NYSE)
CTA-B (Administered by NYSE)
UTP-C (Administered by Nasdaq)
Theta Data receives market data for all of these feeds listed above. The 10 years of history in the system is only for the UTP-C tape, which covers most, but not all tickers. Data prior to 2020-01-01 for some tickers does not exist. For instance, $SPY is not included in the UTP tape, so there is no historical stock data or greeks data prior to 2020-01-01 for it. We are continually adding more data, so check back soon or contact Support.

Historical Greeks Availability: Index Options
Nasdaq indices are currently not included, but will be added at a later date. Similar to equities, our Greeks data on index options (v2 requests must be used) goes back to 2017-01-01. At the moment there are no ongoing updates or history for the Nasdaq Indices Feed, which includes $NDX. This means that $NDX Greeks aren't available, however you can supply the under_price parameter to greeks snapshots, which will force Theta Data to use your supplied price.

Historical Trades, EOD, OHLC: Options
Trades, EOD, and OHLC options data is available as far back at 2012-06-01.

Listing bulk dates
At this time bulk date listing is only supported for quotes / trades. Most requests use the same exact dates as quotes or trades, so there shouldn't be a reason to list dates for other request types.

Historical ETH OPRA data
Extended Trading hours trades (impacts EOD data as well) and quotes are available from 2015 until 2018. However, from 2019 until December 2022, there is no ETH data for quotes and trades. After January 2022, Theta Data has full GTH / ETH coverage. Unfortunately this is a limitation of the data vendor we purchased from. ETH quotes & trades are only for SPX, VIX, DJI, and RUT options.

NOTE

This has no impact on equity options.

Error-Handling
When there is an error making a request, a text response is returned that describes the error. The http response code of the response will correspond to the errors defined below. If the request was successful, the http code 200 is returned. It is imperative that your application properly handles error codes like DISCONNECTED and NO_DATA.

Http Code	Error Name	Description
200	OKAY, NO ERROR	No error.
404	NO_IMPL	There is no implementation of this request. Either the request you are making is invalid or you are using an outdated Theta Terminal version.
429	OS_LIMIT	The operating system is throttlting your requests. This happens when making a large amount of small low latency requests. An easy solution to this error is to retry the request until you no longer get this error code.
470	GENERAL	A general error.
471	PERMISSION	Your account does not have the permissions required to execute the request.
472	NO_DATA	There was no data found for the specified request.
473	INVALID_PARAMS	The parameters / syntax of the request is invalid. Sometimes updating your Theta Terminal to the latest version could resolve this.
474	DISCONNECTED	Connection has been lost to Theta Data MDDS.
475	TERMINAL_PARSE	There was an issue parsing the request after it was received.
476	WRONG_IP	The IP address does not match the IP address that the first request was made on. Make sure you use the same ip to make requests while the terminal is running. You cannot switch between 127.0.0.1 and localhost.
477	NO_PAGE_FOUND	The page does not exist or expired.
570	LARGE_REQUEST	The request asking for too much data. Follow these guidelines.
571	SERVER_STARTING	The server is forcibly and intentionally restarting.
572	UNCAUGHT_ERROR	Reach out to support with the exact request you made.
Download as CSV

Trade Sequences
The trade sequence overflows once it reaches 2,147,483,647 (maximum value of a signed 32-bit integer). When the exchange sequence reaches -1, that means the sequence is 4,294,967,294. Once the sequence reaches 0 for a second time (it starts at 0), that means the exchange sequence has overflowed.