One of the following status codes are returned for each HTTP request made.

Error Codes
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
571	SERVER_STARTING	The server is forcibly and intentionally restarting.
572	UNCAUGHT_ERROR	Reach out to support with the exact request you made.