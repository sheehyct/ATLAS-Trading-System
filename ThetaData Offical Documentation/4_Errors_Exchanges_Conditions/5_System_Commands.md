Shutdown
This kills the Theta Terminal process upon calling. Returns "OK".

Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

WARNING

This sample really will shut down your terminal - try with caution.

http://localhost:25503/v3/terminal/shutdown

MDDS Status
This returns a message indicating the status of the MDDS connection.

Response	Definition
CONNECTED	The terminal is connected.
UNVERIFIED	The terminal was able to connect, but the credentials could not be authenticated.
DISCONNECTED	The terminal is not connected to MDDS.
ERROR	Some other error occured processing this request. Check the terminal log.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

http://localhost:25503/v3/terminal/mdds/status

FPSS Status
This returns a message indicating the status of the FPSS connection.

Response	Definition
CONNECTED	The terminal is connected.
UNVERIFIED	The terminal was able to connect, but the credentials could not be authenticated.
DISCONNECTED	The terminal is not connected to FPSS.
ERROR	Some other error occured processing this request. Check the terminal log.
Sample URL
Paste the URL below into your browser while the Theta Terminal is running.

http://localhost:25503/v3/terminal/fpss/status