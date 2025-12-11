Introduction
Theta Data is a low latency market data platform, providing institutional grade financial data to all. Theta Data has over 12 years of stocks, options, and indices data. While a subscription is required to access most data, some data is provided for free.

SIGNUP

See thetadata.net to sign up today!

The Theta Data Way
We believe that accessing extensive financial datasets should be straightforward and efficient. Few providers can match our throughput at such a competitive price. Our technology-driven approach keeps us at the cutting edge, consistently introducing new features that give our clients a competitive advantage.

The Theta Terminal
The Theta Terminal is a low-latency process that allows for the lossless compression of financial information. It was first released in November 2022 and has undergone major improvements over the years. The goal of Theta Terminal is to deliver data as fast as possible to the user. No expense is spared in terms of smart connection management and data compression. This program is required to access all our data and is the backbone of our data delivery technology. The Theta Terminal hosts a local server on your machine that only a single IP / connection can access. This IP limitation can be removed for commercial clients.

Terminology
The following terms are used throughout the documentation. There definitions are provided below for clarity.

Theta Terminal
A low latency process used to deliver market data over the internet, which utilizes FIC and FIT technology. It connects to the MDDS and FPSS.

Financial Information Compression (FIC)
FIC is a proprietary compression developed by Theta Data used to significantly reduce the size of the data sent over the internet.

Financial Information Tick (FIT)
A FIT or Financial Information Tick is the lowest level of data. All data we provide is in the form of ticks (rows), which are natively arrays of signed int32s.

Market Data Distribution Server (MDDS)
MDDS is the server that the Theta Terminal connects to to make any query-based requests such as snapshots and historical requests.

Feed Processing Stream Server (FPSS)
FPSS is a server that processes exchange feeds. It is reponsible for all types of data streaming otherwise known as continuous updates.

The Interp3 Matching Engine
Superseding the Interp2 engine, this allows us to perfectly match or align any two sets of tick level data and dynamically compute limitless outputs and analysis on the data. Additionally, Interp3 allows us to keep track of real-time dividend yields and interest rates for each tick of data.