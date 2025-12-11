A SIP (Securities Information Processor) is a regulated entity that consolidates all trades and quotes from all exchanges. This allows brokers to execute orders at the National Best Bid and Offer (NBBO).

Options: OPRA
OPRA (Options Price Reporting Authority) provides a nationally consolidated quote and trade feed for US equity and index options. OPRA is administered by CBOE. OPRA data vendors are required to pay redistribution fees. Theta Data receives every NBBO quote and trade from the OPRA feed all in real-time with latencies averaging under 3ms. Most OPRA data vendors filter NBBO quotes because they cannot handle the incredible amount of data coming in. There are millions of quotes sent by OPRA in the first second of the market opening.

OPRA GTH
As part of the OPRA Global Trading Hours (GTH), options for SPX, VIX and XSP are traded outside regular trading hours (RTH). All times are eastern time. The last OPRA GTH session of the week starts at 20:15 ET every Sunday.

OPRA Global Trading Hours (GTH) will extend its daily ending hours of operation effective trade date Monday, August 26, 2024 (starting hours of operation for Sunday night session, August 25, 2024), by 10 minutes, to 9:25 A.M from 9:15 A.M., ET.

Session	Start Time	End Time
Begin GTH Order Acceptance (SPX, VIX, XSP)	20:00	n/a
Global Trading Hours (SPX, VIX, XSP)	20:15	09:25 (next day)
Begin RTH Order Acceptance	07:30	n/a
Regular Trading Hours (RTH)	09:30	16:15
Curb	16:15	17:00
Source: CBOE Hours & Holidays

OPRA Extended Trading Hours:
As part of extended trading hours the following symbols are traded until 16:15 ET.


AUM, AUX, BACD, BPX, BRB, BSZ, BVZ, CDD, CITD, DBA, DBB, DBC, DBO, DBS, DIA, DJX, EEM, EFA, EUI, EUU, GAZ, GBP, GSSD, IWM, IWN, IWO, IWV, JJC, JPMD, KBE, KRE, MDY, MLPN, MNX, MOO, MRUT, MSTD, NDO, NDX, NZD, OEF, OEX, OIL, PZO, QQQ, RUT, RVX, SFC, SKA, SLX, SPX, SPX (PM Expiration), SPY, SVXY, UNG, UUP, UVIX, UVXY, VIIX, VIX, VIXM, VIXY, VXEEM, VXST, VXX, VXZ, XEO, XHB, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XME, XRT, XSP, XSP (AM Expiration), & YUK
Equities: CTA & UTP
US Equities data has 2 SIPs and 3 different SIP networks.

CTA Network A (Administered by NYSE)
CTA Network B (Administered by NYSE)
UTP Network C (Administered by Nasdaq)
Theta Data receives a 15-minute delayed feed from all of these networks. Theta Data receives a real-time feed from Nasdaq Basic.

CTA Tape A has the following trading hours:

Session	Start Time	End Time
Pre-Opening	06:30	9:30
Core Open Auction	09:30	15:50
123(c) Closing Imbalance Period	15:50	16:00
Nasdaq Basic
Nasdaq Basic provides a BBO that is within 1% of the NBBO 99.22% of the time. They also publish a time and sales data feed in real-time. The time and sales information is for orders executed within the Nasdaq execution system as well as trades reported to the FINRA/Nasdaq TRF.

CBOE Global Indices Feed
Theta Data is a real-time recipient of the CGIF. Indices such as SPX and VIX are included in this feed.