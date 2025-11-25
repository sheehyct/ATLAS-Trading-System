# Tiingo API Documentation

## 1. General

### 1.1 Overview

Tiingo's APIs are built to be performant, consistent, and support extensive filters to speed up development time.

#### 1.1.1 Introduction

The endpoints are broken up into two different types:

1. **REST Endpoints** - Provide a RESTful interface to querying data, especially historical data for end-of-day data feeds.

2. **Websocket** - Provide a websocket interface used to stream real-time data. If you are looking for access to a raw firehose of data, the websocket endpoint is how you get it.

#### 1.1.2 Authentication

In order to use the API, you must sign-up to create an account. All accounts are free and if you need higher usage limits, or you have a commercial use case, you can upgrade to the Power and/or Commercial plan.

Once you create an account, your account will be assigned an authentication token. This token is used in place of your username & password throughout the API, so keep it safe like you would your password.

You can find your API token at: https://www.tiingo.com/account/api/token

#### 1.1.3 Usage Limits

To keep the API affordable to all, each account is given generous rate-limits. We limit based on:

- **Hourly Requests** - Reset every hour
- **Daily Requests** - Reset every day at midnight EST
- **Monthly Bandwidth** - Reset the first of every month at midnight EST

We do not rate limit to minute or second, so you are free to make your requests as you desire.

The basic, power, and commercial power plans offer different levels of rate limits. To see what these rate limits are, visit the [pricing page](https://www.tiingo.com/pricing).

#### 1.1.4 Response Formats

For most REST endpoints, you can choose which format the data is returned in:

- **JSON** - Data returned in JSON format. Allows the most flexibility as we can append meta data and debugging data. Requires more bandwidth, which may be slower.

- **CSV** - A "bare-bones" data return type that is often 4-5x faster than JSON. Data is returned in comma-separated-format, helpful for importing into spreadsheet programs like Excel.

To return data in a particular format, pass the **format** parameter with values "json" or "csv".

#### 1.1.5 Symbology

Tiingo's symbol format uses dashes ("-") instead of periods (".") to denote share classes. Our API covers both common share classes and preferred share classes.

Examples:
- Berkshire class A shares: `BRK-A`
- Simon Property Group's Preferred J series shares: `SPG-P-J`

A full list of tickers can be found in [supported_tickers.zip](https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip), which is updated daily.

#### 1.1.6 Permitted Use of Our Data

- **Basic and Power accounts**: Data is for internal and personal use only. You may not redistribute the data in any form.
- **Commercial accounts**: Data is licensed for internal commercial usage. You may not redistribute the data in any form.

For redistribution licensing, email sales@tiingo.com.

---

### 1.2 Connecting

#### 1.2.1 Connecting to the REST API

There are two ways to pass your API token using the REST API:

**Method 1: Pass the token directly within the request URL**

```python
import requests

headers = {
    'Content-Type': 'application/json'
}

requestResponse = requests.get(
    "https://api.tiingo.com/api/test?token=YOUR_API_TOKEN",
    headers=headers
)
print(requestResponse.json())
```

**Response:**
```json
{"message": "You successfully sent a request"}
```

**Method 2: Pass the token in your request headers**

```python
import requests

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token YOUR_API_TOKEN'
}

requestResponse = requests.get(
    "https://api.tiingo.com/api/test/",
    headers=headers
)
print(requestResponse.json())
```

#### 1.2.2 Connecting to the Websocket API

Websockets allow for two-way communication, allowing us to send data to you as soon as it's available. If you want real-time data, this is the fastest way to get it.

The websocket API functions differently as you "subscribe" and "unsubscribe" to data sources. From there, you will receive all updates as soon as they're received without having to make requests.

**Message Types:**
- `A` - New data
- `U` - Updating existing data
- `D` - Deleting existing data
- `I` - Informational/meta data
- `E` - Error messages
- `H` - Heartbeats (can be ignored for most cases)

**Subscription Format:**
```json
{
    "eventName": "subscribe",
    "eventData": {
        "authToken": "YOUR_API_TOKEN",
        "service": "test"
    }
}
```

**Python Websocket Example:**
```python
from websocket import create_connection
import simplejson as json

ws = create_connection("wss://api.tiingo.com/test")

subscribe = {
    'eventName': 'subscribe',
    'eventData': {
        'authToken': 'YOUR_API_TOKEN'
    }
}

ws.send(json.dumps(subscribe))

while True:
    print(ws.recv())
```

**Response:**
```json
{"data": {"subscriptionId": 61}, "response": {"message": "Success", "code": 200}, "messageType": "I"}
{"response": {"message": "HeartBeat", "code": 200}, "messageType": "H"}
```

The HeartBeat message is sent every 30 seconds to keep the connection alive.

Once you get your first data request back on a successful connection, the "data" attribute will contain a **subscriptionId**. This will be used for managing the connection (e.g., adding or removing tickers).

---

## 2. REST Endpoints

### Quick Reference

```
# Meta Data
https://api.tiingo.com/tiingo/daily/<ticker>

# Latest Price
https://api.tiingo.com/tiingo/daily/<ticker>/prices

# Historical Prices
https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=2012-1-1&endDate=2016-1-1
```

---

### 2.1 End-of-Day Prices

#### 2.1.1 Overview

Tiingo's End-of-Day prices use a proprietary error checking framework to help clean data feeds and also help catch missing corporate actions (splits, dividends, and exchange listing changes).

**Availability:**
- Most US Equity prices are available at 5:30 PM EST
- Exchanges may send corrections until 8 PM EST (we update prices as we obtain corrections)
- Mutual Fund NAVs are available after 12 AM EST

**Adjustments:**
Both raw prices and adjusted prices are available. The adjustment methodology follows the standard method set forth by "The Center for Research in Security Prices" (CRSP) in [CRSP Calculations](http://www.crsp.com/products/documentation/crsp-calculations). This methodology incorporates both split and dividend adjustments.

#### 2.1.2 End-of-Day Endpoint

**Endpoints:**
```
# Latest Price Information
https://api.tiingo.com/tiingo/daily/<ticker>/prices

# Historical Price Information
https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=2012-1-1&endDate=2016-1-1&format=csv&resampleFreq=monthly
```

##### Response Fields

| Field Name | JSON Field | Data Type | Description |
|------------|------------|-----------|-------------|
| Date | date | date | The date this data pertains to |
| Open | open | float | The opening price for the asset on the given date |
| High | high | float | The high price for the asset on the given date |
| Low | low | float | The low price for the asset on the given date |
| Close | close | float | The closing price for the asset on the given date |
| Volume | volume | int64 | The number of shares traded for the asset |
| Adj Open | adjOpen | float | The adjusted opening price for the asset on the given date |
| Adj High | adjHigh | float | The adjusted high price for the asset on the given date |
| Adj Low | adjLow | float | The adjusted low price for the asset on the given date |
| Adj Close | adjClose | float | The adjusted closing price for the asset on the given date |
| Adj Volume | adjVolume | int64 | The adjusted number of shares traded for the asset |
| Dividend | divCash | float | The dividend paid out on "date" (note: "date" will be the "exDate" for the dividend) |
| Split | splitFactor | float | The factor used to adjust prices when a company splits, reverse splits, or pays a distribution |

##### Request Parameters

| Field Name | Parameter | Data Type | Required | Description |
|------------|-----------|-----------|----------|-------------|
| Ticker | URL | string | Y | Ticker related to the asset |
| Start Date | startDate | date | N | If startDate or endDate is not null, historical data will be queried. Limits metrics to on or after the startDate (>=). Format: YYYY-MM-DD |
| End Date | endDate | date | N | If startDate or endDate is not null, historical data will be queried. Limits metrics to on or before the endDate (<=). Format: YYYY-MM-DD |
| Resample Freq | resampleFreq | string | N | Choose returned values as daily, weekly, monthly, or annually. Acceptable values: `daily`, `weekly`, `monthly`, `annually`. Note: ONLY DAILY takes into account holidays. |
| Sort | sort | string | N | Specify sort direction and column. Prepend "-" for descending order. E.g., `sort=date` (ascending), `sort=-date` (descending) |
| Response Format | format | string | N | Sets response format: "csv" or "json". Defaults to JSON |
| Columns | columns | string[] | N | Specify which columns to return. Pass array of column names |

**Resample Freq Details:**
- **daily**: Values returned as daily periods, with a holiday calendar
- **weekly**: Values returned as weekly data, with days ending on Friday
- **monthly**: Values returned as monthly data, with days ending on the last standard business day (Mon-Fri) of each month
- **annually**: Values returned as annual data, with days ending on the last standard business day (Mon-Fri) of each year

Note: If you choose a value in-between the resample period, the start date rolls back to consider the entire period. For example, if you choose to resample weekly but your "startDate" parameter is set to Wednesday, the startDate will be adjusted to Monday. Similarly, endDate rolls forward to capture the whole period.

##### Example Request

```python
import requests

headers = {
    'Content-Type': 'application/json'
}

requestResponse = requests.get(
    "https://api.tiingo.com/tiingo/daily/aapl/prices?startDate=2019-01-02&token=YOUR_API_TOKEN",
    headers=headers
)
print(requestResponse.json())
```

**Response:**
```json
[
    {
        "date": "2019-01-02T00:00:00.000Z",
        "close": 157.92,
        "high": 158.85,
        "low": 154.23,
        "open": 154.89,
        "volume": 37039737,
        "adjClose": 157.92,
        "adjHigh": 158.85,
        "adjLow": 154.23,
        "adjOpen": 154.89,
        "adjVolume": 37039737,
        "divCash": 0.0,
        "splitFactor": 1.0
    },
    {
        "date": "2019-01-03T00:00:00.000Z",
        "close": 142.19,
        "high": 145.72,
        "low": 142.0,
        "open": 143.98,
        "volume": 91312195,
        "adjClose": 142.19,
        "adjHigh": 145.72,
        "adjLow": 142.0,
        "adjOpen": 143.98,
        "adjVolume": 91312195,
        "divCash": 0.0,
        "splitFactor": 1.0
    }
]
```

---

### 2.1.3 Meta Endpoint

Our meta information comes from a variety of sources, but is used to help communicate details about an asset in our database to our users.

**For a daily list of all tickers accessible via Tiingo:** [supported_tickers.zip](https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip)

**Endpoint:**
```
https://api.tiingo.com/tiingo/daily/<ticker>
```

##### Response Fields

| Field Name | JSON Field | Data Type | Description |
|------------|------------|-----------|-------------|
| Ticker | ticker | string | Ticker related to the asset |
| Name | name | string | Full-length name of the asset |
| Exchange Code | exchangeCode | string | An identifier that maps which Exchange this asset is listed on |
| Description | description | string | Long-form description of the asset |
| Start Date | startDate | date | The earliest date we have price data available for the asset. When null it means no price data available |
| End Date | endDate | date | The latest date we have price data available for the asset. When null it means no price data available |

##### Example Request

```python
import requests

headers = {
    'Content-Type': 'application/json'
}

requestResponse = requests.get(
    "https://api.tiingo.com/tiingo/daily/aapl?token=YOUR_API_TOKEN",
    headers=headers
)
print(requestResponse.json())
```

**Response:**
```json
{
    "ticker": "AAPL",
    "name": "Apple Inc",
    "exchangeCode": "NASDAQ",
    "startDate": "1980-12-12",
    "endDate": "2019-01-25",
    "description": "Apple Inc. (Apple) designs, manufactures and markets mobile communication and media devices, personal computers, and portable digital music players, and a variety of related software, services, peripherals, networking solutions, and third-party digital content and applications. The Company's products and services include iPhone, iPad, Mac, iPod, Apple TV, a portfolio of consumer and professional software applications, the iOS and OS X operating systems, iCloud, and a variety of accessory, service and support offerings. The Company also delivers digital content and applications through the iTunes Store, App Store, iBookstore, and Mac App Store. The Company distributes its products worldwide through its retail stores, online stores, and direct sales force, as well as through third-party cellular network carriers, wholesalers, retailers, and value-added resellers."
}
```

---

## 5. Appendix

### 5.3 Symbology

This guide covers Tiingo's symbol format for querying different assets throughout the API.

#### 5.3.1 Introduction

Tiingo follows a symbology format to help disambiguate share classes, preferreds, and other types of assets.

#### 5.3.2 Equities

Right now the Tiingo API supports Equity, Mutual Fund, and ETF prices for US markets and Equity Prices for Chinese markets.

Since periods are often used in URLs to signify file extensions, we do not use periods in our symbol names. Instead of periods ("."), we use hyphens ("-"). This means "BRK.A" is "BRK-A" within Tiingo.

**Share Classes Format:**
```
{SYMBOL}-{SHARE CLASS}
```
Example: Berkshire Hathaway Class A shares are "BRK-A"

**Preferred Shares Format:**
```
{SYMBOL}-P-{SHARE CLASS}
```
Example: Simon Property Group's Preferred J series shares are "SPG-P-J"

**Other Notes:**
- Mutual Funds usually end with the letter "X" (e.g., "VFINX")
- Closed-End-Funds (CEFs) begin and end with the letter "X" (e.g., "XAIFX")
- Tiingo supports delisted data for tickers that have not yet been recycled

#### 5.3.3 CryptoCurrency & FX

Tiingo's methodology for CryptoCurrencies and FX currencies follow the same symbol method.

Because forward slashes ("/") are used to separate locations within URLs and directories, Tiingo does not use forward slashes within currency symbols. Instead we remove them.

**Currency Format:**
```
{BASE_CURRENCY}{QUOTE_CURRENCY}
```

Examples:
- "EUR/USD" → `EURUSD` (base: EUR, quote: USD)
- "BTC/ETH" → `BTCETH` (base: BTC, quote: ETH)

---

## Quick Reference

### Base URL
```
https://api.tiingo.com
```

### Common Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/test` | Test connection and authentication |
| `/tiingo/daily/<ticker>` | Get metadata for a ticker |
| `/tiingo/daily/<ticker>/prices` | Get latest/historical price data |

### Authentication

**Option 1 - Query Parameter:**
```
?token=YOUR_API_TOKEN
```

**Option 2 - Header:**
```
Authorization: Token YOUR_API_TOKEN
```

### Example: Get Historical Data for AAPL

```python
import requests

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token YOUR_API_TOKEN'
}

# Get historical prices
response = requests.get(
    "https://api.tiingo.com/tiingo/daily/aapl/prices",
    params={
        'startDate': '2024-01-01',
        'endDate': '2024-12-31',
        'resampleFreq': 'daily'
    },
    headers=headers
)

data = response.json()
```
