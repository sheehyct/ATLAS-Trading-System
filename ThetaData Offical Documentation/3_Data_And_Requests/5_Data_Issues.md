Causes of Perceived Data Issues
Theta Data employs a whole suite of data integrity checks to verify that both historical and real-time data is whole and accurate. It is not uncommon to encounter missing data when you might otherwise expect there to be data. Some but not all mitigating factors are market holidays, trading halts, SIP / exchange processing issues, and liquidity. This article will outline our checklist we use for determine if missing data is to be expected based on the aforementioned mitigating circumstances.

WARNING

If you have read through these items and still believe you see an issue with data, please make a #support ticket on our Discord Server.

General
Missing OHLC, EOD, or trades? Not all contracts / symbols are traded every day or at every time.
Was the market closed? I.e. a special holiday, weekend, or event that caused the market to be closed?
Trading could have been halted on the date in question.
Data from the previous day isn't available from 00:00 ET to 01:45 ET. We have plans to make our midnight reset instant in the near future.
Are you using a V1 or V2 requests? This documentation and the current version of the Theta Terminal is V3. Ensure you are running the latest version of the terminal.
There could have been an issue with the SIP or exchange for the date in question. Refer to the CBOE system notices for OPRA options, CTA alerts for CTA equities, UTP alerts for UTP equities.
If requesting data outside regular trading hours, use the start_time and end_time parameters to specify the time range for the request.
Equities
Although Nasdaq Basic venue will send a zero-ed BBO quote at session close, we remove these quotes / do not process. This prevents snapshots and other processes returning a zero-ed value instead of the last legitimate value reported.
Options Specific
Most option contracts are listed 4-12 weeks before expiration. It is unlikely for a option with a weekly expiration to have multiple years or quarters of data. Contracts that start to get quoted 1-2 months before expiration isn't an indication of missing data but it just means they weren't traded / didn't exist at that time.
If SPX, there won't be data the same day as expiration since SPX options are AM settled. SPXW options are PM settled.
If an index option, make sure you are using the proper index option symbols that OPRA reports.
Greeks data? Check our Making Requests article to ensure the history you're requesting falls within the expected range and there are no expected cases where data might not be there.
Make sure the expiration and strike you're using is a valid expiration from the list contracts, list expirations, or list strikes endpoints.
Strike prices are represented in 10ths of a cent, so $140 would be 140000. When making requests, it is important to use this format when passing the strike price parameter.
If the option contract is 0DTE, deep ITM / OTM strikes might not be quoted / traded.
OPRA sends quotes with zero bid and or asks during premarket. This is a normal behavior with options quotes.
Data with the expiration years of 1882 is test data sent by OPRA. This typically doesn't happen anymore but for earlier years is a normal occurrence.
Prior to 2022-05-16, SPXW (SPX weekly) options were quoted Monday, Wednesday, Friday, not every day of the week.
For 0DTE options, especially index options, deep OTM or ITM contracts that have strike prices that aren't divisible by a certain number (i.e. 2.5, 5, 10) stop getting quoted and traded towards the end of the day.
Greeks may not work as expected and are unsupported outside regular trading hours. Option quotes and underlying prices stop and start updating at different times from the venues, which may cause unexpected values.