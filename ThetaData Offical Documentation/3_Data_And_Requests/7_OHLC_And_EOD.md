OHLC & EOD
Mechanics
OHLCVC or open, high, low, close, volume, count is calculated using trades data. Some trades should be completely ignored from these calculations. For instance, a stock trading a $100 could have a trade reported at a price of $90 on the SIP feed. The most common cause of these suspect trade prices are late reports. Properly ignoring these trades based off the trade conditions is imperative to generating OHLC reports that are usable.

Types of OHLC Bars
There are various methods of calculating OHLC. Theta Data V2 requests use method 1, which we believe to be the standard method of market participants.

Following the SIP rules for calculating OHLC. Depending on the trade's condition, it might not be eligible to be used in the OHLC and or volume calculations. Certain trades could be eligible to update the last price of the session but not the volume. Providers like Theta Data might occasionally have lower volumes than less credible sources as those sources aren't properly filtering out trades based on the condition.
Taking every trade reported by the SIP for the day and applying basic min / max logic to calculate OHLC. Sometimes this can produce unexpected datapoints such as extreme highs or lows significantly outside the expected high / low of the session. We discourage users from utilizing this method. It is important to also note that improperly filtering trades data for volume can result in higher volumes than there should be.
A mixture of method 1 and 2: the trade's condition is ignored if the trade price is within a reasonable distance from the NBBO or last trade. This method can grow in complexity such as considering moving average and average trade price changes.
Types of EOD Reports
There are various methods of calculating EOD data. Theta Data V2 requests use method 2.

The two equity SIPS, UTP & CTA each generate an EOD report at ~17:00 ET. Theta Data will be exposing these reports as well as creating a merged report in the near future. The single option SIP, OPRA publishes EOD reports on a per participant exchange basis. We have found these reports to be unreliable as certain exchanges won't have EOD reports when they should. We believe that it is not practical to generate a nationally consolidated EOD report for options using this information, which leads us to method 2.
Since there is no nationally reported EOD for either stocks, indices, or options, Theta Data generates a normalized EOD Report at 17:15 ET each day. The behavior of the EOD report is the same for each asset class, making cross-referencing EOD data much easier.
Notable Behavior
If you see "missing" EOD data or zero-ed OHLC bars for less liquid option contract (i.e. something deeply out of or in the money), that means there were no trades for that contract during that time.

Calculate your own OHLC
You can calculate OHLC data by looking at each trade and the trade's condition.