Option-Greeks
Model
Theta Data uses the Black Scholes for all Greeks calculations. More specifically, we use the formulas outlined here.

Tick by Tick
A tick is defined as a row of data. Theta Data calculates Greeks for each tick of data and uses the exact underlying tick (price) at the time of the option tick.

Implied Volatility Guess
We use a fast bisection method to calculate implied volatility. If a close to perfect solution does not exist, you will notice the iv_error value in Greeks requests will begin to increase. This happens with deep in the money or out of the money contracts. This behavior can be found with other models / data providers.

Weird Values?
Let us know, but before you do realize:

Rho & Vega must be divided by 100 to get their value.

We use the EU pricing model outlined above.

Parameters
Dividends
Theta Data currently ignores dividends, however you can specify annual_div parameter and Theta Data will use that as the annual dividend amount and calculate the dividend yield for each tick of data, which would then be plugged into the Black Scholes formula.

Rates
By default, Theta Data uses SOFR as its interest rate. SOFR is reported 1 day after the report date. Theta Data uses the last provided SOFR rate for current day data Greeks calculations. You can specify the rate parameter to override this behavior.