# Custom Deephaven image with Alpaca dependencies for ATLAS portfolio tracking
# Using 0.37.0+ for plotly-express plugin compatibility
FROM ghcr.io/deephaven/server:0.37.0

# Install Python dependencies needed for portfolio tracker
RUN pip install --no-cache-dir \
    alpaca-py>=0.42.0 \
    alpaca-trade-api>=3.2.0 \
    python-dotenv>=1.1.1 \
    requests>=2.32.4 \
    deephaven-plugin-plotly-express
