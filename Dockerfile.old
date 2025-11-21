# ============================================================================
# STRAT Trading System - Production Dockerfile
# ============================================================================
# Multi-stage build for optimized production container
# Security-hardened with non-root user and minimal attack surface
# Optimized for VectorBT Pro performance and UV package management
# ============================================================================

ARG PYTHON_VERSION=3.12.11-slim

# ============================================================================
# STAGE 1: Build Dependencies
# ============================================================================
FROM python:${PYTHON_VERSION} as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy

# Install system dependencies and UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv==0.4.10

# Create application user and directories
RUN groupadd --gid 1000 stratuser && \
    useradd --uid 1000 --gid stratuser --shell /bin/bash --create-home stratuser

# Set working directory
WORKDIR /app

# Copy dependency files first for better cache utilization
COPY --chown=stratuser:stratuser pyproject.toml uv.lock ./
COPY --chown=stratuser:stratuser environments/ ./environments/
COPY --chown=stratuser:stratuser shared/ ./shared/

# Install dependencies using UV (dev deps excluded via default-groups = [])
RUN uv sync --frozen

# ============================================================================
# STAGE 2: Runtime Image
# ============================================================================
FROM python:${PYTHON_VERSION} as runtime

# Security and performance environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/environments/backtesting/src" \
    UV_SYSTEM_PYTHON=1 \
    STRAT_ENVIRONMENT=production \
    STRAT_LOG_LEVEL=INFO \
    STRAT_SECURITY_MODE=strict

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user (must match builder stage)
RUN groupadd --gid 1000 stratuser && \
    useradd --uid 1000 --gid stratuser --shell /bin/bash --create-home stratuser

# Create application directories with proper permissions
RUN mkdir -p /app/{logs,data,cache,backtest_results} && \
    chown -R stratuser:stratuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=stratuser:stratuser /app/.venv /app/.venv

# Copy application code
COPY --chown=stratuser:stratuser . .

# Create required directories and set permissions
RUN mkdir -p logs data/historical cache backtest_results && \
    chown -R stratuser:stratuser /app && \
    chmod 755 /app && \
    chmod -R 750 /app/logs /app/data /app/cache /app/backtest_results

# Copy health check script
COPY --chown=stratuser:stratuser docker/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER stratuser

# Expose application ports
EXPOSE 8000 8001 8501

# Add UV and Python to PATH for the stratuser
ENV PATH="/app/.venv/bin:$PATH"

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Default command - can be overridden
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================================================
# STAGE 3: Development Image (Optional)
# ============================================================================
FROM runtime as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv==0.4.10

# Switch back to application user
USER stratuser

# Install development dependencies (includes dev group)
RUN uv sync --group dev

# Development environment variables
ENV STRAT_ENVIRONMENT=development \
    STRAT_LOG_LEVEL=DEBUG \
    STRAT_SECURITY_MODE=development

# Development command
CMD ["python", "main.py"]

# ============================================================================
# Build Arguments and Labels
# ============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL maintainer="STRAT Development Team" \
      org.label-schema.build-date=${BUILD_DATE} \
      org.label-schema.name="STRAT Trading System" \
      org.label-schema.description="Production-ready algorithmic trading system with VectorBT Pro" \
      org.label-schema.url="https://github.com/strat-trading/vectorbt-workspace" \
      org.label-schema.vcs-ref=${VCS_REF} \
      org.label-schema.vcs-url="https://github.com/strat-trading/vectorbt-workspace" \
      org.label-schema.vendor="STRAT Trading" \
      org.label-schema.version=${VERSION} \
      org.label-schema.schema-version="1.0"