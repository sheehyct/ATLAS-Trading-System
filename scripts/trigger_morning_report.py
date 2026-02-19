"""One-off script to trigger the morning report outside of the scheduler."""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                os.environ.setdefault(key.strip(), value.strip())

from strat.signal_automation.coordinators.morning_report import MorningReportGenerator
from strat.signal_automation.config import MorningReportConfig
from strat.signal_automation.alerters.discord_alerter import DiscordAlerter

webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
if not webhook_url:
    print("ERROR: DISCORD_WEBHOOK_URL not set in .env")
    sys.exit(1)

config = MorningReportConfig()
alerter = DiscordAlerter(webhook_url=webhook_url)

gen = MorningReportGenerator(alerters=[alerter], config=config)
print("Starting morning report generation...")
gen.run()
print("Morning report complete.")
