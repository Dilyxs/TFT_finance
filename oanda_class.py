import pandas as pd
import numpy as np
import tpqoa
import datetime
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.pricing as pricing
from datetime import timedelta
import pytz
import logging
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

logging.basicConfig(filename="Oanda_logging.log",
                            level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

class OANDAClientParent:
    def __init__(self, access_token, account_id, environment="practice"):
        """
        Initialize the OANDA API client.
        :param access_token: (str) Your OANDA API token.
        :param account_id: (str) Your OANDA account ID.
        :param environment: (str) "practice" for demo, "live" for real trading.
        """
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=access_token, environment=environment)
        self.now = datetime.datetime.utcnow()
        self.is_friday = self.now.weekday() == 4
        self.nxt_monday = self.find_nxt_monday()

    def find_nxt_monday(self):
        days_ahead = 0 - self.now.weekday()
        if days_ahead <= 0:
            days_ahead += 7  # Move to the next Monday if today is Monday or later in the week

        next_monday = self.now + timedelta(days=days_ahead)
        next_monday_9am = next_monday.replace(hour=9, minute=0, second=0, microsecond=0)

        return next_monday_9am

    def count_trading_days(self, start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        return len(date_range) - 1

    def get_last_friday_execution_time(self):
        """Returns the last Friday at 21:01 UTC in ISO format."""
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        last_friday = now - timedelta(days=(now.weekday() + 3) % 7)
        execution_time = last_friday.replace(hour=21, minute=1, second=0, microsecond=0)
        return execution_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")

    def is_market_closed(self):
        """Check if current time is between Friday 4:30 PM and Monday 1 AM UTC."""
        if (self.now.weekday() == 4 and self.now.hour >= 16 and self.now.minute >= 30) or \
           (self.now.weekday() == 5 or self.now.weekday() == 6) or \
           (self.now.weekday() == 0 and self.now.hour < 1):
            return True
        return False

    def get_account_equity(self):
        """
        Fetch the account equity (balance including unrealized P/L).
        :return: (float) Account equity.
        """
        try:
            r = accounts.AccountSummary(self.account_id)
            response = self.client.request(r)
            return float(response["account"]["NAV"])
        except V20Error as e:
            logging.info(f"Error fetching account equity: {e}")
            return None

    def get_current_price(self, currency_pair):
        """Get current price for the given currency pair"""
        try:
            params = {'instruments': currency_pair}
            request = pricing.PricingInfo(self.account_id, params=params)
            response = self.client.request(request)
            price = float(response['prices'][0]['closeoutBid'])
            return price
        except V20Error as e:
            logging.info(f"Error fetching current price for {currency_pair}: {e}")
            return None

    def calculate_position_size(self, currency_pair, risk_percentage=0.5, balance=None, lots=False):
        """Calculate position size based on account equity and stop loss."""
        if balance is None:
            balance = self.get_account_equity()
            if balance is None:
                return None

        current_price = self.get_current_price(currency_pair)
        if current_price is None:
            return None

        b = balance
        r = risk_percentage / 100
        s = 200  # always 200 pips
        p = current_price
        pip_size = 0.0001 if not currency_pair.endswith(("JPY", "TRY")) else 0.01

        # Account currency is USD & we only trade USD pairs
        if currency_pair.endswith("USD"):  # account currency == quote currency
            C = 1
        else:
            C = 1 / p

        units = (b * r) / (s * pip_size * C)

        if not lots:
            return round(units, 0)
        else:
            lots_value = units / 100_000
            lots_value = round(lots_value, 2)
            return lots_value




