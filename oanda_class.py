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
from mgdb_trade_tracker import DataDB_tracker
from datetime import timedelta
import pytz
import logging
import warnings
from all_trades_db import DataDB_saved_trades
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
        self.db = DataDB_tracker()
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




class OANDA_DB_Manager(OANDAClientParent):
    def __init__(self, access_token, account_id):
        super().__init__(access_token, account_id)
        self.quick_db = DataDB_saved_trades()


    def put_entire_trade_on_db(self, trade):
        close_price = self.get_current_price(trade['pair'])
        trade['close_price'] = close_price
    
        direction = trade['direction']
        entry_price = trade['price']
        trade['p_l'] = close_price - entry_price if direction == 1 else entry_price - close_price
    
        self.quick_db.add_one(DataDB_saved_trades.FOREX_COLL,trade)


    def cleanup_unfilled_trades(self):
        """Delete unfilled trades that are older than 3 days."""
        try:
            trade_list = self.db.query_all(DataDB_tracker.FOREX_COLL, filled="false")
            expired_trades = [i for i in trade_list if (self.now - i['execution_time']).days >= 3]
    
            if expired_trades:
                self.db.delete_many(DataDB_tracker.FOREX_COLL, filled="false", execution_time={"$lte": self.now - timedelta(days=3)})
                print(f"Deleted {len(expired_trades)} expired unfilled trades.")
                logging.info(f"Deleted {len(expired_trades)} expired unfilled trades.")
            else:
                print("No expired unfilled trades to delete.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            logging.error(f"Error during cleanup: {e}")

        
    def collect_trades_oanda(self):
        r = trades.OpenTrades(self.account_id)
        response = self.client.request(r)
        all_trades = [i for i in response['trades']]
        return all_trades


    def check_for_order_completion(self):
        trade_list = self.db.query_all(DataDB_tracker.FOREX_COLL, filled="false")
        active_trades = self.collect_trades_oanda()
    
        built_seen_trades = [[i['pair'], i['units']] for i in trade_list]

        for active_trade in active_trades:
            built_key = [
                active_trade["instrument"],
                str(active_trade["currentUnits"]),
            ]
    
            if built_key in built_seen_trades:
                self.db.update_one(
                    DataDB_tracker.FOREX_COLL,
                    {"pair": active_trade["instrument"], "units": str(active_trade["currentUnits"])},
                    "filled",
                    "true",
                )
                logging.info(f"Trade {active_trade['id']} marked as filled.")

                
    def check_if_close_trade_time(self):
        
        trades = self.db.query_all(DataDB_tracker.FOREX_COLL, filled = "true")
        trades_that_need_2_be_closed = []

        now = self.now.date()
        
        for trade in trades:
            trading_days_elapsed = self.count_trading_days( trade["execution_time"], datetime.datetime.utcnow())

            if trading_days_elapsed >= 10: #more than 2 weeks
                trades_that_need_2_be_closed.append(trade)

        return trades_that_need_2_be_closed


class OANDAExecuter(OANDAClientParent):
    def __init__(self, access_token, account_id):
        super().__init__(access_token, account_id)  

    def convert_response_for_db(self, response):

        execution_time = self.now.replace(minute=50, second=0, microsecond=0, hour= 20)
        try:
            impor = response['orderCreateTransaction']
            data = {
                'pair': impor['instrument'],
                'execution_time': execution_time, 
                'type': impor['type'],
                'direction': 1 if float(impor['units']) > 0 else -1,
                'units': impor['units'],
                'price': float(impor['price']),
                'id_trade': float(impor['id']),
                'filled': "false"
            }         
            return data
        except Exception as e:
            logging.info(f"Error converting response for DB: {e}")

    def place_order(self, instrument, direction, risk_percentage=0.50):
        """
        Places an order. If the market is closed, a limit order is placed with an expiry before Monday 1:15 AM UTC.
        direction: 1 for long, -1 for short.
        """
        position = self.calculate_position_size(currency_pair=instrument, risk_percentage=risk_percentage)
        if position is None:
            logging.info("Failed to calculate position size.")
            return None

        current_price = self.get_current_price(instrument)
        if current_price is None:
            logging.info("Failed to fetch current price.")
            return None

        position *= -1 if direction == -1 else 1  

        order_type = "LIMIT"
        price = str(current_price)  

        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)


        if self.is_friday:
            expiry_time = self.nxt_monday #if it's friday -> give market wekend until set friday time
        else:
            expiry_time = now + timedelta(hours=10)  # Order must get filled in 10 hours

        time_in_force = "GTD"
        gtd_time = expiry_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")  

        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(position),
                "type": order_type,
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT",
                "price": price,
                "gtdTime": gtd_time
            }
        }

        try:
            r = orders.OrderCreate(self.account_id, data=order_data)
            response = self.client.request(r)
            print(response)
            logging.info(response)
            logging.info(f"Order placed: {instrument}")

            data = self.convert_response_for_db(response)
            self.db.add_one(DataDB_tracker.FOREX_COLL, data)

            return response
        except V20Error as e:
            logging.error(f"Error placing order: {e}, OANDA said no")
            return None
    def close_trades(self, trades_to_be_closed, class_user):
        
        try:
            r = trades.OpenTrades(self.account_id)
            response = self.client.request(r)
            open_trades = response.get("trades", [])
    
            extracted_trades_db = [[i['pair'], str(i['units'])] for i in trades_to_be_closed]
    
            for trade in open_trades:
                print(trade)
                trade_details = [trade["instrument"], str(trade["currentUnits"])]
                if trade_details in extracted_trades_db:
                    try:
                        close_request = trades.TradeClose(self.account_id, tradeID=trade["id"])  # Pass trade_id
                        close_response = self.client.request(close_request)
                        
                        if "errorMessage" in close_response:
                            print(f"Failed to close trade {trade['id']}: {close_response['errorMessage']}")
                            logging.error(f"Failed to close trade {trade['id']}: {close_response['errorMessage']}")
                            continue
    
                        self.db.delete_one_filter(DataDB_tracker.FOREX_COLL, {'pair':trade['instrument'], 'units':trade['currentUnits']})
                        print(f"Trade closed for {trade['instrument']}")

                        final_trade = {
                        'pair': trade['instrument'],
                        'direction': 1 if int(trade['currentUnits']) > 0 else -1,
                        'units': trade['currentUnits'],
                        'price': float(trade['price']),
                        'id_trade': trade['id'],
                        'filled': 'true'}
                        
                        
                        class_user.put_entire_trade_on_db(final_trade)
                        logging.info(f"Trade closed for {trade['instrument']}")
                    
                    except V20Error as e:
                        print(f"Error closing trade {trade['id']}: {e}")
                        logging.error(f"Error closing trade {trade['id']}: {e}")
        except V20Error as e:
            print(f"Error retrieving open trades: {e}")
            logging.error(f"Error retrieving open trades: {e}")
