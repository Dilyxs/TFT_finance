import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from oanda_class import OANDAClientParent
from TFT_helper import CombineWithSentiment, calculate_risk, insert_detected_trade, insert_limit_order, MakeAnId, insert_active_trade, map_limit_order_to_active_trade, ReturnMapWithPositions, PositionDoubleMapping, DetermineIfOrderIsFilled, ReturnWithTradeActivation,save_map_to_file
import pandas as pd 
import numpy as np
from MetaApiConn import MetaV2
from PostGresConn import PostgresSQL
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import asyncio
from ReSendClass import EmailSender
load_dotenv()
Oanda = OANDAClientParent(os.getenv('OANDA_ACCESS_TOKEN'), os.getenv('OANDA_ACCOUNT_ID'))
db = PostgresSQL()

pairs = ["AUD_USD", "NZD_USD", "EUR_USD", "USD_CAD", "USD_CHF", "USD_MXN", "GBP_USD", "USD_JPY", "USD_ZAR"]
IsCorrectTime = True if 0 <= datetime.utcnow().minute < 10 else False
MetaMap = {}
MapCurr = {}
async def main():
    for pair in pairs:
        RunCorrectly = False
        tries = 0
        while not RunCorrectly and tries < 5:
            try:
                res = ReturnWithTradeActivation(pair) 
                
                reg_risk, trade_signal = calculate_risk(pair, res, base_risk=0.3)                       
                pip_size = 0.0001 if not pair.endswith(("JPY")) else 0.01
                MapCurr[pair] = trade_signal

                if trade_signal is None:
                    continue
                print(f"Trade Detected with {pair} at direction {trade_signal}")  
                curr = res['close'].values[0]
                currentPrice = Oanda.get_current_price(pair)
                n_digits = 4 if not pair.endswith(("JPY")) else 2
                if curr == round(currentPrice, n_digits):  # need to move it up
                    if trade_signal == 1:
                        currentPrice -= 2 * pip_size
                    else:
                        currentPrice += 2 * pip_size
                
                trade_details, DetectedTrades = insert_detected_trade(db, pair, curr, trade_signal, reg_risk)
                
                s = 200
                StopLoss = (curr - (s * pip_size)) if trade_signal == 1 else (curr + (s * pip_size))
                CustomId = MakeAnId()
                
                # API CONNECTION TO ACTUALLY SEND IN ORDER!
                All_accounts = db.FetchAllData("accountdata")
            
                for acc in All_accounts:
                    if acc['account_id'] in MetaMap:
                        Meta = MetaMap[acc['account_id']]
                    else:
                        Meta = MetaV2(acc['access_token'], acc['account_id'])
                        MetaMap[acc['account_id']] = Meta
                    equity = await Meta.RetrieveAccountEquity()
                    units = Oanda.calculate_position_size(pair, reg_risk, balance=equity, lots=True)
                    
                    # this actually sends the trade
                    order_details = await Meta.SendOrderDetails(
                        trade_signal, pair, units, DetectedTrades.LimitClosure, curr, StopLoss, CustomId, comment=None
                    )
                    print(order_details)
                    
                    trade_code = insert_limit_order(
                        db, order_details, acc['account_id'], trade_details['expectedclosetime'], trade_signal
                    )
                    
                    await Meta.disconnect_all_conn()
                    RunCorrectly = True
            except Exception as e:
                print(e)
                continue
            tries += 1
    path=save_map_to_file(MapCurr)

    Sender = EmailSender(prefix="TFTtrader")
    Sender.SendEmail(subject='TFT_trading for the day', "here is the txt file with results for the day", 'adsayan206@gmail.com', path)

if IsCorrectTime: 
    asyncio.run(main())

