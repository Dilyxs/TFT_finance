import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from TFT_helper import (
    CombineWithSentiment, calculate_risk, insert_detected_trade, insert_limit_order,
    MakeAnId, insert_active_trade, map_limit_order_to_active_trade, ReturnMapWithPositions,
    PositionDoubleMapping, DetermineIfOrderIsFilled, ReturnWithTradeActivation
)
import pandas as pd 
import numpy as np
from MetaApiConn import MetaV2
from PostGresConn import PostgresSQL
from datetime import datetime, timedelta
import os
import asyncio


async def main():
    # cleanup for limit orders
    db = PostgresSQL()

    all_limit_orders = db.FetchAllData("limitorderstable")
    AccountDataMap = {}
    MetaAccounts = {}

    tries = 0
    Success = False
    while not Success and tries < 3:
        try:
            PositionDoubleMap = await PositionDoubleMapping()
            Success = True
        except Exception as e:
            tries += 1
            continue

    if Success:
        for order in all_limit_orders:
            Identifiable_id = order['id_broker']
            print(f"started processing {Identifiable_id}")
            account_id = order['account_id']
            trade_end_time = order.get('tradeexpiration', None)
            time_left = order.get('expirationtimelimit')
            TradeTimeGone = True if datetime.utcnow() > time_left else False  # past expiration?
    
            if account_id not in AccountDataMap:
                data = db.FetchSpecificData("accountdata", f"WHERE account_id='{account_id}'")
                AccountDataMap[account_id] = data[0].get('access_token', None)
            access_token = AccountDataMap[account_id]
    
            # make an Object
            if account_id in MetaAccounts:
                meta = MetaAccounts[account_id]
            else:
                meta = MetaV2(access_token, account_id)
                MetaAccounts[account_id] = meta
    
            IsFilled = DetermineIfOrderIsFilled(Identifiable_id, PositionDoubleMap, account_id, access_token)
    
            if IsFilled:
                trade_dict = map_limit_order_to_active_trade(order)
                trade_code = db.InsertData('activetrades', trade_dict)
    
                if trade_code == 200:
                    print("trade inserted")
                else:
                    print("error inserting data into active trades")
    
            RemoveDict = {'id_broker': Identifiable_id}
    
            # if it's out of time or got filled → remove from limit orders
            if TradeTimeGone or IsFilled:
                trade_code = db.DeleteSpecificData("limitorderstable", RemoveDict)
                if trade_code == 200:
                    print("trade deleted on limit orders table")
                else:
                    print("error deleting from limit orders table")
    
            await meta.disconnect_all_conn()
            
    for value in MetaAccounts.values():
        meta = value
        await meta.disconnect_all_conn()

asyncio.run(main())

