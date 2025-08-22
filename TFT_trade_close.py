# closing trades
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
    db = PostgresSQL()
    all_trades = db.FetchAllData("activetrades")
    now = datetime.utcnow()
    AccountDataMap = {}
    MetaAccounts = {}

    for trade in all_trades:
        ExpirationTime = trade['tradeexpiration']

        if now > ExpirationTime:  # close the trade
            Identifiable_id = trade['id_broker']
            pair = trade['pair']
            account_id = trade['account_id']

            if account_id not in AccountDataMap:
                data = db.FetchSpecificData("accountdata", f"WHERE account_id='{account_id}'")
                AccountDataMap[account_id] = data[0].get('access_token', None)

            access_token = AccountDataMap[account_id]

            if account_id in MetaAccounts:
                meta = MetaAccounts[account_id]
            else:
                meta = MetaV2(access_token, account_id)
                MetaAccounts[account_id] = meta

            try:
                tries = 0
                Success = False
                while not Success and tries < 3:
                    try:
                        await meta.CloseTradeByPosition(Identifiable_id)  # error-prone
                        print("position closed")
                        Success = True
                    except Exception as e:
                        tries += 1
                        continue

            except NotFoundException as e:  # this just means SL got hit
                print("either shit went wrong or stop loss got hit")
                continue

            # now remove from activetrades table
            DataDict = {'id_broker': Identifiable_id}
            trade_code = db.DeleteSpecificData('activetrades', DataDict)
            await meta.disconnect_all_conn()

    # cleanup all Meta connections
    for value in MetaAccounts.values():
        meta = value
        await meta.disconnect_all_conn()

        

asyncio.run(main())



