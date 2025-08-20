import inspect
import pandas as pd
from datetime import datetime
import random
import asyncio
from metaapi_cloud_sdk import MetaApi
from metaapi_cloud_sdk.metaapi.models import PendingTradeOptions, StopOptions
import traceback


def MakeAnId():
    return random.randint(1, 2**53 - 1) 

class MetaApiConnection:
    def __init__(self, api_token, account_id):
        self.api_token = api_token
        self.account_id = account_id
        self.api = MetaApi(token=self.api_token)
        self.active_connections = []

    async def RetrieveAccountEquity(self):
        conn =await self.connect_to_account()
        re  = await conn.get_account_information()
        return re['equity']
    
    async def deploy_account(self):
        account = await self.api.metatrader_account_api.get_account(account_id=self.account_id)
        await account.deploy()
        return account

    async def undeploy_account(self):
        account = await self.api.metatrader_account_api.get_account(account_id=self.account_id)
        await account.undeploy()
        return account
        
    async def connect_to_account(self):
        if not self.active_connections:
            account = await self.api.metatrader_account_api.get_account(account_id=self.account_id)
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()
            self.active_connections.append(connection)
        else:
            connection = self.active_connections[-1]
        return connection

    async def disconnect_all_conn(self):
        while self.active_connections:
            conn = self.active_connections.pop()
            await conn.close()

    async def GetCurrentPrice(self, pair,side):
        """side: 0 for sell, 1 for buy"""
        if "_" in pair:
            pair = pair.replace("_", "")
        conn = await self.connect_to_account()
        res = await conn.get_symbol_price(pair)
        return res['bid'] if side==1 else res['ask']

    def DetermineTradeType(self, side, wanted_price, current_price):
        """
        side: 0 for sell, 1 for buy
        returns: string order type: 'limit' or 'stop_limit' depending on price and side
        """
        if side == 1:  # Buy
            return "limit" if wanted_price < current_price else "stop_limit"
        elif side == 0: 
            return "limit" if wanted_price > current_price else "stop_limit"
        else:
            raise ValueError("Side must be 0 (sell) or 1 (buy)")

    async def PlaceBuyOrder(self, pair, units, ExperationTime, EntryPrice, StopLoss, customId, comment=None):
        connection = await self.connect_to_account()
        current_price = await self.GetCurrentPrice(pair, side=1)
        trade_type = self.DetermineTradeType(side=1, wanted_price=EntryPrice, current_price=current_price)
    
        options = PendingTradeOptions(
            expiration={
                "type": "ORDER_TIME_SPECIFIED",
                "time": ExperationTime
            },
            magic=customId,
            comment=comment
        )
    
        try:
            if trade_type == "limit":
                trade_response = await connection.create_limit_buy_order(
                    symbol=pair,
                    volume=units,
                    open_price=EntryPrice,
                    stop_loss=StopLoss,
                    take_profit=None,
                    options=options
                )
            else:  # stop_limit
                trade_response = await connection.create_stop_limit_buy_order(
                    symbol=pair,
                    volume=units,
                    open_price=EntryPrice,
                    stop_limit_price=EntryPrice,
                    stop_loss=StopLoss,
                    take_profit=None,
                    options=options
                )
        except Exception as e:
            print("Exception type:", type(e))
            print("Exception args:", e.args)
            if hasattr(e, 'details'):
                print("Details:", e.details)
            else:
                import traceback
                traceback.print_exc()
            trade_response = None  # or you can re-raise or handle differently
    
        return trade_response['orderId']
    async def PlaceSellOrder(self, pair, units, ExperationTime, EntryPrice, StopLoss, customId, comment=None):
        connection = await self.connect_to_account()
        current_price = await self.GetCurrentPrice(pair, side=0)
        trade_type = self.DetermineTradeType(side=0, wanted_price=EntryPrice, current_price=current_price)
    
        options = PendingTradeOptions(
            expiration={
                "type": "ORDER_TIME_SPECIFIED",
                "time": ExperationTime
            },
            magic=customId,
            comment=comment
        )
    
        try:
            if trade_type == "limit":
                trade_response = await connection.create_limit_sell_order(
                    symbol=pair,
                    volume=units,
                    open_price=EntryPrice,
                    stop_loss=StopLoss,
                    take_profit=None,
                    options=options
                )
            else:  # stop_limit
                trade_response = await connection.create_stop_limit_sell_order(
                    symbol=pair,
                    volume=units,
                    open_price=EntryPrice,
                    stop_limit_price=EntryPrice,  # corrected param name
                    stop_loss=StopLoss,
                    take_profit=None,
                    options=options
                )
        except Exception as e:
            print("Exception type:", type(e))
            print("Exception args:", e.args)
            if hasattr(e, 'details'):
                print("Details:", e.details)
            else:
                import traceback
                traceback.print_exc()
            trade_response = None
    
        return trade_response['orderId']

    async def GetSingleOrderDetail(self, meta_id):
        conn = await self.connect_to_account()
        details =await conn.get_order(meta_id)

        return details
        

    async def ReturnAllOrders(self):
        conn = await self.connect_to_account()
        trades = await conn.get_orders() 
        return trades

    async def ReturnAllPositions(self):
        conn = await self.connect_to_account()
        res = await conn.get_positions()
        return res

    async def GetSpeficData(self, broker_id):
        conn = await self.connect_to_account()
        res = await conn.get_position(str(broker_id))
        return res
        
    async def CloseTradeByPosition(self, positionId):
        conn = await self.connect_to_account()
        res = await conn.close_position(str(int(positionId)))

        return res

    async def close_trades_by_magic(self, magic_number):
        connection = await self.connect_to_account()
        try:
            positions = await connection.get_positions()
  
            matching_positions = [pos for pos in positions if pos['magic'] == magic_number]
            
            for pos in matching_positions:
                options = {
                    "magic": magic_number,
                    "comment": f"Closing {magic_number}"
                }
                res = await connection.close_position(pos['id'], options=options)
                
            
            return res
        except Exception as e:
            print(f"Error closing trades with magic number {magic_number}: {e}")
            return 0




   
class MetaV2(MetaApiConnection):
    def __init__(self, api_token, account_id):
        super().__init__(api_token, account_id)

    def FormatTimeAccurately(self, given_time):
        """Converts datetime or pandas Timestamp to MetaApi time format"""
        if isinstance(given_time, pd.Timestamp):
            dt = given_time.to_pydatetime()
        elif isinstance(given_time, datetime):
            dt = given_time
        else:
            raise ValueError("GivenTime must be datetime or pandas.Timestamp")
        
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    async def SendOrderDetails(self, side, pair, units, ExperationTime, EntryPrice, StopLoss, customId, comment=None):
        pair = pair if not "_" in pair else pair.replace("_", "")
        """
        side: 0 = sell, 1 = buy
        """
        dt = self.FormatTimeAccurately(ExperationTime)

        if side == 1:
            order_id = await self.PlaceBuyOrder(pair, units, dt, EntryPrice, StopLoss, customId, comment=comment)
        elif side == 0:
            order_id = await self.PlaceSellOrder(pair, units, dt, EntryPrice, StopLoss, customId, comment=comment)
        else:
            raise ValueError("Side must be 0 (sell) or 1 (buy)")

        order_detail = await self.GetSingleOrderDetail(order_id)
        return order_detail
