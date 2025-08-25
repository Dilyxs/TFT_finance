#!/bin/bash
source /root/venvs/py312/bin/activate
cd /root/TFT_trading
echo "$(date) Depolyement Closing All Trades" >> TradingClose.log
python3 TFT_trade_close.py  >> TradingClose.log 2>&1

