#!/bin/bash
source /root/venvs/py312/bin/activate
cd ~/TFT_trading
echo "Deplyement Closing All Trades"
python3 TFT_trade_close.py  >> TradingClose.log 2>&1

