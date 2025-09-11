#!/bin/bash
echo "$(date) -- Deployment Closing All Trades -- " >> /root/TFT_trading/TradingClose.log 2>&1
cd /root/TFT_trading

DATA=$(/root/venvs/py312/bin/python TFT_trade_close.py 2>&1)

if echo "$DATA" | grep -q "Error"; then
    echo "$DATA" | grep "Error" >> /root/TFT_trading/TradingClose.log
else
    echo "No error." >> /root/TFT_trading/TradingClose.log 2>&1
fi

echo "$(date) -- Finished Closing All Trades -- " >> /root/TFT_trading/TradingClose.log 2>&1

