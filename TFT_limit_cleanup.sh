#!/bin/bash
echo "$(date) -- Starting Removing Limits -- " >> /root/TFT_trading/TradingLimitLogic.log 2>&1
cd /root/TFT_trading

DATA=$(/root/venvs/py312/bin/python TFT_limit_cleanup.py 2>&1)

if echo "$DATA" | grep -q "Error"; then
    echo "$DATA" | grep "Error" >> /root/TFT_trading/TradingLimitLogic.log
else
    echo "No error." >> /root/TFT_trading/TradingLimitLogic.log 2>&1
fi

echo "$(date) -- Finished Removing Limits -- " >> /root/TFT_trading/TradingLimitLogic.log 2>&1

