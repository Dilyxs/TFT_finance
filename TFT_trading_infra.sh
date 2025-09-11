#!/bin/bash
echo "$(date) -- Deployment Begins -- " >> /root/TFT_trading/TradingLogic.log 2>&1
cd /root/TFT_trading

DATA=$(timeout 20m /root/venvs/py312/bin/python TFT_trading_infra.py 2>&1)

if echo "$DATA" | grep -q "Error"; then
    echo "$DATA" | grep "Error" >> /root/TFT_trading/TradingLogic.log
else
    echo "No error." >> /root/TFT_trading/TradingLogic.log 2>&1
fi

echo "$(date) -- Deployment Finished -- " >> /root/TFT_trading/TradingLogic.log 2>&1

