#!/bin/bash
source /root/venvs/py312/bin/activate
cd /root/TFT_trading
echo "$(date) Deployement Begins">> TradingLogic.log
timeout 20m python3 TFT_trading_infra.py  >> TradingLogic.log 2>&1

