#!/bin/bash
source /root/venvs/py312/bin/activate
cd /root/TFT_trading
echo "$(date) Starting Removing Limits" >> TradingLimitLogic.log
python3 TFT_limit_cleanup.py >> TradingLimitLogic.log 2>&1


