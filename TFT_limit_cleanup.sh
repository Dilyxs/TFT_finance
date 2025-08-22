#!/bin/bash
source /root/venvs/py312/bin/activate
cd ~/TFT_trading
echo "Starting Removing Limits"
python3 TFT_limit_cleanup.py >> TradingLimitLogic.log 2>&1


