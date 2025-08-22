source /root/venvs/py312/bin/activate
cd ~/TFT_trading
echo "Deployement Begins"
python3 TFT_trading_infra.py  >> TradingLogic.log 2>&1

