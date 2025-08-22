source ~/venv/bin/activate
cd ~/TFT_trading
echo "Deployement Begins"
python TFT_trading_infra.py  >> TradingLogic.log 2>&1

