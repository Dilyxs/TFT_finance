#!/bin/bash

source ~/venv/bin/activate
cd ~/TFT_trading
echo "Deployement Begins"
python3 DeployAccountsMetaApi.py  >> DeploymentMeta.log 2>&1
