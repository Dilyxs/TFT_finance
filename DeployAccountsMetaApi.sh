#!/bin/bash

source /root/venvs/py312/bin/activate
cd /root/TFT_trading
echo "$(date)----Deployement Begins" >> DeploymentMeta.log
python3 DeployAccountsMetaApi.py  >> DeploymentMeta.log 2>&1
