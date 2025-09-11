#!/bin/bash
echo "$(date) -- Deployment Begins -- " >> /root/TFT_trading/DeploymentMeta.log 2>&1
cd /root/TFT_trading

DATA=$(/root/venvs/py312/bin/python DeployAccountsMetaApi.py 2>&1)

if echo "$DATA" | grep -q "Error"; then
    echo "$DATA" | grep "Error" >> /root/TFT_trading/DeploymentMeta.log
else
    echo "No error." >> /root/TFT_trading/DeploymentMeta.log 2>&1
fi

echo "$(date) -- Deployment Finished -- " >> /root/TFT_trading/DeploymentMeta.log 2>&1

