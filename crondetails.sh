#!/bin/bash

# === DeployAccountsMetaApi.sh ===
# Runs at 4:48 PM and 5:18 PM Tuesday–Friday
48 16,17 * * 2-5 /root/TFT_trading/DeployAccountsMetaApi.sh
18 17 * * 2-5 /root/TFT_trading/DeployAccountsMetaApi.sh

# Runs at 3:50 AM and 4:00 AM Monday only
50 3 * * 1 /root/TFT_trading/DeployAccountsMetaApi.sh
0 4 * * 1 /root/TFT_trading/DeployAccountsMetaApi.sh

# === TFT_limit_cleanup.sh ===
# Runs at 4:51 PM Monday–Friday
51 16 * * 1-5 /root/TFT_trading/TFT_limit_cleanup.sh

# === TFT_trade_close.sh ===
# Runs at 4:51 PM Monday–Friday
51 16 * * 1-5 /root/TFT_trading/TFT_trade_close.sh

# === TFT_trading_infra.sh ===
# Runs at 5:02 PM Tuesday–Friday
2 17 * * 2-5 /root/TFT_trading/TFT_trading_infra.sh

# Runs at 3:52 AM Monday
52 3 * * 1 /root/TFT_trading/TFT_trading_infra.sh

