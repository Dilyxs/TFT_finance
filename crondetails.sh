#!/bin/bash

# === DeployAccountsMetaApi.sh ===
# Runs at 21:48 and 22:18 UTC Tuesday–Friday
48 21 * * 2-5 /root/TFT_trading/DeployAccountsMetaApi.sh
18 22 * * 2-5 /root/TFT_trading/DeployAccountsMetaApi.sh

# Runs at 08:50 and 09:00 UTC Monday only
50 8 * * 1 /root/TFT_trading/DeployAccountsMetaApi.sh
0 9 * * 1 /root/TFT_trading/DeployAccountsMetaApi.sh

# === TFT_limit_cleanup.sh ===
# Runs at 21:51 UTC Monday–Friday
51 21 * * 1-5 /root/TFT_trading/TFT_limit_cleanup.sh

# === TFT_trade_close.sh ===
# Runs at 21:51 UTC Monday–Friday
51 21 * * 1-5 /root/TFT_trading/TFT_trade_close.sh

# === TFT_trading_infra.sh ===
# Runs at 22:02 UTC Tuesday–Friday
2 22 * * 2-5 /root/TFT_trading/TFT_trading_infra.sh

# Runs at 08:52 UTC Monday
52 8 * * 1 /root/TFT_trading/TFT_trading_infra.sh

