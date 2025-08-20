import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tpqoa
from datetime import datetime, timedelta, timezone
from cla import ModelTrainer
from sklearn.preprocessing import OneHotEncoder
import re
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from constants import sentiment_link, news_link
import requests
from PostGresConn import PostgresSQL


def get_news():
    db = PostgresSQL()
    data = db.FetchAllData("economicnews")
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    df.rename(columns={
        'currency': 'Currency',
        'time': 'Time',
        'news': 'News',
        'forecast': 'Forecast',
        'actual': 'Actual',
        'previous': 'Previous'
    }, inplace=True)
    return df

def get_market_sentiment(pair):
    def fix_date_time_sync(df):
        for i, row in df.iterrows():
            time_val = row['time']
            date_val = row['date']
            if (pd.isnull(time_val) or not isinstance(time_val, pd.Timestamp)) and isinstance(date_val, pd.Timestamp):
                df.at[i, 'time'] = date_val
            elif (pd.isnull(date_val) or not isinstance(date_val, pd.Timestamp)) and isinstance(time_val, pd.Timestamp):
                df.at[i, 'date'] = time_val   
        return df

    db = PostgresSQL()
    data = db.FetchAllData('sentimentdata')
    df = pd.DataFrame(data)
    df['time'] =  pd.to_datetime(df.time, errors='ignore')
    df['date'] = pd.to_datetime(df.date, errors='ignore')
    df = df[df.currency==pair]
    df = fix_date_time_sync(df)
    return df
def ReturnNewsDf():
    db = PostgresSQL()
    data = db.FetchAllData("economicnews")
    df = pd.DataFrame(data)
    df.set_index(0, inplace=True)
    df.rename(columns={1:'Currency', 2:'Time', 3:'News',4:'Forecast', 5:'Actual', 6:"Previous" }, inplace=True)
    return df



def ApplyScaling(df, pair):
    columns = ['long_ma_cro','short_ma_diff', 'short_ma_slope','long_ma_diff','long_ma_slope','ma_diff', 
               'EMA_50', 'EMA_50_slope', 'AD_EMA', 'AD_EMA_slope', 'RSI', 'ATR','fast_ma', 'short_ma', 
               '200_ma','Change_AD','Change_EMA','BB_MA','BB_UP','BB_LW', 'ATR_14','EMA','KeUp','KeLo',
               'MACD','HIST','ROC','+DI','-DI','ADX_14','VWMA_14','smoothed_rsi','dosc','derivative_oscillator',
               'Williams_R','TSI']
    
    scaler_path = f"general_scaler_{pair}.pkl"
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df[columns] = scaler.transform(df[columns])
    else:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        joblib.dump(scaler, scaler_path)
    
    return df
def clean_value(val):
    """
    Convert a string like '2.5%', '300K', '-0.07M' to a float.
    Returns np.nan if conversion is not possible.
    """
    if isinstance(val, str):
        val = val.strip().replace(',', '')  # remove commas, spaces

        # Handle percentage
        if val.endswith('%'):
            try:
                return float(val.strip('%')) / 100
            except ValueError:
                return np.nan
        
        # Handle suffixes
        suffixes = {
            'K': 1e3,
            'M': 1e6,
            'B': 1e9,
            'T': 1e12
        }

        # Check for suffix
        for suffix, multiplier in suffixes.items():
            if val.upper().endswith(suffix):
                try:
                    num = float(val[:-1])
                    return num * multiplier
                except ValueError:
                    return np.nan

        # Plain number
        try:
            return float(val)
        except ValueError:
            return np.nan
    
    elif isinstance(val, (int, float)):
        return float(val)

    return np.nan

def fetch_scaler(key, scaler_dir="news_scaler"):
    """Check if scaler exists; if yes, load it, else return None."""
    path = os.path.join(scaler_dir, f"{key}.pkl")
    if os.path.exists(path):
        scaler = joblib.load(path)
        return scaler
    return None

def scale_by_group(df, group_cols=['Currency', 'News'], value_cols=['Forecast', 'Actual', 'Previous'], save_scalers=True):
    """
    Applies StandardScaler per group (e.g., Currency + News).
    Uses cached scalers if they exist. Returns scaled DataFrame.
    """
    # Clean News field and values
    df['News'] = df['News'].str.replace(" ", "_")  # <-- FIXED
    for col in value_cols:
        df[col] = df[col].apply(clean_value)

    # ✅ FIXED: Proper row-wise group key
    df['group_key'] = df[group_cols].astype(str).agg('_'.join, axis=1)

    # Create scaler directory if needed
    if save_scalers:
        os.makedirs("news_scaler", exist_ok=True)

    scaled_df = df.copy()

    for key, group in df.groupby('group_key'):
        sub_df = group[value_cols]

        if sub_df.isnull().any().any():
            continue  # skip NaN rows

        scaler = fetch_scaler(key)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(sub_df)
            if save_scalers:
                joblib.dump(scaler, f"news_scaler/{key}.pkl")

        scaled_values = scaler.transform(sub_df)
        scaled_df.loc[group.index, value_cols] = scaled_values

    scaled_df.dropna(inplace=True)  # move this outside loop
    return scaled_df

def ApplyScaling(df, pair):
    columns = ['long_ma_cro','short_ma_diff', 'short_ma_slope','long_ma_diff','long_ma_slope','ma_diff', 
               'EMA_50', 'EMA_50_slope', 'AD_EMA', 'AD_EMA_slope', 'RSI', 'ATR','fast_ma', 'short_ma', 
               '200_ma','Change_AD','Change_EMA','BB_MA','BB_UP','BB_LW', 'ATR_14','EMA','KeUp','KeLo',
               'MACD','HIST','ROC','+DI','-DI','ADX_14','VWMA_14','smoothed_rsi','dosc','derivative_oscillator',
               'Williams_R','Conversion_Line','Base_Line','Leading_Span_A','Leading_Span_B','Lagging_Span','TSI']
    
    scaler_path = f"general_scaler_{pair}.pkl"
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df[columns] = scaler.transform(df[columns])
    else:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        joblib.dump(scaler, scaler_path)
    
    return df




def calculate_derivative_oscillator(df, rsi_length=14, ema1_length=5, ema2_length=3, sma_length=9, signal_length=9):

    # Step 2: Apply double EMA smoothing to RSI
    df['smoothed_rsi'] = df['RSI'].ewm(span=ema1_length, adjust=False).mean()
    df['smoothed_rsi'] = df['smoothed_rsi'].ewm(span=ema2_length, adjust=False).mean()
    
    # Step 3: Subtract SMA from smoothed RSI to get DOSC
    df['dosc'] = df['smoothed_rsi'] - df['smoothed_rsi'].rolling(window=sma_length).mean()
    
    # Step 4: Calculate the signal line
    df['derivative_oscillator'] = df['dosc'].rolling(window=signal_length).mean()
    
    return df


def df_initialization(pair, time_frame = 'D', more_data = False):

    ''' for pair pls enter something like EUR_USD, time_frame = D, H4, H1, M5, M15, M1'''
    if "_" in pair:
        pair_without_dash = pair.replace("_", "")
    else:
        pair_without_dash = pair
        
    if time_frame == "D":
        if  not more_data: 
            df_sen = pd.read_csv(f"{pair_without_dash}_fx.csv")
            df_sen = df_sen.dropna()
            df_sen['short_position_perce'] = df_sen["Short Positions"] * 100 / (df_sen["Short Positions"] + df_sen["Long Positions"])
            df_sen['long_position_perce'] = df_sen["Long Positions"] * 100 / (df_sen["Short Positions"] + df_sen["Long Positions"])
            df_sen = df_sen[['Date', 'short_position_perce']]
            df_sen["Date"] = pd.to_datetime(df_sen['Date'])
            df = pd.read_csv(f"{pair}_{time_frame}.csv")
            df = df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
            df['time'] = pd.to_datetime(df['time'])
            df['Date'] = df['time']
            df_sen['Date'] = pd.to_datetime(df_sen['Date']) + pd.Timedelta(hours=21)
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize() + pd.Timedelta(hours=21)
            df = pd.merge(df, df_sen, on='Date', how='inner')
            return df

    else:
        df = pd.read_csv(f"{pair}_{time_frame}.csv")
        df = df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
        df['time'] = pd.to_datetime(df['time'])
        df['Date'] = df['time']
        return df
    
    if more_data:
        df = pd.read_csv(f"{pair}_{time_frame}.csv")
        df = df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
        df['time'] = pd.to_datetime(df['time'])
        df['Date'] = df['time']
        return df



def add_ma_cross(df, shortlen=9, longlen=21, how_many_days =5, future_or_not = True):
    """
    Calculate short and long moving averages and identify crossover points.
    
    Parameters:
        df (pd.DataFrame): A DataFrame with a 'close' column for price data.
        shortlen (int): Length of the short moving average.
        longlen (int): Length of the long moving average.
        
    Returns:
        pd.DataFrame: Updated DataFrame with short MA, long MA, and crossover points.
    """
    # Calculate moving averages
    df['short_ma_cro'] = df['close'].rolling(window=shortlen).mean()
    df['long_ma_cro'] = df['close'].rolling(window=longlen).mean()
    df['short_ma_diff'] = df['short_ma_cro'].diff()
    df['short_ma_slope'] = df['short_ma_diff'].ewm(span=10, min_periods=10).mean()
    df['short_ma_slope'] *= 10000


    df['long_ma_diff'] = df['long_ma_cro'].diff()
    df['long_ma_slope'] = df['long_ma_diff'].ewm(span=10, min_periods=10).mean()
    df['long_ma_slope'] *= 10000

    df['ma_diff'] = df['short_ma_cro'] - df['long_ma_cro']
    df['price_above_short_ma'] = np.where(df['close'] > df['short_ma_cro'], 1, 0)
    df['price_above_long_ma'] =  np.where(df['close'] > df['long_ma_cro'], 1, 0)
    
    # Identify crossover points
    df['crossover_long_or_short'] = np.where(df['short_ma_cro'] > df['long_ma_cro'], 1, 0)
    if future_or_not:
        df['crossover_long_or_short_future'] = df['crossover_long_or_short'].shift(-how_many_days)
    else:
        pass
    return df





def add_accumulation_distribution(df):
    """Calculate and add accumulation/distribution to the DataFrame."""
    df['acc_dis'] = calculate_accumulation_distribution(df)
    scaler = MinMaxScaler()
    df['acc_dis'] = scaler.fit_transform(df[['acc_dis']])
    return df

def add_ema_slope(df, ema_lenght):
    """Calculate and add EMA and AD EMA slopes to the DataFrame."""
    df['EMA_50_slope'] = derivative_ema(df, ema_length=ema_lenght)
    df['AD_EMA_slope'] = derivative_ad(df)
    return df

def calculate_price_diff_and_gain_loss(df):
    """Calculate price differences and gain/loss."""
    df['price_diff'] = df['close'].diff(periods=5)
    df['gain_or_loss'] = df['price_diff'].apply(lambda x: 1 if x > 0 else -1)
    return df

def calculate_future_price_diff_and_gain_loss(df, how_many_days):
    """Calculate future price differences and gain/loss."""
    df['future_close'] = df['close'].shift(-how_many_days)
    df['price_diff_future'] = df['future_close'] - df['close']
    df['gain_or_loss_future'] = df['price_diff_future'].apply(lambda x: 1 if x > 0 else -1)
    return df

def calculate_rsi(df, window_length=14):
    """Calculate RSI for the DataFrame."""
    df['delta'] = df['close'].diff()
    df['gain'] = df['delta'].where(df['delta'] > 0, 0)
    df['loss'] = -df['delta'].where(df['delta'] < 0, 0)
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=1).mean()
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=1).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['rs']))
    return df

def calculate_atr(df, atr_window=14):
    """Calculate ATR using EMA."""
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=atr_window, adjust=False).mean()
    return df

def add_moving_averages(df, fastLength=50, shortLength=20):
    """Calculate moving averages and add long/short conditions."""
    df['fast_ma'] = df['close'].rolling(window=fastLength).mean()
    df['short_ma'] = df['close'].rolling(window=shortLength).mean()
    df['200_ma'] = df['close'].rolling(window=10).mean()
    df['c_long'] = df['close'] > df['200_ma']
    df['c_short'] = df['close'] < df['200_ma']
    df['long_condition'] = np.where((df['close'] > df['short_ma']) & df['c_long'], 1, 0)
    df['short_condition'] = np.where((df['close'] < df['short_ma']) & df['c_short'], 1, 0)
    return df

def change_in_slope(df, number_of_days_back =2):
    df['Change_AD'] = df['AD_EMA_slope'].shift(number_of_days_back)
    df['Change_EMA'] =  df['EMA_50_slope']. shift(number_of_days_back)
    return df


def is_close_above_ema(df):
    df['close_above_EMA'] = np.where((df['close'] > df['EMA_50']), 1, 0)
    return df

def calculate_rate_of_change(df, window=14):
    """
    Calculate the Rate of Change (ROC) indicator.
    """
    df['ROC'] = df['close'].pct_change(periods=window) * 100
    return df


def clean_dataframe(df):
    """Drop intermediate and unnecessary columns."""
    df = df.drop(columns=[
        'c_long', 'c_short',
        'high_low', 'high_close', 'low_close',
         'EMA_50_diff', 'AD_EMA_diff', 'price_diff',
        'loss', 'delta', 'gain', 'avg_gain', 'avg_loss', 'rs','body_lower', 'body_upper',
       'body_bottom_perc'
    ])

    return df

def calculate_accumulation_distribution(df):
    """
    Calculate the Accumulation/Distribution (A/D) indicator for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'close', 'high', 'low', and 'volume' columns.
        
    Returns:
        pd.Series: A series representing the A/D line.
    """
    if 'close' not in df or 'high' not in df or 'low' not in df or 'volume' not in df:
        raise ValueError("DataFrame must contain 'close', 'high', 'low', and 'volume' columns.")
    
    # Handle missing volume data
    if df['volume'].isnull().all():
        raise ValueError("No volume data provided in the DataFrame.")
    
    # Calculate the money flow multiplier
    mfm = np.where(
        (df['high'] == df['low']), 
        0, 
        ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']))
    )
    
    # Calculate money flow volume
    mfv = mfm * df['volume']
    
    # Accumulate the money flow volume to get the A/D line
    ad_line = mfv.cumsum()
    
    return ad_line

def derivative_ad(df):
    # Ensure the A/D line is calculated
    if 'acc_dis' not in df:
        df['acc_dis'] = calculate_accumulation_distribution(df)

    # Calculate EMA of the A/D line
    df['AD_EMA'] = df['acc_dis'].ewm(span=50, min_periods=50).mean()
    
    # Calculate the difference (first derivative approximation)
    df['AD_EMA_diff'] = df['AD_EMA'].diff()
    
    # Smooth the derivative using another EMA
    df['AD_EMA_slope'] = df['AD_EMA_diff'].ewm(span=10, min_periods=10).mean()
    
    # Scale the slope for readability (optional)
    df['AD_EMA_slope'] *= 10000
    
    return df['AD_EMA_slope']


def derivative_ema(df, ema_length=50):
  
    df[f'EMA_50'] = df['close'].ewm(span=ema_length, min_periods=ema_length).mean()
    
    # Calculate the difference (first derivative approximation)
    df['EMA_50_diff'] = df['EMA_50'].diff()
    
    # Smooth the derivative using another EMA with a span of 10
    df['EMA_50_slope'] = df['EMA_50_diff'].ewm(span=10, min_periods=10).mean()
    
    # Scale the slope for readability (optional)
    df['EMA_50_slope'] *= 10000
    
    return df['EMA_50_slope']

def derivative_ema_in_future(df, how_many_days):
    """Calculate future price differences and gain/loss."""
    df['future_EMA_50_slope'] = df['EMA_50_slope'].shift(-how_many_days)
    df['slope_diff'] = df['future_EMA_50_slope'] - df['EMA_50_slope']
    df['slope_gain_or_loss_future'] = df['slope_diff'].apply(lambda x: 1 if x > 0 else -1)
    return df
def ATR(df: pd.DataFrame, n=14):
    prev_c = df.close.shift(1)
    tr1 = df.high - df.low
    tr2 = abs(df.high - prev_c)
    tr3 = abs(prev_c - df.low)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df[f"ATR_{n}"] = tr.rolling(window=n).mean()
    return df

def KeltnerChannels(df: pd.DataFrame, n_ema=20, n_atr=10):
    df['EMA'] = df.close.ewm(span=n_ema, min_periods=n_ema).mean()
    df = ATR(df, n=n_atr)
    c_atr = f"ATR_{n_atr}"
    df['KeUp'] = df[c_atr] * 2 + df['EMA_50']
    df['KeLo'] = df['EMA_50'] - df[c_atr] * 2 
    df.drop(c_atr, axis=1, inplace=True)
    return df


def MACD(df: pd.DataFrame, n_slow=26, n_fast=12, n_signal=9):

    ema_long = df.close.ewm(min_periods=n_slow, span=n_slow).mean()
    ema_short = df.close.ewm(min_periods=n_fast, span=n_fast).mean()

    df['MACD'] = ema_short - ema_long
    df['SIGNAL_MACD'] = df.MACD.ewm(min_periods=n_signal, span=n_signal).mean()
    df['HIST'] = df.MACD - df.SIGNAL_MACD

    return df


def BollingerBands(df: pd.DataFrame, n=20, s=2):
    typical_p = ( df.close + df.high + df.low ) / 3
    stddev = typical_p.rolling(window=n).std()
    df['BB_MA'] = typical_p.rolling(window=n).mean()
    df['BB_UP'] = df['BB_MA'] + stddev * s
    df['BB_LW'] = df['BB_MA'] - stddev * s
    return df

def BB_encoder(df, how_many_days):
    df['BB_binary'] = np.where((df['close'] > df['BB_MA']), 1, 0)
    df['BB_binary_future'] = df['BB_binary'].shift(-how_many_days)
    return df

def BB_encoder_for_practice(df):
    df['BB_binary'] = np.where((df['close'] > df['BB_MA']), 1, 0)
    return df
def ADX(df: pd.DataFrame, n=14):
    df['TR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['high'], min)
    df['+DM'] = df['high'].diff().clip(lower=0)
    df['-DM'] = -df['low'].diff().clip(upper=0)

    # Rolling averages for TR, +DM, -DM
    tr_rolling = df['TR'].rolling(window=n).sum()
    plus_dm_rolling = df['+DM'].rolling(window=n).sum()
    minus_dm_rolling = df['-DM'].rolling(window=n).sum()

    # Calculate smoothed +DI, -DI
    df['+DI'] = 100 * (plus_dm_rolling / tr_rolling)
    df['-DI'] = 100 * (minus_dm_rolling / tr_rolling)

    # Calculate DX and ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df[f"ADX_{n}"] = df['DX'].rolling(window=n).mean()

    # Clean up intermediate columns
    df.drop(['TR', '+DM', '-DM', 'DX'], axis=1, inplace=True)
    return df


def VWMA(df: pd.DataFrame,how_many_days = 5, n=14, future = True):
    # Calculate VWMA
    df[f"VWMA_{n}"] = (df['close'] * df['volume']).rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    if future:
        df['VWMA_future'] = df[f"VWMA_{n}"].shift(-how_many_days)
        df['VWMA_future_encoded'] = np.where(df.VWMA_future > df[f"VWMA_{n}"], 1, 0)
    else:
        pass
    return df






import pandas as pd

HANGING_MAN_BODY = 15.0
HANGING_MAN_HEIGHT = 75.0
SHOOTING_STAR_HEIGHT = 25.0
SPINNING_TOP_MIN = 40.0
SPINNING_TOP_MAX = 60.0
MARUBOZU = 98.0
ENGULFING_FACTOR = 1.1

MORNING_STAR_PREV2_BODY = 90.0
MORNING_STAR_PREV_BODY = 10.0

TWEEZER_BODY = 15.0
TWEEZER_HL = 0.01
TWEEZER_TOP_BODY = 40.0
TWEEZER_BOTTOM_BODY = 60.0
THREE_CROWS_SOLDIERS_BODY = 50.0 

apply_marubozu = lambda x: x.body_perc > MARUBOZU

def apply_hanging_man(row):
    if row.body_bottom_perc > HANGING_MAN_HEIGHT:
        if row.body_perc < HANGING_MAN_BODY:
            return True
    return False

def apply_shooting_star(row):
    if row.body_top_perc < SHOOTING_STAR_HEIGHT:
        if row.body_perc < HANGING_MAN_BODY:
            return True
    return False

def apply_spinning_top(row):
    if row.body_top_perc < SPINNING_TOP_MAX:
        if row.body_bottom_perc > SPINNING_TOP_MIN:
            if row.body_perc < HANGING_MAN_BODY:
                return True
    return False

def apply_engulfing(row):
    if row.direction != row.direction_prev:
        if row.body_size > row.body_size_prev * ENGULFING_FACTOR:
            return True
    return False

def apply_tweezer_top(row):
    if abs(row.body_size_change) < TWEEZER_BODY:
        if row.direction == -1 and row.direction != row.direction_prev:
            if abs(row.low_change) < TWEEZER_HL and abs(row.high_change) < TWEEZER_HL:
                if row.body_top_perc < TWEEZER_TOP_BODY:
                    return True
    return False               

def apply_tweezer_bottom(row):
    if abs(row.body_size_change) < TWEEZER_BODY:
        if row.direction == 1 and row.direction != row.direction_prev:
            if abs(row.low_change) < TWEEZER_HL and abs(row.high_change) < TWEEZER_HL:
                if row.body_bottom_perc > TWEEZER_BOTTOM_BODY:
                    return True
    return False     


def apply_morning_star(row, direction=1):
    if row.body_perc_prev_2 > MORNING_STAR_PREV2_BODY:
        if row.body_perc_prev < MORNING_STAR_PREV_BODY:
            if row.direction == direction and row.direction_prev_2 != direction:
                if direction == 1:
                    if row.close > row.mid_point_prev_2:
                        return True
                else:
                    if row.close < row.mid_point_prev_2:
                        return True
    return False


def apply_three_crows(row):
    # Check if all three candles are bearish
    if row.direction_prev_2 == -1 and row.direction_prev == -1 and row.direction == -1:
        # Check if each candle body is at least 50% of the full range (H to L)
        if row.body_perc_prev_2 > THREE_CROWS_SOLDIERS_BODY:
            if row.body_perc_prev > THREE_CROWS_SOLDIERS_BODY:
                if row.body_perc > THREE_CROWS_SOLDIERS_BODY:
                    return True
    return False

def apply_three_soldiers(row):
    # Check if all three candles are bullish
    if row.direction_prev_2 == 1 and row.direction_prev == 1 and row.direction == 1:
        # Check if each candle body is at least 50% of the full range (H to L)
        if row.body_perc_prev_2 > THREE_CROWS_SOLDIERS_BODY:
            if row.body_perc_prev > THREE_CROWS_SOLDIERS_BODY:
                if row.body_perc > THREE_CROWS_SOLDIERS_BODY:
                    return True
    return False



def apply_candle_props(df: pd.DataFrame):
    df_an = df.copy()
    direction = df_an.close - df_an.open
    body_size = abs(direction)
    direction = [1 if x >= 0 else -1 for x in direction]
    full_range = df_an.high - df_an.low
    body_perc = (body_size / full_range) * 100
    body_lower = df_an[['close', 'open']].min(axis=1)
    body_upper = df_an[['close', 'open']].max(axis=1)
    body_bottom_perc = ((body_lower - df_an.low) / full_range) * 100
    body_top_perc = 100 - (((df_an.high - body_upper) / full_range) * 100)

    mid_point = full_range / 2 + df_an.low

    low_change = df_an.low.pct_change() * 100
    high_change = df_an.high.pct_change() * 100
    body_size_change = body_size.pct_change() * 100

    df_an['body_lower'] = body_lower
    df_an['body_upper'] = body_upper
    df_an['body_bottom_perc'] = body_bottom_perc
    df_an['body_top_perc'] = body_top_perc
    df_an['body_perc'] = body_perc
    df_an['direction'] = direction
    df_an['body_size'] = body_size
    df_an['low_change'] = low_change
    df_an['high_change'] = high_change
    df_an['body_size_change'] = body_size_change
    df_an['mid_point'] = mid_point
    df_an['mid_point_prev_2'] = mid_point.shift(2)
    df_an['body_size_prev'] = df_an.body_size.shift(1)
    df_an['direction_prev'] = df_an.direction.shift(1)
    df_an['direction_prev_2'] = df_an.direction.shift(2)
    df_an['body_perc_prev'] = df_an.body_perc.shift(1)
    df_an['body_perc_prev_2'] = df_an.body_perc.shift(2)

    return df_an


def set_candle_patterns(df_an: pd.DataFrame):
    df_an['HANGING_MAN'] = df_an.apply(apply_hanging_man, axis=1)
    df_an['SHOOTING_STAR'] = df_an.apply(apply_shooting_star, axis=1)
    df_an['SPINNING_TOP'] = df_an.apply(apply_spinning_top, axis=1)
    df_an['MARUBOZU'] = df_an.apply(apply_marubozu, axis=1)
    df_an['ENGULFING'] = df_an.apply(apply_engulfing, axis=1)
    df_an['TWEEZER_TOP'] = df_an.apply(apply_tweezer_top, axis=1)
    df_an['TWEEZER_BOTTOM'] = df_an.apply(apply_tweezer_bottom, axis=1)
    df_an['MORNING_STAR'] = df_an.apply(apply_morning_star, axis=1)
    df_an['EVENING_STAR'] = df_an.apply(apply_morning_star, axis=1, direction=-1)
    df_an['THREE_BLACK_CROWS'] = df_an.apply(apply_three_crows, axis=1)
    df_an['THREE_WHITE_SOLDIERS'] = df_an.apply(apply_three_soldiers, axis=1)

def apply_patterns(df: pd.DataFrame):
    df_an = apply_candle_props(df)
    set_candle_patterns(df_an)
    patterns = [
    'HANGING_MAN', 'SHOOTING_STAR', 'SPINNING_TOP', 'MARUBOZU', 
    'ENGULFING', 'TWEEZER_TOP', 'TWEEZER_BOTTOM', 'MORNING_STAR', 
    'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS'
    ]

    for pattern in patterns:
        df_an[pattern] = df_an[pattern].astype(int)
    list = []
    list.sort()
    return df_an


def WilliamsR(df: pd.DataFrame, length=14):
    # Calculate rolling high and low
    high_rolling = df['high'].rolling(window=length).max()
    low_rolling = df['low'].rolling(window=length).min()
    
    # Calculate Williams %R
    df['Williams_R'] = (df['close'] - high_rolling) / (high_rolling - low_rolling) * -100
    df['Williams_encoded'] = np.where(df['Williams_R']> 0.5, 1, 0)
    return df
def Ichimoku(df: pd.DataFrame, conversionPeriods=9, basePeriods=26, laggingSpan2Periods=52, displacement=26):
    # Conversion Line
    df['Conversion_Line'] = (df['high'].rolling(window=conversionPeriods).max() +
                             df['low'].rolling(window=conversionPeriods).min()) / 2
    
    # Base Line
    df['Base_Line'] = (df['high'].rolling(window=basePeriods).max() +
                       df['low'].rolling(window=basePeriods).min()) / 2
    
    # Leading Span A
    df['Leading_Span_A'] = ((df['Conversion_Line'] + df['Base_Line']) / 2).shift(displacement)
    
    # Leading Span B
    df['Leading_Span_B'] = ((df['high'].rolling(window=laggingSpan2Periods).max() +
                             df['low'].rolling(window=laggingSpan2Periods).min()) / 2).shift(displacement)
    
    # Lagging Span
    df['Lagging_Span'] = df['close'].shift(-displacement)
    
    return df
def DMI(df: pd.DataFrame, n=14):
    df['TR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['high'], min)
    df['+DM'] = df['high'].diff().clip(lower=0)
    df['-DM'] = -df['low'].diff().clip(upper=0)

    # Rolling averages for TR, +DM, -DM
    tr_rolling = df['TR'].rolling(window=n).sum()
    plus_dm_rolling = df['+DM'].rolling(window=n).sum()
    minus_dm_rolling = df['-DM'].rolling(window=n).sum()

    # Calculate smoothed +DI, -DI
    df['+DI'] = 100 * (plus_dm_rolling / tr_rolling)
    df['-DI'] = 100 * (minus_dm_rolling / tr_rolling)

    # Calculate DX and ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df[f"ADX_{n}"] = df['DX'].rolling(window=n).mean()

    # Clean up intermediate columns
    df.drop(['TR', '+DM', '-DM', 'DX'], axis=1, inplace=True)
    return df
def TSI(df: pd.DataFrame, r=25, s=13):
    """
    True Strength Index (TSI) implementation.
    r: the first smoothing period
    s: the second smoothing period
    """
    # Calculate momentum
    momentum = df['close'].diff()
    abs_momentum = abs(momentum)

    # Smooth momentum and absolute momentum
    double_smoothed_momentum = momentum.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    double_smoothed_abs_momentum = abs_momentum.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()

    # Calculate TSI
    df['TSI'] = (double_smoothed_momentum / double_smoothed_abs_momentum) * 100

    # Binary classification based on a threshold
    df['TSI_Binary'] = (df['TSI'] > 0).astype(int)  # Example: Positive TSI is bullish
    return df


def supertrend(df, atr_period=10, atr_multiplier=3.0, change_atr_method=True):
    """
    Calculate the Supertrend indicator.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and optionally 'hl2' columns.
        atr_period (int): ATR calculation period.
        atr_multiplier (float): Multiplier for the ATR to calculate the bands.
        change_atr_method (bool): Use a different ATR calculation method if True.

    Returns:
        pd.DataFrame: DataFrame with added Supertrend columns: 'supertrend', 'trend', 'buy_signal', 'sell_signal'.
    """
    # Ensure 'hl2' column exists
    if 'hl2' not in df.columns:
        df['hl2'] = (df['high'] + df['low']) / 2

    # ATR calculation
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(prev_close - df['low'])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    if change_atr_method:
        df['atr'] = tr.rolling(window=atr_period).mean()
    else:
        df['atr'] = tr.expanding().mean()

    # Supertrend calculation
    df['basic_upperband'] = df['hl2'] - (atr_multiplier * df['atr'])
    df['basic_lowerband'] = df['hl2'] + (atr_multiplier * df['atr'])

    df['upperband'] = df['basic_upperband']
    df['lowerband'] = df['basic_lowerband']

    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > df['upperband'].iloc[i - 1]:
            df['upperband'].iloc[i] = max(df['upperband'].iloc[i], df['upperband'].iloc[i - 1])
        if df['close'].iloc[i - 1] < df['lowerband'].iloc[i - 1]:
            df['lowerband'].iloc[i] = min(df['lowerband'].iloc[i], df['lowerband'].iloc[i - 1])

    df['trend'] = 1
    for i in range(1, len(df)):
        if df['trend'].iloc[i - 1] == 1 and df['close'].iloc[i] < df['upperband'].iloc[i - 1]:
            df['trend'].iloc[i] = -1
        elif df['trend'].iloc[i - 1] == -1 and df['close'].iloc[i] > df['lowerband'].iloc[i - 1]:
            df['trend'].iloc[i] = 1
        else:
            df['trend'].iloc[i] = df['trend'].iloc[i - 1]

    df['supertrend'] = np.where(df['trend'] == 1, df['upperband'], df['lowerband'])

    # Signals
    df['buy_signal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
    df['sell_signal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)

    return df

def add_ema_slope_class(df, ema_lenght):
    """Calculate and add EMA and AD EMA slopes to the DataFrame."""
    df['EMA_50_slope'] = derivative_ema(df, ema_length=ema_lenght)
    df['ema_encoded_trend'] = np.where(df.EMA_50_slope > 0, 1, 0)
    return df

def derivative_ema_class(df, ema_length=9):
  
    df[f'EMA_50'] = df['close'].ewm(span=ema_length, min_periods=ema_length).mean()

    df['EMA_50_diff'] = df['EMA_50'].diff()
    df['EMA_50_slope'] = df['EMA_50_diff'].ewm(span=10, min_periods=10).mean()
    df['EMA_50_slope'] *= 10000

    
    return df['EMA_50_slope']



def df_builder(pair,how_many_days = 5, time_frame="D", more_data = False, number_of_days_back = 3, ema_length = 50, shortlen = 20, longlen = 30):
    df = df_initialization(pair, time_frame, more_data)
    df = add_accumulation_distribution(df)
    df = add_ma_cross(df, shortlen=shortlen, longlen = longlen, how_many_days=how_many_days)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df= derivative_ema_in_future(df, how_many_days)
    df= BollingerBands(df)
    df = BB_encoder(df, how_many_days = how_many_days)
    df = ATR(df)
    df= KeltnerChannels(df) 
    df = MACD(df)
    df= calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, how_many_days=how_many_days)
    df = supertrend(df)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df['ATR_future'] = df['ATR_14'].shift(-5)
    df['ADX_future'] = df['ADX_14'].shift(-5)
    df['dist_MA'] = df['close'] - df['BB_MA']
    df['z_score'] = (df['close'] - df['BB_MA']) / df['close'].rolling(window=20).std()
    df['volatility'] = df['close'].rolling(window=20).std()
    df['day_close_diff_day_open'] = df['close']-df['open']
    df['RSI_future'] = df['RSI'].shift(-how_many_days)
    df['RSI_future_encoded'] = np.where(df.RSI_future > df.RSI, 1, 0)
    df = apply_patterns(df)
    df = clean_dataframe(df)
    return df

def df_for_prediction(df, number_of_days_back = 3, ema_length = 30, shortlen = 20, longlen = 30, how_many_days = 5, future = True):
    try:
        df = df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
    finally:
        pass
    df = add_accumulation_distribution(df)
    df = add_ma_cross(df, shortlen=shortlen, longlen = longlen, future_or_not=future)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df= derivative_ema_in_future(df, how_many_days)
    df= BollingerBands(df)
    df = BB_encoder(df, how_many_days = how_many_days)
    df = ATR(df)
    df= KeltnerChannels(df) 
    df = MACD(df)
    df= calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, future=future)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df['ATR_future'] = df['ATR_14'].shift(-5)
    df['ADX_future'] = df['ADX_14'].shift(-5)
    df['dist_MA'] = df['close'] - df['BB_MA']
    df['z_score'] = (df['close'] - df['BB_MA']) / df['close'].rolling(window=20).std()
    df['volatility'] = df['close'].rolling(window=20).std()
    df['day_close_diff_day_open'] = df['close']-df['open']
    df['RSI_future'] = df['RSI'].shift(-how_many_days)
    df['RSI_future_encoded'] = np.where(df.RSI_future > df.RSI, 1, 0)
    df = apply_patterns(df)
    df = clean_dataframe(df)
    return df


def get_last_100_candles(pair_w_dash):
    api = tpqoa.tpqoa("oanda.cfg")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=200)
    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")

    granularity = "D"
    df = api.get_history(instrument=pair_w_dash, start=start_time_str, end=end_time_str,
                         granularity=granularity, price="B")
    
    df = df.reset_index()

    return df[-100:]



def get_last_1000_candles(pair_w_dash):
    api = tpqoa.tpqoa("oanda.cfg")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=4500)
    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")

    granularity = "D"
    df = api.get_history(instrument=pair_w_dash, start=start_time_str, end=end_time_str,
                         granularity=granularity, price="B")
    
    df = df.reset_index()

    return df




def setup_df(pair):
    currency1, currency2 = pair.split('_')  
    df_pred = get_last_100_candles(pair)
    price_df = df_pred
    news_df = pd.read_csv("news_data_fun.csv")
    # Convert time columns to datetime format
    price_df['Time'] = price_df['time']
    price_df['Time'] = pd.to_datetime(price_df['time'])
    news_df['Time'] = pd.to_datetime(news_df['Time'], format='%a %b %d, %Y, %I:%M %p')
 
    filtered_news_df = news_df[news_df['Currency'].isin([currency1, currency2])]

    # Identify news containing "Interest"
    filtered_news_df['is_interest'] = filtered_news_df['News'].str.contains("Interest", case=False, na=False)

    # Remove duplicates but keep interest-related news
    non_interest_news_df = filtered_news_df[~filtered_news_df['is_interest']]
    interest_news_df = filtered_news_df[filtered_news_df['is_interest']]

    non_interest_news_df = non_interest_news_df.sort_values('Time').drop_duplicates(
        subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous']
    )

    # Combine back interest-related news to ensure they are not removed
    filtered_news_df = pd.concat([non_interest_news_df, interest_news_df]).sort_values('Time')

    # Merge news with price data
    merged_df = pd.merge_asof(
        price_df.sort_values('Time'),
        filtered_news_df.drop(columns=['is_interest']),  # Remove helper column
        on='Time',
        direction='backward',
        allow_exact_matches=False  
    )

    no_news_mask = merged_df[['Currency', 'News', 'Forecast', 'Actual', 'Previous']].isna().all(axis=1)
    merged_df.loc[no_news_mask, ['Currency', 'News', 'Forecast', 'Actual', 'Previous']] = 'No News'

    # Remove duplicates from the merged DataFrame
    merged_df = merged_df.drop_duplicates(subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous'])
    
    
    merged_df = pd.merge(price_df, 
                        merged_df, 
                        on='Time',
                        how= 'outer')

 

    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_y')])
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_x')])
    merged_df = merged_df.fillna(0)
    merged_df['Currency'] = np.where(merged_df['Currency']==currency1, 2,
                                     np.where(merged_df['Currency'] == currency2,1,0 ))
    merged_df['News'] = merged_df['News'].astype(str)
    encoder = OneHotEncoder(sparse_output=False) 
    encoded = encoder.fit_transform(merged_df[['News']])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['News']))

    merged_df = pd.concat([merged_df, encoded_df], axis=1)

    merged_df['Forecast'] = merged_df['Forecast'].apply(remove_punctuations)
    merged_df['Actual'] = merged_df['Actual'].apply(remove_punctuations)
    merged_df['Previous'] = merged_df['Previous'].apply(remove_punctuations)

    df_pred = merged_df

    number_of_days_back = 3 
    ema_length = 30 
    shortlen = 20 
    longlen = 30 
    how_many_days = 5
    future = True
    try:
        df_pred = df_pred.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
    finally:
        pass
    df = add_accumulation_distribution(df_pred)
    df = add_ma_cross(df, shortlen=shortlen, longlen = longlen, future_or_not=future)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df= derivative_ema_in_future(df, how_many_days)
    df= BollingerBands(df)
    df = BB_encoder(df, how_many_days = how_many_days)
    df = ATR(df)
    df= KeltnerChannels(df) 
    df = MACD(df)
    df= calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, future=future)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df['ATR_future'] = df['ATR_14'].shift(-5)
    df['ADX_future'] = df['ADX_14'].shift(-5)
    df['dist_MA'] = df['close'] - df['BB_MA']
    df['z_score'] = (df['close'] - df['BB_MA']) / df['close'].rolling(window=20).std()
    df['volatility'] = df['close'].rolling(window=20).std()
    df['day_close_diff_day_open'] = df['close']-df['open']
    df['RSI_future'] = df['RSI'].shift(-how_many_days)
    df['RSI_future_encoded'] = np.where(df.RSI_future > df.RSI, 1, 0)
    df_pred = apply_patterns(df)


    return df_pred


def df_for_training_classification(pair, timeframe):
    df = pd.read_csv(f'{pair}_{timeframe}.csv')
    df = df_for_prediction(df, number_of_days_back=3, ema_length=30, shortlen=20, longlen=30)
    df = df.reset_index(drop=True)
    return df


def classifier_ma(df, df_pred):
    target = 'crossover_long_or_short_future'
    features = ['TSI_Binary', 'Williams_encoded', 'price_above_short_ma','price_above_long_ma', 'short_ma_slope', 'long_ma_slope', 'ma_diff', 'Change_EMA', 'long_ma_diff', 'EMA_50_slope', 'ATR', 'short_condition', 'ROC', 'crossover_long_or_short', 'BB_binary']
    
    trainer_classification = ModelTrainer(df, features, target=target)
    ac, y_pred, y_pred_proba = trainer_classification.best_classifier_random_forest()
    alw = trainer_classification.get_test_data_with_predictions()
    print(trainer_classification.accuracy)
    df_combined = pd.merge(df, alw, on="time", how='inner')
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_comb_2 = df_combined.filter(['time', 'close', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
    
    if "close" not in df_comb_2.columns:
        df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
        df_comb_2['close'] = df_comb_2['close_x']
    df_1 = df_comb_2

    scaled_features = trainer_classification.scaler.transform(df_pred[features])
    df_pred['Predicted_y'] = trainer_classification.classifier.predict(scaled_features)
    proba = trainer_classification.classifier.predict_proba(scaled_features)
    df_pred['Confidence_y'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))

    return df_1, df_pred




def classifier_BB(df, df_pred):
    features = [ 'price_above_short_ma', 'price_above_long_ma', 'short_condition', 'crossover_long_or_short', 'close_above_EMA' ,'BB_binary',  'gain_or_loss', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS' ] 

    target = 'BB_binary_future' 
    trainer_classification = ModelTrainer(df, features, target=target)
    ac, y_pred, y_pred_proba = trainer_classification.best_classifier_random_forest()
    alw = trainer_classification.get_test_data_with_predictions()
    print(trainer_classification.accuracy)
    df_combined = pd.merge(df, alw, on="time", how='inner')
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_comb_2 = df_combined.filter(['time', 'close', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
    
    if "close" not in df_comb_2.columns:
        df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
        df_comb_2['close'] = df_comb_2['close_x']
    df_1 = df_comb_2

    scaled_features = trainer_classification.scaler.transform(df_pred[features])
    df_pred['Predicted_x'] = trainer_classification.classifier.predict(scaled_features)
    proba = trainer_classification.classifier.predict_proba(scaled_features)
    df_pred['Confidence_x'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))
    return df_1, df_pred




def classifier_RSI(df, df_pred):
    features = [
    'TSI_Binary', 
    'HANGING_MAN', 
    'TWEEZER_TOP', 
    'long_condition', 
    'price_above_short_ma', 
    'MARUBOZU', 
    'direction_prev', 
    'THREE_WHITE_SOLDIERS', 
    'THREE_BLACK_CROWS', 
    'BB_binary', 
    'short_condition', 
    'crossover_long_or_short', 
    'z_score', 
    'RSI', 
    'MACD', 

    ]


    target = 'RSI_future_encoded'
    trainer_classification = ModelTrainer(df, features, target=target)
    ac, y_pred, y_pred_proba = trainer_classification.best_classifier_random_forest()
    alw = trainer_classification.get_test_data_with_predictions()
    print(trainer_classification.accuracy)
    df_combined = pd.merge(df, alw, on="time", how='inner')
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_comb_2 = df_combined.filter(['time', 'close', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence', 'open', 'high','volume','low'])

    
    if "close" not in df_comb_2.columns:
        df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
        df_comb_2['close'] = df_comb_2['close_x']
    df_1 = df_comb_2

    scaled_features = trainer_classification.scaler.transform(df_pred[features])
    df_pred['Predicted_z'] = trainer_classification.classifier.predict(scaled_features)
    df_pred['Confidence_z'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))
    return df_1, df_pred







def classifier_VWMA(df, df_pred):
    features = [
    'price_above_long_ma',
    'MARUBOZU',
    'THREE_WHITE_SOLDIERS',
    'price_above_short_ma',
    'crossover_long_or_short',
    'long_condition',
    'ENGULFING',
    'BB_binary',
    'close_above_EMA',
    'SHOOTING_STAR',
    'short_condition',
    'TWEEZER_TOP',
    'gain_or_loss',
    'HANGING_MAN']
    target = 'VWMA_future_encoded'
    trainer_classification = ModelTrainer(df, features, target=target)
    ac, y_pred, y_pred_proba = trainer_classification.best_classifier_random_forest()
    alw = trainer_classification.get_test_data_with_predictions()
    print(trainer_classification.accuracy)
    df_combined = pd.merge(df, alw, on="time", how='inner')
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_comb_2 = df_combined.filter(['time', 'close', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
    
    if "close" not in df_comb_2.columns:
        df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence'])
        df_comb_2['close'] = df_comb_2['close_x']
    df_1 = df_comb_2

    scaled_features = trainer_classification.scaler.transform(df_pred[features])
    df_pred['Predicted_a'] = trainer_classification.classifier.predict(scaled_features)
    df_pred['Confidence_a'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))
    return df_1, df_pred

    
def df_merger(pair, df_1, df_2, df_3, df_4, for_prediction = False):
    for column in df_1.columns:
        if column == 'time':
            continue
        df_1[f"{column}_x"] = df_1[column]
        
    for column in df_2.columns:
        if column == 'time':
            continue
        df_2[f"{column}_y"] = df_2[column]
        
    for column in df_3.columns:
        if column == 'time':
            continue
        df_3[f"{column}_z"] = df_3[column]
        
    for column in df_4.columns:
        if column == 'time':
            continue
        df_4[f"{column}_a"] = df_4[column]

# Remove old columns but retain 'time'
    df_1 = df_1[[col for col in df_1.columns if col.endswith('_x') or col == 'time']]
    df_2 = df_2[[col for col in df_2.columns if col.endswith('_y') or col == 'time']]
    df_3 = df_3[[col for col in df_3.columns if col.endswith('_z') or col == 'time']]
    df_4 = df_4[[col for col in df_4.columns if col.endswith('_a') or col == 'time']]
    merged_df = df_1.merge(df_2, on='time', how='outer')  # Merge df_1 and df_2 on 'time'
    merged_df = merged_df.merge(df_3, on='time', how='outer')  # Merge with df_3
    merged_df = merged_df.merge(df_4, on='time', how='outer')


    how_many_days = 5
    merged_df = merged_df.filter(['open_z', 'high_z', 'volume_z', 'low_z','time', 'close_x', 'Predicted_x', 'Predicted_y',"Predicted_z", "Predicted_a", 'Confidence_x', 'Confidence_y', 'Confidence_z', 'Confidence_a'])
    if not for_prediction:
        merged_df['future_close'] = merged_df['close_x'].shift(-how_many_days)
    merged_df['close'] = merged_df['close_x']
    merged_df['open'] = merged_df['open_z']
    merged_df['low'] = merged_df['low_z']
    merged_df['volume'] = merged_df['volume_z']
    merged_df['high'] = merged_df['high_z']
    merged_df =merged_df.dropna()
    merged_df = merged_df.drop(columns=['open_z', 'high_z', 'volume_z', 'low_z'])
    merged_df['pair'] = pair


    return merged_df
def merger_with_market_direction(df1, pair_without_dash):
    df_direction = pd.read_csv(f"{pair_without_dash}_fx.csv")
    df_direction['market_direction'] = np.where(
    df_direction['Short Positions'] > df_direction['Long Positions'] , 1,0
    
    )
    df2 = df_direction
    
    df1['time'] = pd.to_datetime(df1['time'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    df1 = df1.sort_values('time')
    df2 = df2.sort_values('Date')
    
    merged_df = pd.merge_asof(
        df1, 
        df2[['Date', 'market_direction']], 
        left_on='time', 
        right_on='Date',
        direction='backward'  
    )
    
    merged_df = merged_df.drop(columns=['Date'])
    df = merged_df.dropna()
    df =df.reset_index()
    df['result'] = np.where(
        (df['Predicted_x'] == df['Predicted_y']) & 
        (df['Predicted_y'] == df['Predicted_z']) & 
        (df['Predicted_z'] == df['Predicted_a']) &  
        (df['trend'] == 1)&
        (df['ema_encoded_trend'] == 1) & 
        (df['long_condition'] == 1)&
    
        
        (df['market_direction'] == 1), 
        1, 
        np.where(
            (df['Predicted_x'] == df['Predicted_y']) & 
            (df['Predicted_y'] == df['Predicted_z']) & 
            (df['Predicted_z'] == df['Predicted_a']) & 
            (df['trend'] == -1)&
             (df['ema_encoded_trend'] == 0) & 
            (df['short_condition'] == 1)&
            (df['market_direction'] == 0), 
            -1, 
            0
        )
    )
    return df  


def merger_2_with_direction(df1):
    df_direction = pd.read_csv(f"{pair_without_dash}_fx.csv")
    df_direction['market_direction'] = np.where(
    df_direction['Short Positions'] > df_direction['Long Positions'] , 1,0
    
    )
    df2 = df_direction
    
    df1['time'] = pd.to_datetime(df1['time'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    df1 = df1.sort_values('time')
    df2 = df2.sort_values('Date')
    
    merged_df = pd.merge_asof(
        df1, 
        df2[['Date', 'market_direction']], 
        left_on='time', 
        right_on='Date',
        direction='backward'  
    )
    
    merged_df = merged_df.drop(columns=['Date'])
    df = merged_df.dropna()
    df =df.reset_index()

    return df
def merger_for_pred(merger_df_pred_c):
    df_direction = pd.read_csv(f"{pair_without_dash}_fx.csv")
    df_direction['market_direction'] = np.where(
    df_direction['Short Positions'] > df_direction['Long Positions'] , 1,0
    
    )
    df2 = df_direction
    df2 = df2.dropna()
    merged_df_pred_c['time'] = pd.to_datetime(merged_df_pred_c['time'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    merged_df_pred_c['date_only'] = merged_df_pred_c['time'].dt.date
    merged_df_pred_c = merged_df_pred_c.sort_values('time')
    df2 = df2.sort_values('Date')
    merged_df_z = pd.merge_asof(
        merged_df_pred_c,
        df2[['Date', 'market_direction']],
        left_on='time',
        right_on='Date',
        direction='backward'
    )
    merged_df_z = merged_df_z.drop(columns=['date_only'])
    merged_df_z = merged_df_z[
        ['time', 'close', 'volume', 'Predicted_x', 'Predicted_y', 'Predicted_z', 'Predicted_a',
         'Confidence_x', 'Confidence_y', 'Confidence_z', 'Confidence_a', 'atr', 'trend',
         'ema_encoded_trend', '200_ma', 'long_condition', 'short_condition', 'market_direction']
    ]
    return merged_df_z

def final_classification(df, df_pred, pair):
    target = 'future_close_encoded'
    features = [ 'Predicted_x', 'Predicted_y', 'Predicted_z', 'Predicted_a',
       'Confidence_x', 'Confidence_y', 'Confidence_z', 'Confidence_a','atr', 'trend', 'ema_encoded_trend',
       '200_ma', 'long_condition', 'short_condition', 'market_direction']
    trainer_classification = ModelTrainer(df, features, target=target)
    ac, y_pred, y_pred_proba = trainer_classification.best_classifier_random_forest()
    alw = trainer_classification.get_test_data_with_predictions()

    if pair[:3] != "USD" or pair == "USD_JPY":
        new_df = alw[alw['Confidence'] >= .8]
        new_df_2 = new_df[new_df['Actual'] == new_df['Predicted']]
        
        accuracy_when_prediction_1 = len(new_df_2.index) / len(new_df.index)
        print(accuracy_when_prediction_1)
        print(trainer_classification.accuracy)
        df_combined = pd.merge(df, alw[['time','Predicted', 'Actual', 'Confidence']], on="time", how='inner')
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        df_comb_2 = df_combined
        if "close" not in df_comb_2.columns:
            df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence', 'future_close'])
            df_comb_2['close'] = df_comb_2['close_x']
        df_comb_2 = df_comb_2[df_comb_2.Confidence >= .9]
        df_comb_2['profit_loss'] = np.where(df_comb_2.Predicted== 1, df_comb_2.future_close - df_comb_2.close, df_comb_2.close - df_comb_2.future_close )
        result_sum = df_comb_2.profit_loss.sum() 
        row_count = len(df_comb_2)
        print(f"{result_sum} over {row_count}")    
        result_sum = df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']]['profit_loss'].sum()
        row_count = len(df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']])
    
        print(f"{result_sum} over {row_count}")
    
    
        scaled_features = trainer_classification.scaler.transform(df_pred[features])
        df_pred['Predicted_final'] = trainer_classification.classifier.predict(scaled_features)
        df_pred['Confidence_final'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))
        df_pred['classification_peak_accuracy'] = accuracy_when_prediction_1
    
        return df_pred, df_comb_2, df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']]

    else:
        alw['Predicted'] = np.where(alw.Predicted == 1, 0 ,1)
        #same as the rest
        new_df = alw[alw['Confidence'] >= .8]
        new_df_2 = new_df[new_df['Actual'] == new_df['Predicted']]
        accuracy_when_prediction_1 = len(new_df_2.index) / len(new_df.index)
        print(accuracy_when_prediction_1)
        manual_accuracy = len(alw[alw['Predicted'] == alw.Actual]) / len(alw)
        print(f"accuracy is {manual_accuracy}")
        df_combined = pd.merge(df, alw[['time','Predicted', 'Actual', 'Confidence']], on="time", how='inner')
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        df_comb_2 = df_combined
        if "close" not in df_comb_2.columns:
            df_comb_2 = df_combined.filter(['time', 'close_x', 'crossover_long_or_short', 'Predicted', 'Actual', 'Confidence', 'future_close'])
            df_comb_2['close'] = df_comb_2['close_x']
        df_comb_2 = df_comb_2[df_comb_2.Confidence >= .9]
        df_comb_2['profit_loss'] = np.where(df_comb_2.Predicted== 1, df_comb_2.future_close - df_comb_2.close, df_comb_2.close - df_comb_2.future_close )
        result_sum = df_comb_2.profit_loss.sum() 
        row_count = len(df_comb_2)
        print(f"{result_sum} over {row_count}")    
        result_sum = df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']]['profit_loss'].sum()
        row_count = len(df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']])
    
        print(f"{result_sum} over {row_count}")


        scaled_features = trainer_classification.scaler.transform(df_pred[features])  #now do opposite for the prediction df
        df_pred['Predicted_normal'] = trainer_classification.classifier.predict(scaled_features)
        df_pred['Predicted_final'] = np.where(df_pred.Predicted_normal == 1, 0, 1)
        df_pred['Confidence_final'] = np.max(trainer_classification.classifier.predict_proba(scaled_features))
        df_pred['classification_peak_accuracy'] = accuracy_when_prediction_1

    
        return df_pred, df_comb_2, df_comb_2[df_comb_2['market_direction'] == df_comb_2['Predicted']]
        

def setup_df_for_final_class(mg, pair_without_dash):
        
    mg = supertrend(mg)
    mg = add_ema_slope_class(mg, ema_lenght=9)
    mg = add_moving_averages(mg)
    mg= mg.dropna()
    mg = mg.reset_index(drop = True)
    mg= mg.drop(columns=['hl2','supertrend',  'basic_upperband','basic_lowerband','buy_signal', 'sell_signal', 'upperband', 'lowerband', 'fast_ma', 'short_ma', 'c_long', 'c_short', 'EMA_50_slope', 'EMA_50_diff', 'EMA_50' ])
    mg_2 = merger_with_market_direction(mg, pair_without_dash)
    mg_2['future_close_encoded'] = np.where(mg_2.future_close > mg_2.close, 1, 0)
    return mg_2

def df_for_news(currency_pair, timeframe, ema_length=30, shortlen=20, longlen=30, how_many_days=5,
                number_of_days_back=3, more_data=True, for_prediction=False, systen_of_class=3):

    price_df = get_last_1000_candles(currency_pair)
    price_df = price_df[price_df.complete]
    news_df = get_news()


    # Convert time columns to datetime format
    price_df['Time'] = pd.to_datetime(price_df['time'])
    news_df['Time'] = pd.to_datetime(news_df['Time'], format='%a %b %d, %Y, %I:%M %p')

    currency1, currency2 = currency_pair.split('_')
    filtered_news_df = news_df[news_df['Currency'].isin([currency1, currency2])]

    # Identify news containing "Interest"
    filtered_news_df['is_interest'] = filtered_news_df['News'].str.contains("Interest", case=False, na=False)

    # Remove duplicates but keep interest-related news
    non_interest_news_df = filtered_news_df[~filtered_news_df['is_interest']]
    interest_news_df = filtered_news_df[filtered_news_df['is_interest']]

    non_interest_news_df = non_interest_news_df.sort_values('Time').drop_duplicates(
        subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous']
    )

    # Combine back interest-related news to ensure they are not removed
    filtered_news_df = pd.concat([non_interest_news_df, interest_news_df]).sort_values('Time')

    # Merge news with price data
    merged_df = pd.merge_asof(
        price_df.sort_values('Time'),
        filtered_news_df.drop(columns=['is_interest']),  # Remove helper column
        on='Time',
        direction='backward',
        allow_exact_matches=False
    )

    no_news_mask = merged_df[['Currency', 'News', 'Forecast', 'Actual', 'Previous']].isna().all(axis=1)
    merged_df.loc[no_news_mask, ['Currency', 'News', 'Forecast', 'Actual', 'Previous']] = 'No News'

    # Remove duplicates from the merged DataFrame
    merged_df = merged_df.drop_duplicates(subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous'])

    merged_df = pd.merge(price_df, merged_df, on='Time', how='outer')

    merged_df['c'] = merged_df['c_x']
    merged_df['complete'] = merged_df['complete_x']
    merged_df['l'] = merged_df['l_x']
    merged_df['h'] = merged_df['h_x']
    merged_df['o'] = merged_df['o_x']
    merged_df['volume'] = merged_df['volume_x']

    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_y')])
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_x')])
    merged_df = merged_df.fillna(0)

    merged_df['Currency'] = np.where(merged_df['Currency'] == currency1, 2,
                                     np.where(merged_df['Currency'] == currency2, 1, 0))
    merged_df['News'] = merged_df['News'].astype(str)

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(merged_df[['News']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['News']))
    merged_df = pd.concat([merged_df, encoded_df], axis=1)

    merged_df = scale_by_group(merged_df)
    merged_df.drop(columns=['group_key'], inplace=True)

    if not for_prediction:
        merged_df['future_close'] = merged_df.c.shift(-5)

        if systen_of_class == 3:
            if ("JPY" not in currency_pair) and ("THB" not in currency_pair):
                merged_df['future_close_encoded'] = np.where(
                    merged_df.future_close > 0.005 + merged_df.c, 2,
                    np.where(merged_df.future_close < merged_df.c - 0.005, 0, 1)
                )
            else:
                merged_df['future_close_encoded'] = np.where(
                    merged_df.future_close > 0.5 + merged_df.c, 2,
                    np.where(merged_df.future_close < merged_df.c - 0.5, 0, 1)
                )

        elif systen_of_class == 2:
            merged_df['future_close_encoded'] = np.where(
                merged_df.future_close > merged_df.c, 1, 0
            )

    merged_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


    merged_df = merged_df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})

    test_df = merged_df

    df = add_accumulation_distribution(merged_df)
    df = add_ma_cross(df, shortlen=shortlen, longlen=longlen, how_many_days=how_many_days)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = BollingerBands(df)
    df = BB_encoder(df, how_many_days=how_many_days)
    df = is_close_above_ema(df)
    df = derivative_ema_in_future(df, how_many_days)
    df = BollingerBands(df)
    df = ATR(df)
    df = KeltnerChannels(df)
    df = MACD(df)
    df = calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, how_many_days=how_many_days)
    df = calculate_derivative_oscillator(df)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df = apply_patterns(df)
    df = clean_dataframe(df)
    df = df.rename(columns={"Time": 'time'})

    return df



def df_for_interference(currency_pair, ema_length=30, shortlen=20, longlen=30, how_many_days=5,
                number_of_days_back=3, for_prediction=True, systen_of_class=3):

    price_df = get_last_1000_candles(currency_pair)
    news_df = get_news()

    # Convert time columns to datetime format
    price_df['Time'] = pd.to_datetime(price_df['time'])
    news_df['Time'] = pd.to_datetime(news_df['Time'], format='%a %b %d, %Y, %I:%M %p')

    currency1, currency2 = currency_pair.split('_')
    filtered_news_df = news_df[news_df['Currency'].isin([currency1, currency2])]

    # Identify news containing "Interest"
    filtered_news_df['is_interest'] = filtered_news_df['News'].str.contains("Interest", case=False, na=False)

    # Remove duplicates but keep interest-related news
    non_interest_news_df = filtered_news_df[~filtered_news_df['is_interest']]
    interest_news_df = filtered_news_df[filtered_news_df['is_interest']]

    non_interest_news_df = non_interest_news_df.sort_values('Time').drop_duplicates(
        subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous']
    )

    # Combine back interest-related news to ensure they are not removed
    filtered_news_df = pd.concat([non_interest_news_df, interest_news_df]).sort_values('Time')

    # Merge news with price data
    merged_df = pd.merge_asof(
        price_df.sort_values('Time'),
        filtered_news_df.drop(columns=['is_interest']),  # Remove helper column
        on='Time',
        direction='backward',
        allow_exact_matches=False
    )

    no_news_mask = merged_df[['Currency', 'News', 'Forecast', 'Actual', 'Previous']].isna().all(axis=1)
    merged_df.loc[no_news_mask, ['Currency', 'News', 'Forecast', 'Actual', 'Previous']] = 'No News'

    # Remove duplicates from the merged DataFrame
    merged_df = merged_df.drop_duplicates(subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous'])

    merged_df = pd.merge(price_df, merged_df, on='Time', how='outer')

    merged_df['c'] = merged_df['c_x']
    merged_df['complete'] = merged_df['complete_x']
    merged_df['l'] = merged_df['l_x']
    merged_df['h'] = merged_df['h_x']
    merged_df['o'] = merged_df['o_x']
    merged_df['volume'] = merged_df['volume_x']

    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_y')])
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_x')])
    merged_df = merged_df.fillna(0)

    merged_df['Currency'] = np.where(merged_df['Currency'] == currency1, 2,
                                     np.where(merged_df['Currency'] == currency2, 1, 0))
    merged_df['News'] = merged_df['News'].astype(str)

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(merged_df[['News']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['News']))
    merged_df = pd.concat([merged_df, encoded_df], axis=1)

    merged_df = scale_by_group(merged_df)
    merged_df.drop(columns=['group_key'], inplace=True)

    if not for_prediction:
        merged_df['future_close'] = merged_df.c.shift(-5)

        if systen_of_class == 3:
            if ("JPY" not in currency_pair) and ("THB" not in currency_pair):
                merged_df['future_close_encoded'] = np.where(
                    merged_df.future_close > 0.005 + merged_df.c, 2,
                    np.where(merged_df.future_close < merged_df.c - 0.005, 0, 1)
                )
            else:
                merged_df['future_close_encoded'] = np.where(
                    merged_df.future_close > 0.5 + merged_df.c, 2,
                    np.where(merged_df.future_close < merged_df.c - 0.5, 0, 1)
                )

        elif systen_of_class == 2:
            merged_df['future_close_encoded'] = np.where(
                merged_df.future_close > merged_df.c, 1, 0
            )

    merged_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


    merged_df = merged_df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})

    test_df = merged_df

    df = add_accumulation_distribution(merged_df)
    df = add_ma_cross(df, shortlen=shortlen, longlen=longlen, how_many_days=how_many_days, future_or_not=False)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df = BollingerBands(df)
    df = BB_encoder(df, how_many_days=how_many_days)
    df = ATR(df)
    df = KeltnerChannels(df)
    df = MACD(df)
    df = calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, how_many_days=how_many_days, future=False)
    df = calculate_derivative_oscillator(df)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df = apply_patterns(df)
    df = df.rename(columns={"Time": 'time'})


    return df



def train_and_evaluate_rnn_model(df, features, target, pair, batch_size=32, epochs=150, confidence_threshold=0.9, timeframe="D", timesteps=10):
    rows_c = round(0.75 * len(df))
    valid_features = [feature for feature in features if feature in df.columns]
    X = df.loc[:, valid_features]
    y = df.loc[:, target]  

    X_train, X_test = X[:rows_c], X[rows_c:]
    y_train, y_test = y[:rows_c], y[rows_c:]
    df_test = df[rows_c:].copy()

    # Scale features
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    joblib.dump(sc, f'{pair}_{timeframe}_{target}_rnn_scaler_news.pkl')

    # Reshape input for LSTM (samples, timesteps, features)
    def reshape_for_rnn(X, timesteps):
        X_reshaped = []
        for i in range(len(X) - timesteps):
            X_reshaped.append(X[i:i + timesteps])
        return np.array(X_reshaped)

    X_train_rnn = reshape_for_rnn(X_train_sc, timesteps)
    X_test_rnn = reshape_for_rnn(X_test_sc, timesteps)

    # Adjust y_train and y_test to match new sequence shape
    y_train = y_train[timesteps:]
    y_test = y_test[timesteps:]

    # Build the RNN (LSTM model)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(timesteps, X_train.shape[1])),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=len(set(df[f'{target}'])), activation='softmax')  
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_rnn, y_train, batch_size=batch_size, epochs=epochs)

    # Save the model
    model.save(f'{pair}_{timeframe}_combined_model_rnn_news_{target}.h5')

    # Make predictions
    y_pred_prob = model.predict(X_test_rnn, batch_size=batch_size)  
    y_pred = np.argmax(y_pred_prob, axis=1)  

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Add predictions to df_test
    df_test = df_test.iloc[timesteps:]  # Align df_test with new y_test length
    df_test['Prediction_raw'] = y_pred
    df_test['Confidence'] = np.max(y_pred_prob, axis=1)  

    # Filter predictions with high confidence
    sdf = df_test[[target, "Prediction_raw", "Confidence"]]
    hp = sdf[sdf["Confidence"] > confidence_threshold]
    hp['correct'] = (hp[target] == hp["Prediction_raw"]).astype(int)

    # Calculate high-confidence accuracy
    accuracy_peak = (hp["correct"].sum() / len(hp)) if len(hp) > 0 else 0
    print(f"Accuracy for {pair} in {target} is {accuracy_peak}")

    df_test.to_csv(f"{pair}_{timeframe}_{target}_combined_rnn_news.csv", index=False)

    return accuracy, accuracy_peak, df_test
def train_and_evaluate_ann_model(df, features, target, pair, batch_size=32, epochs=150, confidence_threshold=0.9, timeframe="D"):
    rows_c = round(0.75 * len(df))
    valid_features = [feature for feature in features if feature in df.columns]
    X = df.loc[:, valid_features]
    y = df.loc[:, target]  

    X_train, X_test = X[:rows_c], X[rows_c:]
    y_train, y_test = y[:rows_c], y[rows_c:]
    df_test = df[rows_c:].copy()

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    joblib.dump(sc, f'{pair}_{timeframe}_{target}_gen_scaler_news.pkl')

    # Build the ANN
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation="relu"),
        tf.keras.layers.Dense(units=6, activation="relu"),
        tf.keras.layers.Dense(units=len(set(df[f'{target}'])), activation='softmax')  
    ])

    # Compile the model
    ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    ann.fit(X_train_sc, y_train, batch_size=batch_size, epochs=epochs)

    # Save the model
    ann.save(f'{pair}_{timeframe}_combined_model_news_{target}.h5')

    # Make predictions
    y_pred_prob = ann.predict(X_test_sc, batch_size=batch_size)  # Probabilities for each class
    y_pred = np.argmax(y_pred_prob, axis=1)  # Get predicted class

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Add predictions to df_test
    df_test['Prediction_raw'] = y_pred
    df_test['Confidence'] = np.max(y_pred_prob, axis=1)  # Highest probability for predicted class

    # Filter predictions with high confidence
    sdf = df_test[[target, "Prediction_raw", "Confidence"]]
    hp = sdf[sdf["Confidence"] > confidence_threshold]
    hp['correct'] = (hp[target] == hp["Prediction_raw"]).astype(int)

    # Calculate high-confidence accuracy
    accuracy_peak = (hp["correct"].sum() / len(hp)) if len(hp) > 0 else 0
    print(f"Accuracy for {pair} in {target} is {accuracy_peak}")

    # Save test set
    df_test.to_csv(f"{pair}_{timeframe}_{target}_combined_news.csv", index=False)

    # Combine train and test sets
    df_train = df[:rows_c].copy()
    combined_final = pd.concat([df_train, df_test])

    # Save the combined dataset
    combined_final.to_csv(f"{pair}_{timeframe}_{target}_combined_final_news.csv", index=False)

    return accuracy, accuracy_peak, df_test


target = 'future_close_encoded'

dict_for_features = {
    "GBP_USD":[ 'Currency',
 'Forecast',
 'Actual',
'News_Average Hourly Earnings (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (MoM)', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_GDP (YoY)', 'News_GDP (YoY)  (Q2)', 'News_GDP (YoY)  (Q3)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales', 'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',
 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS'
],
    "USD_JPY":[ 'Currency',
 'Forecast',
 'Actual',
'News_Average Hourly Earnings (MoM)', 'News_BoJ Interest Rate Decision', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales',  'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS'
],
    "AUD_USD":[ 'Currency',
 'Forecast',
 'Actual',
 'News_Average Hourly Earnings (MoM)', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)',  'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales',  'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_RBA Interest Rate Decision', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate', 
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS'
],
    "NZD_USD": [ 'Currency',
 'Forecast',
 'Actual',
'News_Average Hourly Earnings (MoM)', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales',  'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_RBNZ Interest Rate Decision', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS' 
],
    "USD_CAD":[ 'Currency',
 'Forecast',
 'Actual',
 'News_Average Hourly Earnings (MoM)', 'News_BoC Interest Rate Decision', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales',  'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS'
],
    "USD_CHF": [ 'Currency',
 'Forecast',
 'Actual',
'News_Average Hourly Earnings (MoM)', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)', 'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales', 'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS'
],
    "EUR_USD":[
 'Currency',
 'Forecast',
 'Actual',
 'Previous',
'News_Average Hourly Earnings (MoM)', 'News_CB Consumer Confidence', 'News_CPI (MoM)', 'News_CPI (YoY)', 'News_Chicago PMI', 'News_Core CPI (MoM)', 'News_Core PCE Price Index (MoM)', 'News_Core PCE Price Index (YoY)', 'News_Core Retail Sales (MoM)', 'News_Crude Oil Inventories', 'News_Durable Goods Orders (MoM)',  'News_Existing Home Sales', 'News_Fed Interest Rate Decision', 'News_GDP (QoQ)  (Q1)', 'News_GDP (QoQ)  (Q2)', 'News_GDP (QoQ)  (Q3)', 'News_GDP (QoQ)  (Q4)', 'News_German CPI (MoM)', 'News_German GDP (QoQ)  (Q2)', 'News_German GDP (QoQ)  (Q3)', 'News_German GDP (QoQ)  (Q4)', 'News_ISM Manufacturing PMI', 'News_ISM Manufacturing Prices', 'News_ISM Non-Manufacturing PMI', 'News_ISM Non-Manufacturing Prices', 'News_Initial Jobless Claims', 'News_JOLTS Job Openings', 'News_New Home Sales', 'News_Nonfarm Payrolls', 'News_PPI (MoM)', 'News_Philadelphia Fed Manufacturing Index', 'News_Retail Sales (MoM)', 'News_S&P Global Manufacturing PMI', 'News_S&P Global Services PMI', 'News_Unemployment Rate',
 'acc_dis',
 'short_ma_cro',
 'long_ma_cro',
 'short_ma_diff',
 'short_ma_slope',
 'long_ma_diff',
 'long_ma_slope',
 'ma_diff',
 'price_above_short_ma',
 'price_above_long_ma',
 'crossover_long_or_short',
 'EMA_50',
 'EMA_50_slope',
 'AD_EMA',
 'AD_EMA_slope',
 'RSI',
 'ATR',
 'fast_ma',
 'short_ma',
 '200_ma',
 'long_condition',
 'short_condition',
 'Change_AD',
 'Change_EMA',
 'close_above_EMA',

 'BB_MA',
 'BB_UP',
 'BB_LW',
 'BB_binary',
 'ATR_14',
 'EMA',
 'KeUp',
 'KeLo',
 'MACD',
 'SIGNAL_MACD',
 'HIST',
 'ROC',
 '+DI',
 '-DI',
 'ADX_14',
 'VWMA_14',
 'smoothed_rsi',
 'dosc',
 'derivative_oscillator',
 'Williams_R',
 'Williams_encoded',
 'Conversion_Line',
 'Base_Line',
 'Leading_Span_A',
 'Leading_Span_B',

 'TSI',
 'TSI_Binary',

 'body_size_prev',
 'direction_prev',
 'direction_prev_2',
 'HANGING_MAN',
 'SHOOTING_STAR',
 'SPINNING_TOP',
 'MARUBOZU',
 'ENGULFING',
 'TWEEZER_TOP',
 'TWEEZER_BOTTOM',
 'MORNING_STAR',
 'EVENING_STAR',
 'THREE_BLACK_CROWS',
 'THREE_WHITE_SOLDIERS']
    
    
}


def rnn_profitability(df, confidence_threshold = .9):
    df['profit_loss'] = np.where(df.Prediction_raw.isin([2]), df.future_close - df.close, 
                        np.where(df.Prediction_raw.isin([0]), df.close - df.future_close, 0))
    
    df_wq = df[df.Confidence > confidence_threshold]
    gain_pip_per_trade = df_wq.profit_loss.sum() / len(df_wq)
    profit_total = df_wq.profit_loss.sum()
    trades_cont = len(df_wq)

    return gain_pip_per_trade, profit_total, trades_cont
    

def technicals_addder(merged_df):
    number_of_days_back = 3 
    ema_length = 30 
    shortlen = 20 
    longlen = 30 
    how_many_days = 5
    df = add_accumulation_distribution(merged_df)
    df = add_ma_cross(df, shortlen=shortlen, longlen = longlen, how_many_days=how_many_days)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df= derivative_ema_in_future(df, how_many_days)
    df= BollingerBands(df)
    df = BB_encoder(df, how_many_days = how_many_days)
    df = ATR(df)
    df= KeltnerChannels(df) 
    df = MACD(df)
    df= calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, how_many_days=how_many_days)
    df = calculate_derivative_oscillator(df)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df = apply_patterns(df)
    df = clean_dataframe(df)
    df = df.rename(columns = {"Time":'time'})
    return df

def for_prediction_2(pair, for_prediction = True):
    price_df = get_last_1000_candles(pair)
    
    news_df = get_news()
    
    # Convert time columns to datetime format
    price_df['Time'] = pd.to_datetime(price_df['time'])
    news_df['Time'] = pd.to_datetime(news_df['Time'], format='%a %b %d, %Y, %I:%M %p')
    
    currency1, currency2 = pair.split('_')    
    filtered_news_df = news_df[news_df['Currency'].isin([currency1, currency2])]
    
    # Identify news containing "Interest"
    filtered_news_df['is_interest'] = filtered_news_df['News'].str.contains("Interest", case=False, na=False)
    
    # Remove duplicates but keep interest-related news
    non_interest_news_df = filtered_news_df[~filtered_news_df['is_interest']]
    interest_news_df = filtered_news_df[filtered_news_df['is_interest']]
    
    non_interest_news_df = non_interest_news_df.sort_values('Time').drop_duplicates(
        subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous']
    )
    
    # Combine back interest-related news to ensure they are not removed
    filtered_news_df = pd.concat([non_interest_news_df, interest_news_df]).sort_values('Time')
    
    # Merge news with price data
    merged_df = pd.merge_asof(
        price_df.sort_values('Time'),
        filtered_news_df.drop(columns=['is_interest']),  # Remove helper column
        on='Time',
        direction='backward',
        allow_exact_matches=False  
    )
    
    no_news_mask = merged_df[['Currency', 'News', 'Forecast', 'Actual', 'Previous']].isna().all(axis=1)
    merged_df.loc[no_news_mask, ['Currency', 'News', 'Forecast', 'Actual', 'Previous']] = 'No News'
    
    # Remove duplicates from the merged DataFrame
    merged_df = merged_df.drop_duplicates(subset=['Time', 'Currency', 'News', 'Forecast', 'Actual', 'Previous'])
    
    
    merged_df = pd.merge(price_df, 
                        merged_df, 
                        on='Time',
                        how= 'outer')
    
    merged_df['c'] = merged_df['c_x']
    merged_df['complete'] = merged_df['complete_x']
    merged_df['l'] = merged_df['l_x']
    merged_df['h'] = merged_df['h_x']
    merged_df['o'] = merged_df['o_x']  # Fix typo: 'o_h' should be 'o_x'
    merged_df['volume'] = merged_df['volume_x']
    
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_y')])
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_x')])
    merged_df = merged_df.fillna(0)
    merged_df['Currency'] = np.where(merged_df['Currency']==currency1, 2,
                                    np.where(merged_df['Currency'] == currency2,1,0 ))
    merged_df['News'] = merged_df['News'].astype(str)
    encoder = OneHotEncoder(sparse_output=False) 
    encoded = encoder.fit_transform(merged_df[['News']])
    
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['News']))
    
    merged_df = pd.concat([merged_df, encoded_df], axis=1)
    merged_df = scale_by_group(merged_df)
    merged_df.drop(columns=['group_key'], inplace=True)
    
    if not for_prediction:
        merged_df['future_close'] = merged_df.c.shift(-5)
        
        if "JPY" not in pair:
            merged_df['future_close_encoded'] = np.where(
                merged_df.future_close > 0.005 + merged_df.c, 2, 
                np.where(merged_df.future_close < merged_df.c - 0.005, 0, 1)
            )
        else:
            merged_df['future_close_encoded'] = np.where(
                merged_df.future_close > 0.5 + merged_df.c, 2, 
                np.where(merged_df.future_close < merged_df.c - 0.5, 0, 1))
    
    elif for_prediction:
        pass


    number_of_days_back = 3 
    ema_length = 30 
    shortlen = 20 
    longlen = 30 
    how_many_days = 5
    merged_df = merged_df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low"})
    df = add_accumulation_distribution(merged_df)
    df = add_ma_cross(df, shortlen=shortlen, longlen = longlen, how_many_days=how_many_days)
    df = add_ema_slope(df, ema_lenght=ema_length)
    df = calculate_price_diff_and_gain_loss(df)
    df = calculate_future_price_diff_and_gain_loss(df, how_many_days)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = add_moving_averages(df)
    df = change_in_slope(df, number_of_days_back)
    df = is_close_above_ema(df)
    df= derivative_ema_in_future(df, how_many_days)
    df= BollingerBands(df)
    df = BB_encoder(df, how_many_days = how_many_days)
    df = ATR(df)
    df= KeltnerChannels(df) 
    df = MACD(df)
    df= calculate_rate_of_change(df)
    df = ADX(df)
    df = VWMA(df, how_many_days=how_many_days)
    df = calculate_derivative_oscillator(df)
    df = WilliamsR(df)
    df = Ichimoku(df)
    df = DMI(df)
    df = TSI(df)
    df = apply_patterns(df)
    df = df.rename(columns = {"Time":'time'})
    df['Forecast'] = df['Forecast'].astype(float)
    df['Actual'] = df['Actual'].astype(float)
    df['Previous'] = df['Previous'].astype(float)
    return df





def train_and_evaluate_model(df, features, target, pair, timeframe="D", confidence_threshold=0.76, model_type="xgb"):
    rows_c = round(0.75 * len(df))
    valid_features = [feature for feature in features if feature in df.columns]
    X = df.loc[:, valid_features]
    y = df.loc[:, target]

    X_train, X_test = X[:rows_c], X[rows_c:]
    y_train, y_test = y[:rows_c], y[rows_c:]
    df_test = df[rows_c:].copy()
    df_train = df[:rows_c].copy()


    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    joblib.dump(sc, f'{pair}_scaler_{timeframe}.pkl')

    if model_type == "xgb":
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_sc, y_train)
        best_params = grid_search.best_params_
        print(f"Best XGBoost Params for {pair}: {best_params}")

        model = xgb.XGBClassifier(**best_params)

    else:  
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'num_leaves': [20, 31, 40],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        model = lgb.LGBMClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_sc, y_train)
        best_params = grid_search.best_params_
        print(f"Best LightGBM Params for {pair}: {best_params}")

        model = lgb.LGBMClassifier(**best_params)


    model.fit(X_train_sc, y_train)
    joblib.dump(model, f'{pair}_{timeframe}_{model_type}_model.pkl')


    y_pred_prob_train = model.predict_proba(X_train_sc)
    y_pred_train = np.argmax(y_pred_prob_train, axis=1)
    df_train['Prediction_raw'] = y_pred_train
    df_train['Confidence'] = np.max(y_pred_prob_train, axis=1)
    
    y_pred_prob_test = model.predict_proba(X_test_sc)
    y_pred = np.argmax(y_pred_prob_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    df_test['Prediction_raw'] = y_pred
    df_test['Confidence'] = np.max(y_pred_prob_test, axis=1)
    print(y_test.value_counts())


    sdf = df_test[[target, "Prediction_raw", "Confidence"]]
    hp = sdf[sdf["Confidence"] > confidence_threshold]
    hp['correct'] = (hp[target] == hp["Prediction_raw"]).astype(int)
    accuracy_peak = (hp["correct"].sum() / len(hp)) if len(hp) > 0 else 0
    
    df_complete = pd.concat([df_train, df_test])
    
    print(f"{model_type.upper()} Accuracy for {pair} in {target} is {accuracy_peak}")
    return accuracy, accuracy_peak, df_test, df_complete, best_params


class MultiModelClassifier:
    def __init__(self, df, features, target, pair, batch_size=16, epochs=130, confidence_threshold=0.9, timeframe="D", timesteps=10):
        self.df = df
        self.features = [feature for feature in features if feature in df.columns]
        self.target = target
        self.pair = pair
        self.batch_size = batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.timeframe = timeframe
        self.timesteps = timesteps
        self.models = {}
        self.scaler = None
        self.forecast_steps = None
     #   self.freq = "D"
      #  self.prediction_length = 5  
       # self.start_train = pd.to_datetime(self.df.iloc[:1, :]['time'].values)[0]
       # self.start_test =   pd.to_datetime(df.iloc[round(0.75 * len(self.df)):, :]['time'].values)[0]

    def _prepare_data(self, time_required=False):
        df = self.df.copy()
        
        if time_required:
            df['time_idx'] = range(len(df))  
    
            X = df[self.features + ['time_idx']]  
            y = df[self.target]
    
            rows_c = round(0.75 * len(df))
            X_train, X_test = X[:rows_c], X[rows_c:]
            y_train, y_test = y[:rows_c], y[rows_c:]
            df_test = df[rows_c:].copy()

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train.drop(columns=['time_idx']))
            X_test_sc = scaler.transform(X_test.drop(columns=['time_idx']))
    

            X_train_sc = np.column_stack((X_train['time_idx'], X_train_sc))
            X_test_sc = np.column_stack((X_test['time_idx'], X_test_sc))
    
        else:
            X = df[self.features]
            y = df[self.target]
    
            rows_c = round(0.75 * len(df))
            X_train, X_test = X[:rows_c], X[rows_c:]
            y_train, y_test = y[:rows_c], y[rows_c:]
            df_test = df[rows_c:].copy()
    
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
    

        self.scaler = scaler
        joblib.dump(scaler, f'{self.pair}_{self.timeframe}_{self.target}_scaler.pkl')
    
        return X_train_sc, X_test_sc, y_train, y_test, df_test
    
    





    def train_arima(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
    
        model = auto_arima(y_train, seasonal=False, stepwise=True, trace=True)
    
        arima_model = ARIMA(y_train, order=model.order)
        arima_model_fit = arima_model.fit()
    
        joblib.dump(arima_model_fit, f'{self.pair}_{self.timeframe}_arima_model.pkl')
    
        # Make forecast
        forecast_steps = len(y_test)
        self.forecast_steps = forecast_steps
        y_pred = arima_model_fit.forecast(steps=forecast_steps)
    
        mse = mean_squared_error(y_test, y_pred)
        
        # Confidence calculation
        confidence = 1 / (1 + mse) 
        
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Model Confidence: {confidence:.2f}") 
    
        df_test['Prediction_arima'] = y_pred
        df_test['Confidence_arima'] = confidence
        self.models['arima'] = arima_model_fit
    
        return df_test

    def train_ann(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        ann = Sequential([
            Dense(units=8, activation="relu"),
            Dense(units=8, activation="relu"),
            Dense(units=len(set(self.df[self.target])), activation='softmax')
        ])
        ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        ann.fit(X_train_sc, y_train, batch_size=self.batch_size, epochs=self.epochs)
        ann.save(f'{self.pair}_{self.timeframe}_ann_model.h5')

        y_pred_prob = ann.predict(X_test_sc, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        df_test['Prediction_ann'] = y_pred
        df_test['Confidence_ann'] = np.max(y_pred_prob, axis=1)

        self.models['ann'] = ann
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        
        return df_test


        

    def train_rnn(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        
        def reshape_for_rnn(X, timesteps):
            X_reshaped = [X[i:i + timesteps] for i in range(len(X) - timesteps)]
            return np.array(X_reshaped)
        
        X_train_rnn = reshape_for_rnn(X_train_sc, self.timesteps)
        X_test_rnn = reshape_for_rnn(X_test_sc, self.timesteps)
        y_train, y_test = y_train[self.timesteps:], y_test[self.timesteps:]

        rnn = Sequential([
            LSTM(units=8, return_sequences=True, input_shape=(self.timesteps, X_train_sc.shape[1])),
            LSTM(units=8),
            Dense(units=len(set(self.df[self.target])), activation='softmax')
        ])

        
        rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        rnn.fit(X_train_rnn, y_train, batch_size=self.batch_size, epochs=self.epochs)
        rnn.save(f'{self.pair}_{self.timeframe}_rnn_model.h5')

        y_pred_prob = rnn.predict(X_test_rnn, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        df_test = df_test.iloc[self.timesteps:]
        df_test['Prediction_rnn'] = y_pred
        df_test['Confidence_rnn'] = np.max(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        self.models['rnn'] = rnn

        
        return df_test


    def train_cnn(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        X_train_cnn = X_train_sc[..., np.newaxis]
        X_test_cnn = X_test_sc[..., np.newaxis]
        
        cnn = Sequential([
            Conv1D(filters=6, kernel_size=3, activation='relu', input_shape=(X_train_sc.shape[1], 1)),
            Conv1D(filters=6, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(units=8, activation='relu'),
            Dense(units=len(set(self.df[self.target])), activation='softmax')
        ])
        cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn.fit(X_train_cnn, y_train, batch_size=self.batch_size, epochs=self.epochs)
        cnn.save(f'{self.pair}_{self.timeframe}_cnn_model.h5')
        
        y_pred_prob = cnn.predict(X_test_cnn, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        df_test['Prediction_cnn'] = y_pred
        df_test['Confidence_cnn'] = np.max(y_pred_prob, axis=1)
        
        self.models['cnn'] = cnn
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        return df_test
        
 


    def train_tabnet(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        
        tabnet = TabNetClassifier()
        tabnet.fit(X_train_sc, y_train, max_epochs=self.epochs, batch_size=self.batch_size)
        
        y_pred_prob = tabnet.predict_proba(X_test_sc)
        y_pred = np.argmax(y_pred_prob, axis=1)
        df_test['Prediction_tabnet'] = y_pred
        df_test['Confidence_tabnet'] = np.max(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        self.models['tabnet'] = tabnet
        return df_test

    
    def train_gru(self):
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        
        def reshape_for_rnn(X, timesteps):
            X_reshaped = [X[i:i + timesteps] for i in range(len(X) - timesteps)]
            return np.array(X_reshaped)
        
        X_train_gru = reshape_for_rnn(X_train_sc, self.timesteps)
        X_test_gru = reshape_for_rnn(X_test_sc, self.timesteps)
        y_train, y_test = y_train[self.timesteps:], y_test[self.timesteps:]
        
        gru = Sequential([
            GRU(units=8, return_sequences=True, input_shape=(self.timesteps, X_train_sc.shape[1])),
            GRU(units=8),
            Dense(units=len(set(self.df[self.target])), activation='softmax')
        ])
        
        gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        gru.fit(X_train_gru, y_train, batch_size=self.batch_size, epochs=self.epochs)
        
        y_pred_prob = gru.predict(X_test_gru, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        df_test = df_test.iloc[self.timesteps:]
        df_test['Prediction_gru'] = y_pred
        df_test['Confidence_gru'] = np.max(y_pred_prob, axis=1)
        
        self.models['gru'] = gru
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        return df_test

    def train_xgb_boost(self):
        df_merged = pd.read_csv("optuma_given_param.csv")
        params = (df_merged[(df_merged.pair == self.pair) & (df_merged.model == 'XBGboost')]['param']).values[0]  # Not a typo!!!
        params = ast.literal_eval(params)
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
        model.fit(X_train_sc, y_train)
        
        joblib.dump(model, f'{self.pair}_{self.timeframe}_xgb_model.pkl') 
        
        y_pred_prob_train = model.predict_proba(X_test_sc)
        y_pred_train = np.argmax(y_pred_prob_train, axis=1)
        
        df_test['Prediction_xgb'] = y_pred_train
        df_test['Confidence_xgb'] = np.max(y_pred_prob_train, axis=1)
        
        self.models['xgb'] = model 
        
        accuracy = accuracy_score(y_test, y_pred_train) 
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        return df_test
        


        
    def train_lgb_boost(self):
        df_merged = pd.read_csv("optuma_given_param.csv")
        params = (df_merged[(df_merged.pair == self.pair) & (df_merged.model == 'LBGboost')]['param']).values[0]  # Not a typo!!!
        params = ast.literal_eval(params)
        X_train_sc, X_test_sc, y_train, y_test, df_test = self._prepare_data()
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_sc, y_train)
        
        joblib.dump(model, f'{self.pair}_{self.timeframe}_lgb_model.pkl') 
        
        y_pred_prob_train = model.predict_proba(X_test_sc)
        y_pred_train = np.argmax(y_pred_prob_train, axis=1)
        
        df_test['Prediction_lgb'] = y_pred_train
        df_test['Confidence_lgb'] = np.max(y_pred_prob_train, axis=1)
        
        self.models['lgb'] = model 
        
        accuracy = accuracy_score(y_test, y_pred_train) 
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        return df_test


 
    def run_all_models(self):
        #df_test_nbeats = self.train_nbeats()
        df_test_arima = self.train_arima()
        df_test_xgb = self.train_xgb_boost()
        df_test_lgb = self.train_lgb_boost()
        df_test_gru = self.train_gru()
        df_test_ann = self.train_ann()
        df_test_rnn = self.train_rnn()
        df_test_cnn = self.train_cnn()


        data_frames = [df_test_ann, df_test_rnn, df_test_cnn, df_test_gru, df_test_xgb, df_test_lgb,  df_test_arima]
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=self.df.index.name or self.df.index.names[0], how='outer'), data_frames)
        return df_merged
class MultiModelClassifierPredict:
    def __init__(self, trained_model, df, features, target, pair, confidence_threshold=0.9):
        self.df = df
        self.features = [feature for feature in features if feature in df.columns]
        self.target = target
        self.pair = pair
        self.confidence_threshold = confidence_threshold
        self.trained_model = trained_model
        self.batch_size = 16
        self.timesteps = 10
        self.forecast_steps = trained_model.forecast_steps

    def _prepare_data_for_prediction(self):
        X = self.df[self.features]
        X_sc = self.trained_model.scaler.transform(X)
        return X_sc

    def predict_arima(self):
        arima_model = self.trained_model.models['arima']

        forecast_result = arima_model.get_forecast(steps=self.forecast_steps)
        
        y_pred = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int() 
    
        confidence = 1 / (1 + (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean())
        
        df_test = self.df.copy()
        df_test['Prediction_arima'] = np.nan
        df_test.iloc[-len(y_pred):, df_test.columns.get_loc('Prediction_arima')] = y_pred
        df_test['Confidence_arima'] = confidence
        
        return df_test


    def predict_ann(self):
        X_sc = self._prepare_data_for_prediction()
        ann = self.trained_model.models['ann']
        y_pred_prob = ann.predict(X_sc, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_ann'] = y_pred
        df_test['Confidence_ann'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def predict_rnn(self):
        X_sc = self._prepare_data_for_prediction()
        X_rnn = np.array([X_sc[i:i + self.timesteps] for i in range(len(X_sc) - self.timesteps)])
        rnn = self.trained_model.models['rnn']
        
        y_pred_prob = rnn.predict(X_rnn, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.iloc[self.timesteps:].copy()
        df_test['Prediction_rnn'] = y_pred
        df_test['Confidence_rnn'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def predict_cnn(self):
        X_sc = self._prepare_data_for_prediction()
        cnn = self.trained_model.models['cnn']
        
        X_cnn = X_sc[..., np.newaxis]
        y_pred_prob = cnn.predict(X_cnn, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_cnn'] = y_pred
        df_test['Confidence_cnn'] = np.max(y_pred_prob, axis=1)
        
        return df_test


    def predict_tabnet(self):
        X_sc = self._prepare_data_for_prediction()
        
        # Convert to DataFrame for NaN handling
        X_df = pd.DataFrame(X_sc)
    
        # Identify valid rows (no NaN values)
        valid_mask = ~X_df.isna().any(axis=1)
        
        # Get indices of valid rows
        valid_indices = X_df.index[valid_mask]
    
        # Drop NaN rows from X_sc for prediction
        X_sc_clean = X_df.dropna().values
    
        tabnet = self.trained_model.models['tabnet']
    
        y_pred_prob = tabnet.predict_proba(X_sc_clean)
        y_pred = np.argmax(y_pred_prob, axis=1)
    
        # Copy the original DataFrame
        df_test = self.df.copy()
    
        # Assign predictions only to valid indices
        df_test.loc[valid_indices, 'Prediction_tabnet'] = y_pred
        df_test.loc[valid_indices, 'Confidence_tabnet'] = np.max(y_pred_prob, axis=1)
    
        return df_test


    def predict_xgb_boost(self):
        X_sc = self._prepare_data_for_prediction()
        model = self.trained_model.models['xgb']
        
        y_pred_prob = model.predict_proba(X_sc)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_xgb'] = y_pred
        df_test['Confidence_xgb'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def predict_lgb_boost(self):
        X_sc = self._prepare_data_for_prediction()
        model = self.trained_model.models['lgb']
        
        y_pred_prob = model.predict_proba(X_sc)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        df_test = self.df.copy()
        df_test['Prediction_lgb'] = y_pred
        df_test['Confidence_lgb'] = np.max(y_pred_prob, axis=1)
        
        return df_test

    def run_all_models(self):
        models = [
            self.predict_arima(), self.predict_xgb_boost(), self.predict_lgb_boost(),
            self.predict_ann(), self.predict_rnn(), self.predict_cnn(),
            self.predict_tabnet()
        ]
        
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=self.df.index.name or self.df.index.names[0], how='outer'), models)
        return df_merged

