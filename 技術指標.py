import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, StochRSIIndicator
from ta.trend import MACD, CCIIndicator
from ta.volume import acc_dist_index

# 下載台股資料（以 2330.TW 台積電為例）
ticker = '2603.TW'
start_d = '2020-11-01'
end_d = '2023-12-31'
df = yf.download(ticker, start=start_d, end=end_d)
df.columns = df.columns.droplevel('Ticker')
df.dropna(inplace=True)

# 計算開盤價差、最高價差、最低價差
df['Open_diff'] = df['Open'].diff()
df['High_diff'] = df['High'].diff()
df['Low_diff'] = df['Low'].diff()

# RSI（12日）
df['RSI_12'] = RSIIndicator(close=df['Close'], window=12).rsi()

# MA（5日、10日、20日）
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# 威廉指標 W%R（12日）
high12 = df['High'].rolling(window=12).max()
low12 = df['Low'].rolling(window=12).min()
df['W%R_12'] = -100 * ((high12 - df['Close']) / (high12 - low12))

# 乖離率（BIAS）
df['BIAS_6'] = (df['Close'] - df['Close'].rolling(window=6).mean()) / df['Close'].rolling(window=6).mean() * 100
df['BIAS_12'] = (df['Close'] - df['Close'].rolling(window=12).mean()) / df['Close'].rolling(window=12).mean() * 100

# 心理線（PSY）
df['up'] = df['Close'].diff() > 0
df['PSY_6'] = df['up'].rolling(window=6).sum() / 6 * 100

# KD 指標（9日）
stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=9, smooth_window=3)
df['K'] = stoch.stoch()
df['D'] = stoch.stoch_signal()

# MACD（12, 26, 9）
macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd_diff()

# 成交量（千股）
df['Volume'] = df['Volume'] / 1000

# 動量指標（MTM）
df['MTM_10'] = df['Close'] - df['Close'].shift(10)

# A/D 指標
ad = acc_dist_index(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
df['AD'] = ad

# CCI（10日）
df['CCI_10'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=10).cci()

# 漲跌（以明天的收盤價與今天比較）
df['UpDown'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 顯示結果
print(df.tail())
df.to_csv(ticker + '_' + start_d + '_' + end_d + '.csv',encoding='utf-8-sig')
