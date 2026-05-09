import requests
import pandas as pd
from io import StringIO
import datetime
import numpy as np
import time
start_d = '20.08'
end_d = '23.12'
stock_symbol = '0050'

def DateConvert(twdate):#將民國換算
    try:
        x = twdate.split('/')
        return datetime.date(int(x[0])+1911, int(x[1]), int(x[2]))
    except:
        pass

#抓取台股資料https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY
#台股大盤https://www.twse.com.tw/indicesReport/MI_5MINS_HIST
url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
params = {}
params['stockNo'] = stock_symbol
params['response'] = "csv"
df = pd.DataFrame()
def stock_data(month):#抓月資料
    time.sleep(1)
    params['date'] = month
    print(month)
    r = requests.get(url, params=params)

    df1 = pd.read_csv(StringIO(r.text), engine='python', skiprows=1, skipfooter=4, thousands=',')
    df1['日期'] = df1['日期'].map(DateConvert)
    df1 = df1.loc[:, ~df1.columns.str.contains("Unnamed")]
    try:
        df1 = df1.drop("成交金額", axis='columns')
        df1 = df1.drop("漲跌價差", axis='columns')
        df1 = df1.drop("成交筆數", axis='columns')
    except:
        pass
    df1.replace('None', np.nan, inplace=True)
    df1.dropna(how='all', inplace=True)
    return df1

def ym(d,n):#月分加減
    y = int(d.split(".")[0])
    m = int(d.split(".")[1])
    m = m+n
    if m == 0:
        y = y - 1
        m = 12
    elif m == 13:
        y = y + 1
        m = 1
    if m < 10:
        return str(y) + '.0' + str(m)
    else:
        return str(y) + '.' + str(m)

#main
loop = ym(start_d,-1)
while(loop != end_d):
    loop = ym(loop,1)
    #print(loop)
    df1 = stock_data('20'+ loop.split(".")[0] + loop.split(".")[1] +'01')
    #print(df1)
    time.sleep(3)#避免API限制
    df = pd.concat([df, df1])
try:
    #MA
    df.set_index("日期", inplace=True)
    df["MA10"] = df["收盤價"].rolling(window=10).mean()
    df["MA50"] = df["收盤價"].rolling(window=50).mean()
except:
    pass
df.to_csv(stock_symbol + '_' + start_d + 'to' + end_d + '.csv',encoding='utf-8-sig')