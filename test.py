import pandas as pd

# 讀取數據
sID =  "2603"
df = pd.read_csv("analysis_ternary" + sID + ".csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")

total_count = len(df)

# 計算各值的比例
proportion = df.value_counts(normalize=True)  # `normalize=True` 會計算比例

# 顯示結果
print(sID + "資料共有" + str(df.size) + "筆")
print("TWSE數據比例")
print(proportion)
"""
result = [
    {"d":5, "sum": 0, "num":0},
    {"d":10, "sum": 0, "num":0}
]
loop = 0
while loop < 10:
    loop = loop + 1
    result[1]["sum"] = result[1]["sum"] + loop
    result[1]["num"] = result[1]["num"] + 1
print(result)"""