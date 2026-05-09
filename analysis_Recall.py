import pandas as pd
import numpy as np

# 1. 載入數據
readF = "analysis_ternary2603.csv"
dayN = 5
try:
    # 必須確保 'date' 欄位被正確解析為日期並設為索引
    df = pd.read_csv(readF, parse_dates=["date"], index_col="date")
except FileNotFoundError:
    print("錯誤：找不到檔案。請確認檔案路徑是否正確。")
    exit()

# 確保 'analysis' 欄位是數值型態，以便進行加總
df['analysis'] = pd.to_numeric(df['analysis'], errors='coerce')

# 2. 【核心】計算過去 3 個日曆日（72 小時）內所有數據的加總
# window='3D' 確保計算基於日曆時間
# min_periods=1 確保在 72 小時內只要有 1 筆數據就進行加總
# 注意：此處使用 '3D' 作為時間窗，包含當天
#       若您希望嚴格只計算 T, T-1, T-2 三天，且中間的日期沒有資料，
#       則此方法會包含所有在 3D 時間窗內出現的數據點。
df['analysis_3day_sum'] = df['analysis'].rolling(
    window=str(dayN)+'D', 
    min_periods=1 
).sum()

# 3. 定義映射函數
# 根據要求：Sum > 0 -> 1, Sum = 0 -> 0, Sum < 0 -> -1
def map_to_ternary(value):
    if pd.isna(value):
        # 由於 min_periods=1，只有在數據集開頭 3 天內都沒有數據時才會出現 NaN，
        # 但為求保險仍保留 NaN 處理
        return np.nan 
    elif value > 0:
        return 1.0
    elif value < 0:
        return -1.0
    else:
        return 0.0

# 4. 應用映射函數到新的加總欄位，產生最終的回顧特徵
df['analysis_3day_recall'] = df['analysis_3day_sum'].apply(map_to_ternary)

# 5. 執行替換與清理操作
# 5a. 將新的三天回顧數據覆蓋回原始的 'analysis' 欄位
df['analysis'] = df['analysis_3day_recall']

# 5b. 刪除所有臨時的計算欄位
df.drop(columns=['analysis_3day_sum', 'analysis_3day_recall'], inplace=True)


# 6. 輸出處理後的結果，驗證數據結構
print("--- 處理後的 DataFrame 結構 (基於日曆日 3D window) ---")
print(df.head(10))

print("\n--- 數據說明 ---")
print("analysis 欄位已更新。數值代表過去 3 個日曆日內所有有數據日期的分數總和，並映射到 {-1, 0, 1}。")

df.to_csv(str(dayN) + "DayRecall"+readF,encoding='utf-8-sig')