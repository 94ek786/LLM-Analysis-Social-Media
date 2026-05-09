# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC # 導入支援向量機分類器
from sklearn.metrics import accuracy_score
import itertools # 導入用於產生組合的工具
import json # 導入用於輸出 JSON 的工具
import time

# --- 0. 設定與特徵列表 ---
stock_id = '2603'
# 您提供的所有候選特徵
all_candidate_features = [
    'MA_10', 'W%R_12', 'BIAS_6', 'AD', 'CCI_10', 'Volume', 
    'Open_diff', 'High_diff', 'Low_diff','Price_Change', "收盤指數", 
    'RSI_12', 'K', 'D','MACD', 'MA_5', "analysis", "analysis_C"
]

# *** 1. 設定您要固定的特徵 ***
# 這些特徵將會出現在 *所有* 的組合中
fixed_features = []

# *** 2. 設定要額外測試的可選特徵數量 ***
# 程式將測試 (固定特徵 + 0個可選特徵), (固定特徵 + 1個可選特徵), ..., (固定特徵 + N個可選特徵)
# 例如：設為 3，將測試 (固定+0), (固定+1), (固定+2), (固定+3) 的組合
# 警告：如果您有 13 個可選特徵，設為 13 會跑 2^13 = 8192 次，仍需很久
MAX_OPTIONAL_FEATURES = 9 

# 輸出的 JSON 檔案名稱
output_filename = "Svm" + stock_id + ".json"

# --- 自動計算可選特徵 ---
# 檢查固定特徵是否在候選清單中
for f in fixed_features:
    if f not in all_candidate_features:
        print(f"錯誤：您指定的固定特徵 '{f}' 不在 'all_candidate_features' 清單中。")
        exit()

# 'optional_features' 是 'all_candidate_features' 中排除了 'fixed_features' 之後的特徵
optional_features = [f for f in all_candidate_features if f not in fixed_features]

# --- 1. 數據加載與初步處理 ---
# (與您原始腳本相同)
try:
    df_stock = pd.read_csv(stock_id + "TW.csv", encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
    df_index = pd.read_csv("TWSE_21.01to23.12.csv", encoding="utf-8-sig", parse_dates=["日期"], index_col="日期")
    df_analysis = pd.read_csv("analysis_ternary" + stock_id + ".csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
    df_C = pd.read_csv("analysis_ternaryC.csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e.filename}。請確認所有 CSV 檔案都在同一個資料夾中。")
    exit()

df_index.index.name = "Date"
df = pd.merge(df_stock, df_index, on="Date", how="left", suffixes=("", ""))
df.index.name = "date"
df = pd.merge(df, df_analysis, on="date", how="left", suffixes=("", ""))
df["analysis"].fillna(0, inplace=True)
df = pd.merge(df, df_C, on="date", how="left", suffixes=("", "_C"))
df["analysis_C"].fillna(0, inplace=True)
df.fillna(method='ffill', inplace=True)
df['Price_Change'] = df['Close'].pct_change()
df.dropna(inplace=True) # 刪除包含 Price_Change NaN 的第一行

# --- 2. 特徵選擇與目標變數定義 ---

# 檢查所有需要的候選特徵是否存在
missing_cols = [col for col in all_candidate_features if col not in df.columns]
if missing_cols:
    print(f"錯誤：CSV 檔案中缺少以下特徵欄位: {', '.join(missing_cols)}")
    exit()
if 'Close' not in df.columns:
     print(f"錯誤：CSV 檔案中缺少 'Close' 欄位，無法計算未來漲跌。")
     exit()

# 準備特徵 X (使用 *所有* 候選特徵)
X_data_all = df[all_candidate_features].copy()

# * 修正：計算真實的 5 天後漲跌作為目標 y *
df['Close_Future_5d'] = df['Close'].shift(-5)
df['Target_5d'] = (df['Close_Future_5d'] > df['Close']).astype(int)
target_col = 'Target_5d' # 確定目標欄位名稱

# 準備 y_data
y_data = df[target_col].copy()

# 合併 X 和 y，並刪除 NaN (主要來自 y_data 末尾的 NaN)
combined = pd.concat([X_data_all, y_data], axis=1)
combined.dropna(inplace=True)

X_data_all = combined[all_candidate_features]
y_data = combined[target_col]

print(f"--- 特徵與目標準備完畢 (預測 5 天後漲跌) ---")
print(f"總候選特徵數: {len(all_candidate_features)}")
print(f"固定特徵: {fixed_features}")
print(f"可選特徵: {len(optional_features)} 個 ({optional_features[:3]}...)")
print(f"X (all features) shape: {X_data_all.shape}")
print(f"y shape: {y_data.shape}")

# --- 3. 數據標準化 (一次性處理所有特徵) ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features_all = scaler.fit_transform(X_data_all)
X_scaled_all = pd.DataFrame(scaled_features_all, index=X_data_all.index, columns=all_candidate_features)

# --- 4. 數據集分割 (一次性分割) ---
train_size = int(len(X_scaled_all) * 0.8)
X_train_full, y_train = X_scaled_all[:train_size], y_data[:train_size]
X_test_full, y_test = X_scaled_all[train_size:], y_data[train_size:]

print(f"--- 數據集大小 ---")
print(f"訓練集 X_train_full: {X_train_full.shape}")
print(f"測試集 X_test_full: {X_test_full.shape}")
print(f"訓練集 y_train (5天後漲跌): 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
print(f"測試集 y_test (5天後漲跌):  0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

# --- 5. 循環測試特徵組合 (固定 + 可選) ---
results = [] # 用於儲存所有結果

# 檢查 MAX_OPTIONAL_FEATURES 是否超過可選特徵的總數
if MAX_OPTIONAL_FEATURES > len(optional_features):
    MAX_OPTIONAL_FEATURES = len(optional_features)
    print(f"警告：MAX_OPTIONAL_FEATURES 已重設為 {MAX_OPTIONAL_FEATURES} (可選特徵的總數)")

# 決定 r 的起始值。
# 如果 fixed_features 是空的，r 必須從 1 開始，否則會產生 0 特徵組合。
# 如果 fixed_features 不是空的，r 從 0 開始 (代表只測試固定特徵)。
start_r = 0
if not fixed_features: # 檢查 fixed_features 是否為空
    start_r = 1
    print("注意：未指定固定特徵。將從 1 個可選特徵開始測試。")

# r 從 0 開始 (代表只用固定特徵)
total_combinations = sum(len(list(itertools.combinations(optional_features, r))) for r in range(0, MAX_OPTIONAL_FEATURES + 1))
print(f"\n--- 開始測試特徵組合 ---")
print(f"將測試 {len(fixed_features)} 個固定特徵 + 0 到 {MAX_OPTIONAL_FEATURES} 個可選特徵的組合。")
print(f"總共需要訓練 {total_combinations} 個模型...")

start_time = time.time()
combination_count = 0

# r = 0 代表只用固定特徵
# r = 1 代表 固定特徵 + 1個可選特徵 ...
for r in range(start_r, MAX_OPTIONAL_FEATURES + 1):
    
    num_total_features = len(fixed_features) + r
    print(f"\n--- 正在測試 {num_total_features} 個特徵的組合 (固定: {len(fixed_features)}, 可選: {r}) ---")
    
    # 獲取 "可選特徵" 的組合
    optional_combinations = itertools.combinations(optional_features, r)
    
    for opt_combo in optional_combinations:
        combination_count += 1
        
        # 最終的組合 = 固定的 + 這次迭代的可選組合
        combo_list = fixed_features + list(opt_combo)
        
        # 1. 從已分割的數據中選取特徵
        X_train = X_train_full[combo_list]
        X_test = X_test_full[combo_list]
        
        # 2. 建立並訓練 SVM 模型
        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # 3. 評估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 4. 儲存結果
        result_entry = {
            "使用的特徵": combo_list,
            "總特徵數": len(combo_list),
            "其的輸出結果": accuracy # 儲存準確率
        }
        results.append(result_entry)
        
        # 打印進度
        if combination_count % 20 == 0 or total_combinations < 20:
            print(f"  已完成 {combination_count} / {total_combinations} (準確率: {accuracy:.4f} for {combo_list})")

end_time = time.time()
print(f"\n--- 所有組合測試完畢 ---")
print(f"總共耗時: {(end_time - start_time):.2f} 秒")
print(f"已訓練 {combination_count} 個模型。")

# --- 6. 將結果排序並輸出到 JSON 檔案 ---
# 根據 "其的輸出結果" (準確率) 降冪排序
sorted_results = sorted(results, key=lambda x: x['其的輸出結果'], reverse=True)

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 確保中文特徵名稱能正確顯示
        # indent=4 讓 JSON 檔案格式化，易於閱讀
        json.dump(sorted_results, f, ensure_ascii=False, indent=4)
    print(f"\n--- 結果已成功排序並儲存至 {output_filename} ---")
    print(f"--- 準確率最高的前 3 名 ---")
    for i, res in enumerate(sorted_results[:3]):
        print(f"  Top {i+1}: {res['其的輸出結果']:.4f} - 特徵: {res['使用的特徵']}")
        
except IOError as e:
    print(f"錯誤：無法寫入 JSON 檔案: {e}")

print("--- 程式執行完畢 ---")