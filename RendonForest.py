import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier # 導入隨機森林分類器
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 數據加載與初步處理 ---
# 讀取數據
df_stock = pd.read_csv("2330TW.csv", encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
df_index = pd.read_csv("TWSE_21.01to23.12.csv", encoding="utf-8-sig", parse_dates=["日期"], index_col="日期")
df_analysis = pd.read_csv("analysis_ternaryB.csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
df_C = pd.read_csv("analysis_ternaryC.csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
# 按日期合併數據
df_index.index.name = "Date"
df = pd.merge(df_stock, df_index, on="Date", how="left", suffixes=("", ""))
df.index.name = "date"
df = pd.merge(df, df_analysis, on="date", how="left", suffixes=("", ""))
df["analysis"].fillna(0, inplace=True)
df = pd.merge(df, df_C, on="date", how="left", suffixes=("", "_C"))
df["analysis_C"].fillna(0, inplace=True)
df.fillna(method='ffill', inplace=True)

# 計算每日收盤價變動率
df['Price_Change'] = df['Close'].pct_change()

# 檢查並處理缺失值 (pct_change 會產生 NaN)
print("--- 數據缺失值檢查 ---")
print(df.isnull().sum())
df.dropna(inplace=True)

# --- 2. 特徵選擇與目標變數定義 ---
#, 'MA_10', 'W%R_12', 'BIAS_6', 'AD', 'CCI_10', 'Volume', 'Open_diff', 'High_diff', 'Low_diff','Price_Change', "收盤指數", "analysis", "analysis_C"
features = [
    'RSI_12', 'K', 'D','MACD', 'MA_5'
]
target = 'UpDown' # 原始目標欄位

missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"錯誤：CSV 檔案中缺少以下欄位: {', '.join(missing_cols)}")
    exit()

# 準備特徵 X (使用當前時間點 t 的特徵)
X_data = df[features].copy()

# * 修正：計算真實的 5 天後漲跌作為目標 y *
# 1. 獲取 5 天後的收盤價
df['Close_Future_5d'] = df['Close'].shift(-5)

# 2. 創建新的目標欄位 'Target_5d'
#    如果 5 天後收盤價 > 今天收盤價，則為 1 (漲)，否則為 0 (跌或平)
df['Target_5d'] = (df['Close_Future_5d'] > df['Close']).astype(int)

# 3. 準備 y_data (使用新計算的 Target_5d)
y_data = df['Target_5d'].copy()
#y_data = df[target].shift(-1)

# 由於 shift(-1) 會在最後一行產生 NaN，需要再次處理
# 同時也需要確保 X 和 y 的索引對齊
combined = pd.concat([X_data, y_data], axis=1)
combined.dropna(inplace=True) # 刪除最後一行因 shift 產生的 NaN

X_data = combined[features]
y_data = combined['Target_5d']#

print(f"--- 特徵與目標準備完畢 ---")
print(f"X shape: {X_data.shape}")
print(f"y shape: {y_data.shape}")

# --- 3. 數據標準化 ---
# 雖然隨機森林對縮放不敏感，但保持縮放是個好習慣，方便未來更換模型
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(X_data)
X_scaled = pd.DataFrame(scaled_features, index=X_data.index, columns=features)

# --- 4. 數據集分割 (訓練、驗證、測試) ---
# 仍然按時間順序分割，以模擬真實預測情況
# 這裡為了簡化，直接分成訓練集和測試集 (80% 訓練, 20% 測試)
# 如果需要更精細的調優，可以再分出驗證集
train_size = int(len(X_scaled) * 0.8)

X_train, y_train = X_scaled[:train_size], y_data[:train_size]
X_test, y_test = X_scaled[train_size:], y_data[train_size:]

print(f"--- 數據集大小 ---")
print(f"訓練集 X_train: {X_train.shape}")
print(f"測試集 X_test: {X_test.shape}")

# 檢查類別分佈
print("--- 類別分佈檢查 ---")
print(f"訓練集 y_train: 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
print(f"測試集 y_test:  0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

# --- 5. 建立並訓練隨機森林模型 ---
# n_estimators: 森林中樹的數量
# random_state: 控制隨機性，確保結果可重現
# class_weight='balanced': 自動調整權重以應對可能的輕微不平衡
# n_jobs=-1: 使用所有可用的 CPU 核心進行訓練
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

print("--- 開始訓練隨機森林模型 ---")
model.fit(X_train, y_train)
print("--- 模型訓練完成 ---")

# --- 6. 模型評估 ---
# 在測試集上進行預測
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"--- 模型評估 (測試集) ---")
print(f"準確率 (Accuracy): {accuracy:.4f}")

# 計算並顯示混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print("--- 混淆矩陣 (Confusion Matrix) ---")
print("   預測 0   預測 1")
# 確保 cm 有兩個維度
if cm.shape == (1, 1):
    if np.unique(y_test)[0] == 0: cm_display = np.array([[cm[0,0], 0], [0, 0]])
    else: cm_display = np.array([[0, 0], [0, cm[0,0]]])
elif cm.shape == (1, 2):
     if np.unique(y_test)[0] == 0: cm_display = np.array([[cm[0,0], cm[0,1]], [0, 0]])
     else: cm_display = np.array([[0, 0], [cm[0,0], cm[0,1]]])
elif cm.shape == (2, 1):
    if y_pred.size > 0 and int(round(y_pred[0])) == 0 : cm_display = np.array([[cm[0,0], 0], [cm[1,0], 0]])
    elif y_pred.size > 0: cm_display = np.array([[0, cm[0,0]], [0, cm[1,0]]])
    else: cm_display = np.array([[0,0],[0,0]]) # Handle empty y_pred if necessary
else: cm_display = cm

print(f"實際 0: {cm_display[0,0]:^7d} {cm_display[0,1]:^7d}")
print(f"實際 1: {cm_display[1,0]:^7d} {cm_display[1,1]:^7d}")


# 顯示更詳細的分類報告 (包含 Precision, Recall, F1-score)
print("--- 分類報告 (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=['跌/平 (0)', '漲 (1)']))

# 繪製混淆矩陣熱力圖
plt.figure(figsize=(6, 4))
sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', xticklabels=['預測 跌/平 (0)', '預測 漲 (1)'], yticklabels=['實際 跌/平 (0)', '實際 漲 (1)'])
plt.xlabel('預測標籤 (Predicted Label)')
plt.ylabel('真實標籤 (True Label)')
plt.title('混淆矩陣 (Confusion Matrix) - 隨機森林')
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"無法設定中文字體，圖表標籤可能顯示異常: {e}")
plt.show()

# --- 7. 特徵重要性分析 (隨機森林特有) ---
# 查看模型認為哪些特徵對預測最重要
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # 按重要性排序

plt.figure(figsize=(12, 6))
plt.title("特徵重要性 (Feature Importances) - 隨機森林")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("--- 最重要的特徵 ---")
for i in range(len(features)):
    print(f"{i+1}. 特徵: {features[indices[i]]} ({importances[indices[i]]:.4f})")


print("--- 程式執行完畢 ---")