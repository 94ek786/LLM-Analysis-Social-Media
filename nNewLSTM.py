# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # 導入 Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 數據加載與初步處理 ---
file_path = '0050.TW_2018-11-01_2024-12-31.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_path}。請確認檔案路徑是否正確。")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Price_Change'] = df['Close'].pct_change()

print("--- 數據缺失值檢查 (加入 Price_Change 後) ---")
print(df.isnull().sum())
df.dropna(inplace=True)

# --- 2. 特徵選擇與目標變數定義 ---
features = [
    'RSI_12', 'MA_5', 'MA_10', 'W%R_12', 'BIAS_6', 'K', 'D',
    'MACD', 'AD', 'CCI_10', 'Volume', 'Open_diff', 'High_diff', 'Low_diff',
    'Price_Change', 'PSY_6', 'MTM_10'
]
target = 'UpDown'

missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"錯誤：CSV 檔案中缺少以下欄位: {', '.join(missing_cols)}")
    exit()

data = df[features + [target]].copy()

# --- 3. 數據標準化 ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])

# --- 4. 創建時間序列數據集 ---
time_steps = 30
X, y = [], []

for i in range(time_steps, len(scaled_features)):
    X.append(scaled_features[i-time_steps:i])
    y.append(data[target].iloc[i])

X, y = np.array(X), np.array(y)

print(f"--- 數據維度 ---")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# --- 5. 數據集分割 (訓練、驗證、測試) ---
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"--- 數據集大小 ---")
print(f"訓練集: {X_train.shape[0]} 筆")
print(f"驗證集: {X_val.shape[0]} 筆")
print(f"測試集: {X_test.shape[0]} 筆")

print("--- 類別分佈檢查 ---")
print(f"訓練集 y_train: 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
print(f"驗證集 y_val:   0={np.sum(y_val == 0)}, 1={np.sum(y_val == 1)}")
print(f"測試集 y_test:  0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

# --- 6. 建立 LSTM 模型 (使用 Bidirectional) ---
model = Sequential()
# * 修改：使用 Bidirectional 包裹 LSTM 層 *
# Bidirectional 會創建一個正向 LSTM 和一個反向 LSTM，並將它們的輸出合併（預設是拼接）
model.add(Bidirectional(LSTM(units=32), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# 輸出層 (二元分類)
model.add(Dense(units=1, activation='sigmoid'))

# 編譯模型
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 顯示模型架構
print("--- 模型架構 (使用 Bidirectional LSTM) ---")
model.summary() # 注意看參數數量會比單向 LSTM 多

# --- 7. 訓練模型 ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("--- 開始訓練模型 (Bidirectional, 小批次) ---")
# * 修改：減小批次大小 *
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16, # <--- 減小批次大小
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
print("--- 模型訓練完成 ---")

# --- 8. 模型評估 ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"--- 模型評估 (測試集) ---")
print(f"損失 (Loss): {loss:.4f}")
print(f"準確率 (Accuracy): {accuracy:.4f}")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

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
    if int(round(y_pred[0])) == 0: cm_display = np.array([[cm[0,0], 0], [cm[1,0], 0]])
    else: cm_display = np.array([[0, cm[0,0]], [0, cm[1,0]]])
else: cm_display = cm

print(f"實際 0: {cm_display[0,0]:^7d} {cm_display[0,1]:^7d}")
print(f"實際 1: {cm_display[1,0]:^7d} {cm_display[1,1]:^7d}")

# 繪製混淆矩陣熱力圖
plt.figure(figsize=(6, 4))
sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', xticklabels=['預測 跌/平 (0)', '預測 漲 (1)'], yticklabels=['實際 跌/平 (0)', '實際 漲 (1)'])
plt.xlabel('預測標籤 (Predicted Label)')
plt.ylabel('真實標籤 (True Label)')
plt.title('混淆矩陣 (Confusion Matrix)')
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"無法設定中文字體，圖表標籤可能顯示異常: {e}")
plt.show()

# --- 9. 繪製訓練過程中的損失和準確率 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='訓練準確率 (Training Accuracy)')
plt.plot(history.history['val_accuracy'], label='驗證準確率 (Validation Accuracy)')
plt.title('模型準確率 (Model Accuracy)')
plt.xlabel('輪數 (Epoch)')
plt.ylabel('準確率 (Accuracy)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='訓練損失 (Training Loss)')
plt.plot(history.history['val_loss'], label='驗證損失 (Validation Loss)')
plt.title('模型損失 (Model Loss)')
plt.xlabel('輪數 (Epoch)')
plt.ylabel('損失 (Loss)')
plt.legend()
plt.tight_layout()
plt.show()

print("--- 程式執行完畢 ---")
