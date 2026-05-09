import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Input, Multiply, Softmax, Dense, Dropout, LSTM, Bidirectional, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import json

# 讀取數據
df_stock = pd.read_csv("2330TW.csv", encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
df_index = pd.read_csv("TWSE_21.01to23.12.csv", encoding="utf-8-sig", parse_dates=["日期"], index_col="日期")
df_analysis = pd.read_csv("analysis_contentOnlyA.csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
df_C = pd.read_csv("analysis_contentOnlyC.csv", encoding="utf-8-sig", parse_dates=["date"], index_col="date")
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

# 選擇特徵變數　
features_to_scale = ['RSI_12', 'K', 'D','MACD', 'MA_5', 'MA_10', 'W%R_12', 'BIAS_6', 'AD', 'CCI_10', 'Volume', 'Open_diff', 'High_diff', 'Low_diff','Price_Change', "收盤指數", "analysis", "analysis_C"]
features_not_scale = []

all_features = features_to_scale + features_not_scale

# 計算漲跌標籤
df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)  # 5天後的價格比現在高 => 1，否則 0

# 刪除未來5天無法標記的行
df.dropna(inplace=True)

# 標準化數據
# 切分訓練與測試資料索引
train_index = df.loc["2021-01-01":"2022-12-31"].index
test_index = df.loc["2023-01-01":].index
# 用訓練資料 fit scaler
scaler = MinMaxScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(df.loc[train_index, features_to_scale]), columns=features_to_scale, index=train_index)
scaled_test = pd.DataFrame(scaler.transform(df.loc[test_index, features_to_scale]), columns=features_to_scale, index=test_index)
# 合併非標準化欄位
scaled_train = pd.concat([scaled_train, df.loc[train_index, features_not_scale]], axis=1)
scaled_test = pd.concat([scaled_test, df.loc[test_index, features_not_scale]], axis=1)
# 再合併回 df_scaled
df_scaled = pd.concat([scaled_train, scaled_test], axis=0).sort_index()
try:
    df_scaled["analysis"] = df_scaled["analysis"] * 1000
    df_scaled["analysis_C"] = df_scaled["analysis_C"] * 1000
except:
    pass

#
loop1 = 5
mr = []
result = [
    {"time_steps":5, "sum": 0, "num":0},
    {"time_steps":10, "sum": 0, "num":0},
    {"time_steps":15, "sum": 0, "num":0},
    {"time_steps":20, "sum": 0, "num":0}
]
while loop1 < 21:
    loop = 1
    while loop < 11:
        time_steps = loop1  # 使用過去N天的數據來預測
        def create_sequences(data, labels, time_steps):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:i + time_steps])
                y.append(labels[i + time_steps])
            return np.array(X), np.array(y)

        # 切分訓練與測試集
        train_size = df.loc["2021-01-01":"2022-12-31"].shape[0]
        train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]
        train_labels, test_labels = df['target'].values[:train_size], df['target'].values[train_size:]

        # 產生序列數據
        X_train, y_train = create_sequences(train_data, train_labels, time_steps)
        X_test, y_test = create_sequences(test_data, test_labels, time_steps)

        # LSTM 模型
        input_layer = Input(shape=(time_steps, X_train.shape[2]))

        # 取得每個特徵的重要性權重（共享時間軸）
        feature_attention = Dense(X_train.shape[2], activation='softmax', name='feature_attention')(
            Lambda(lambda x: K.mean(x, axis=1))(input_layer)  # 對時間步平均
        )
        feature_weights = Lambda(lambda x: K.expand_dims(x, axis=1))(feature_attention)  # shape: (batch, 1, features)

        # 對原始輸入乘上權重（Broadcasting）
        weighted_input = Multiply()([input_layer, feature_weights])

        # 接後續 LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(weighted_input)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        x = LSTM(32)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])


        # 訓練模型
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=16, callbacks=[early_stopping])

        # 預測結果
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # 計算準確率
        overall_accuracy = accuracy_score(y_test, y_pred) * 100

        actual_up = y_test == 1
        actual_down = y_test == 0
        correct_up = np.sum((y_pred == y_test) & actual_up)
        correct_down = np.sum((y_pred == y_test) & actual_down)
        up_accuracy = (correct_up / np.sum(actual_up)) * 100 if np.sum(actual_up) > 0 else 0
        down_accuracy = (correct_down / np.sum(actual_down)) * 100 if np.sum(actual_down) > 0 else 0
        """
        # 建立只輸出 attention 權重的新模型
        attention_model = Model(inputs=model.input,
                            outputs=model.get_layer("feature_attention").output)

        # 使用 test 資料的平均輸入來取得 attention 權重
        attention_weights = attention_model.predict(X_test)
        # 權重 shape = (batch_size, num_features)
        average_weights = np.mean(attention_weights, axis=0)
        plt.figure(figsize=(10, 5))
        plt.bar(all_features, average_weights)
        plt.xticks(rotation=45)
        plt.title("Feature Attention Weights")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.show()

        # 繪製 Loss 曲線
        plt.figure(figsize=(8, 5))  
        plt.plot(history.history["loss"], label="Train Loss", color="blue")
        plt.plot(history.history["val_loss"], label="Validation Loss", color="red")
        plt.title("LSTM Model Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Binary Crossentropy)")
        plt.legend()
        plt.show()
        

        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        

        # 顯示結果
        mr.append("時間序列"+ str(loop1) +"第"+ str(loop) +"訓練")
        mr.append(f"整體準確率: {overall_accuracy:.2f}%")
        mr.append(f"實際上漲時的預測準確率: {up_accuracy:.2f}%")
        mr.append(f"實際下跌時的預測準確率: {down_accuracy:.2f}%")
        """
        if loop1 == 5:
            result[0]["sum"] = result[0]["sum"] + overall_accuracy
            result[0]["num"] = result[0]["num"] + 1
        elif loop1 == 10:
            result[1]["sum"] = result[1]["sum"] + overall_accuracy
            result[1]["num"] = result[1]["num"] + 1
        elif loop1 == 15:
            result[2]["sum"] = result[2]["sum"] + overall_accuracy
            result[2]["num"] = result[2]["num"] + 1
        elif loop1 == 20:
            result[3]["sum"] = result[3]["sum"] + overall_accuracy
            result[3]["num"] = result[3]["num"] + 1
        loop = loop + 1
    loop1 = loop1 + 5

for i in mr:
    #print(i)
    pass
print(result)
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)