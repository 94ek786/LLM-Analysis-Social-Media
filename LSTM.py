import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

# 讀取數據
df_stock = pd.read_csv("0050.TW_indicators.csv", encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
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

# 選擇特徵變數　, "analysis", "analysis_C"
features = ["Volume", "Close", "OBV", "OBV_change", "BB_MID", "BB_STD", "BB_UPPER", "BB_LOWER", "收盤指數", "EMA_5", "EMA_20", "MACD", "MACD_signal", "MACD_hist", "RSI_14"]
target = "Close"

# 數據標準化
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features + [target]])

# 訓練集 & 測試集（2021-2022 訓練，2023 測試）
train_size = df.loc["2021-01-01":"2022-12-31"].shape[0]
train_data = df_scaled[:train_size]
test_data = df_scaled[train_size:]

# LSTM 需要時間步長，因此定義一個函數
def create_sequences(data, time_steps, future_days):
    X, y = [], []
    for i in range(len(data) - time_steps - future_days ):
        #print(data[i:i + time_steps])
        X.append(data[i:i + time_steps])  # 輸入特徵
        y.append(data[i + time_steps + future_days, -1])  # 預測收盤價
    return np.array(X), np.array(y)



loop1 = 5
mr = []
result = [
    {"time_steps":5, "num":0, "mean_percentage_error": 0, "MAE": 0, "accuracy": 0},
    {"time_steps":10, "num":0, "mean_percentage_error": 0, "MAE": 0, "accuracy": 0},
    {"time_steps":15, "num":0, "mean_percentage_error": 0, "MAE": 0, "accuracy": 0},
    {"time_steps":20, "num":0, "mean_percentage_error": 0, "MAE": 0, "accuracy": 0}
]
while(loop1 < 21):
    loop2 = 1
    while(loop2 < 101):
        #當 val_loss 連續 20 個 epoch 沒有改善時，自動停止訓練
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-5, restore_best_weights=True)
        # 生成訓練和測試數據
        time_steps = loop1  # N天作為一個輸入序列
        future_days = 5     # 預測 N 天後的價格
        X_train, y_train = create_sequences(train_data, time_steps, future_days)
        X_test, y_test = create_sequences(test_data, time_steps, future_days)
        #print(test_data[time_steps + future_days:, :-1].shape)
        #print(y_test.reshape(-1, 1).shape) 
        # 構建 LSTM 模型
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
            Dropout(0.1),
            LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(25, activation="relu"),
            Dense(1)  # 預測單一數值（收盤價）
        ])

        # 編譯模型
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        # 訓練模型
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=16, callbacks=[early_stopping])

        # 預測股價
        y_pred = model.predict(X_test)
        # 反向標準化
        y_test_actual = scaler.inverse_transform(np.concatenate((test_data[time_steps + future_days:, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
        y_pred_actual = scaler.inverse_transform(np.concatenate((test_data[time_steps + future_days:, :-1], y_pred), axis=1))[:, -1]
        """
        # 繪製結果
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[train_size+time_steps+future_days:], y_test_actual, label="Actual Price", color="blue")
        plt.plot(df.index[train_size+time_steps+future_days:], y_pred_actual, label="Predicted Price", color="red")
        plt.title("2330 Stock Price Prediction with LSTM")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # 繪製訓練 Loss 曲線
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train Loss", color="blue")
        plt.plot(history.history["val_loss"], label="Validation Loss", color="red")
        plt.title("Model Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.show()
        """

        #計算預測與實際的漲跌準確率
        # ======= 計算預測與實際的漲跌方向 =======
        actual_trend = np.sign(np.diff(y_test_actual))  # 計算實際漲跌 (+1: 上漲, -1: 下跌)
        predicted_trend = np.sign(np.diff(y_pred_actual))  # 計算預測漲跌
        # ======= 整體漲跌準確率 =======
        overall_accuracy = np.mean(actual_trend == predicted_trend) * 100
        # ======= 上漲與下跌的細分準確率 =======
        actual_up = actual_trend == 1  # 找出實際上漲的天數
        actual_down = actual_trend == -1  # 找出實際下跌的天數

        correct_up = np.sum((actual_trend == predicted_trend) & actual_up)  # 預測正確且實際上漲
        correct_down = np.sum((actual_trend == predicted_trend) & actual_down)  # 預測正確且實際下跌

        up_accuracy = (correct_up / np.sum(actual_up)) * 100 if np.sum(actual_up) > 0 else 0
        down_accuracy = (correct_down / np.sum(actual_down)) * 100 if np.sum(actual_down) > 0 else 0
        # ======= 印出結果 =======
        #mr.append("預測" + str(loop1) + "天後，使用" + str(loop2) + "個輸入序列")
        mr.append(f"漲跌準確率: {overall_accuracy:.2f}%")
        #mr.append(f"實際上漲時的預測準確率: {up_accuracy:.2f}%")
        #mr.append(f"實際下跌時的預測準確率: {down_accuracy:.2f}%")

        #計算數字偏差（MSE 與 MAE）
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        #mr.append(f"均方誤差 (MSE): {mse:.4f}")
        mr.append(f"平均絕對誤差 (MAE): {mae:.4f}")

        # ======= 計算測試期間的平均誤差百分比 =======
        percentage_error = abs(y_pred_actual - y_test_actual) / y_test_actual * 100
        mean_percentage_error = np.mean(percentage_error)
        mr.append(f"平均誤差百分比: {mean_percentage_error:.2f}%")
        if loop1 == 5:
            result[0]["accuracy"] = result[0]["accuracy"] + overall_accuracy
            result[0]["mean_percentage_error"] = result[0]["mean_percentage_error"] + mean_percentage_error
            result[0]["MAE"] = result[0]["MAE"] + mae
            result[0]["num"] = result[0]["num"] + 1
        elif loop1 == 10:
            result[1]["accuracy"] = result[1]["accuracy"] + overall_accuracy
            result[1]["mean_percentage_error"] = result[1]["mean_percentage_error"] + mean_percentage_error
            result[1]["MAE"] = result[1]["MAE"] + mae
            result[1]["num"] = result[1]["num"] + 1
        elif loop1 == 15:
            result[2]["accuracy"] = result[2]["accuracy"] + overall_accuracy
            result[2]["mean_percentage_error"] = result[2]["mean_percentage_error"] + mean_percentage_error
            result[2]["MAE"] = result[2]["MAE"] + mae
            result[2]["num"] = result[2]["num"] + 1
        elif loop1 == 20:
            result[3]["accuracy"] = result[3]["accuracy"] + overall_accuracy
            result[3]["mean_percentage_error"] = result[3]["mean_percentage_error"] + mean_percentage_error
            result[3]["MAE"] = result[3]["MAE"] + mae
            result[3]["num"] = result[3]["num"] + 1
        loop2 = loop2 + 1
    loop1 = loop1 + 5
for i in mr:
    #print(i)
    pass
print(result)
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)