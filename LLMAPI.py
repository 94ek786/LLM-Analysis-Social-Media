import os
import pandas as pd
import openai
import time

# 設定 OpenAI API Key（請替換成你的 API Key）
#API URL https://free.v36.cm
#API KEY sk-j0140PE9nBvx74pG9d04F1431aBe43518b869439F391BaEd
openai.base_url = "https://free.v36.cm/v1/"
openai.api_key = "sk-j0140PE9nBvx74pG9d04F1431aBe43518b869439F391BaEd"

# 讀取 CSV 檔案
read_class = 'B'
file_paths = ["filtered_" + read_class + ".csv"]
dfs = []

for file in file_paths:
    if os.path.exists(file):  # 確保檔案存在
        df = pd.read_csv(file)
        df['category'] = file.split('_')[1][0]  # 取 A, B, C 作為分類標籤
        dfs.append(df)

# 合併所有分類的資料
df_all = pd.concat(dfs, ignore_index=True)

# 確保 'date' 欄位為日期格式
df_all['date'] = pd.to_datetime(df_all['date'])

# 依照日期分組
grouped_by_date = df_all.groupby('date')

# 嘗試讀取已存檔的分析結果，避免重複處理
analysis_file = "daily_analysis" + read_class + ".csv"
if os.path.exists(analysis_file):
    df_existing = pd.read_csv(analysis_file)
    processed_dates = set(df_existing['date'])  # 已處理過的日期集合
    analysis_results = df_existing.to_dict('records')  # 轉為列表
else:
    processed_dates = set()
    analysis_results = []

# 定義 GPT-3.5 分析函式
def analyze_articles(date, articles):
    """使用 GPT-3.5 對每日文章內容進行總結分析"""
    prompt_text = f"請針對以下 {len(articles)} 篇文章進行分析並一起算出這些文章總和的情緒分數：\n\n"
    
    for idx, article in enumerate(articles):
        prompt_text += f"{idx+1}. {article}\n\n"
    
    prompt_text += "請依照以下格式[今日的情緒分數為:(score)]，將這些文章給出一個全部的情緒指標分數"

    # 呼叫 OpenAI API
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "你是一個金融市場情緒分析師，你的任務是分析每天的文章，並給出1至10分的情緒指標分數，其中標準為5分，當文章中有看好未來市場的描述時加分，當文章中不看好時則扣分。"},
                      {"role": "user", "content": prompt_text}],
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API 錯誤（{date.strftime('%Y-%m-%d')}）：{e}")
        return "API 分析失敗"

# 依照日期處理文章
for date, group in grouped_by_date:
    date_str = date.strftime('%Y-%m-%d')

    # 跳過已處理的日期
    if date_str in processed_dates:
        print(f"跳過 {date_str}，因為已經分析過。")
        continue

    articles = group['content'].dropna().tolist()  # 取得當日所有文章內容
    if not articles:
        continue

    print(f"分析 {date_str} 的文章...")
    analysis = analyze_articles(date, articles)

    # 加入分析結果
    analysis_results.append({"date": date_str, "analysis": analysis})

    # **每次完成一天的分析後立即存檔**
    pd.DataFrame(analysis_results).to_csv(analysis_file, index=False, encoding='utf-8-sig')

    print(f"已儲存 {date_str} 的分析結果到 {analysis_file}。")
    
    # 避免 API 限制，每次請求間隔 2 秒
    time.sleep(2)

print("每日文章分析完成，已儲存至 "+ analysis_file)