import os
import transformers
import torch
import time
import pandas as pd

# 檢查 GPU 是否可用
if torch.cuda.is_available():
    print("CUDA 可用！")
else:
    print("CUDA 不可用，將在 CPU 上運行。")
    
model_id = "Llama3.1_8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


# 讀取 CSV 檔案
read_class = 'B'
file_paths = ["filtered_" + read_class + ".csv"]
dfs = []

for file in file_paths:
    if os.path.exists(file):  # 確保檔案存在
        df = pd.read_csv(file)
        df['category'] = file.split('_')[1][0]  # 取 A, B, C 作為分類標籤
        dfs.append(df)

#------------main
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

#分析
def analyze_articles(date, articles):
    start_time = time.perf_counter()

    # 要計時的程式碼
    prompt_text = f"請針對以下 {len(articles)} 篇文章進行分析並一起算出這些文章整合的情緒分數：\n\n"

    for idx, (content, messages) in enumerate(articles):
        prompt_text += f"第 {idx+1} 篇文章:\n內容: {content}\n留言(|符號隔開的代表不同人的留言): {messages}\n\n"
    
    prompt_text += "請依照以下格式[今日的情緒分數為:(score)]，將這些文章給出一個整合的情緒指標分數"

    
    messages=[{"role": "system", "content": "你是一個金融市場情緒分析師，你的任務是分析每天的文章，並給出-5至+5分的情緒指標分數，並在最後在依照以下格式輸出整體的情緒分數[今日的情緒分數為:(score)]，以下為範例，範例一:文章極度看好未來市場+5 留言70%支持文章 最終情緒分數為(5*70%)=3.5，範例二:文章稍微不看好市場-2 留言30%支持文章 最終情緒分數為(-2*30%)=-0.6。"},
                {"role": "user", "content": prompt_text}]

    try:
        outputs = pipeline(
            messages,
            max_new_tokens=4096,
        )
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"程式碼執行時間：{elapsed_time} 秒")
        return outputs[0]["generated_text"][-1]
    except Exception as e:
        print(f"錯誤（{date.strftime('%Y-%m-%d')}）：{e}")
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"程式碼執行時間：{elapsed_time} 秒")
        return "分析失敗"

# 依照日期處理文章
for date, group in grouped_by_date:
    date_str = date.strftime('%Y-%m-%d')

    # 跳過已處理的日期
    if date_str in processed_dates:
        print(f"跳過 {date_str}，因為已經分析過。")
        continue

    articles = group[['content', 'messages']].dropna().values.tolist()  # 取得當日所有文章內容
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