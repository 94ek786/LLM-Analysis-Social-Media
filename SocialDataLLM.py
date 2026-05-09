import time
import pandas as pd
import base64
import os
from google import genai
from google.genai import types
import re

def extract_numbers(text):#取數字
    result = []
    current_num = ""
    for char in text:
        if char.isdigit() or char in ['.', '-']:  # 如果是數字或小數點、負號
            current_num += char
        elif current_num:  # 如果當前有累積的數字，並且遇到非數字字符，將其轉換為浮點數
            result.append(float(current_num))
            current_num = ""
    
    # 處理最後一個數字
    if current_num:
        result.append(float(current_num))
    
    return result[0]

def analyze_articles(date, articles):
    client = genai.Client(
        api_key=("AIzaSyA40AmOo2uZurFwIgXOyhTPfNA7FUTMyLE")
    )
    model = "gemini-2.0-flash"
    start_time = time.perf_counter()
    
    prompt_text = f"請針對以下 {len(articles)} 篇文章進行分析並給予這些文章整合的情緒分數：\n\n"
    for idx, (content, messages) in enumerate(articles):
        #
        prompt_text += f"第 {idx+1} 篇文章:\n內容: {content}\n留言(|符號隔開的代表不同人的留言): {messages}\n\n"
    
    prompt_text += "請依照以下格式[今日的情緒分數為:(score)]，將這些文章給出一個整合的情緒指標分數"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        ),
    ]
    #你是一個金融市場情緒分析師，你的任務是分析每天的文章，並給出-5至+5分的情緒指標分數，以下為範例，範例一:文章極度看好未來市場+5 留言70%支持文章 最終情緒分數為(5*70%)=3.5，範例二:文章稍微不看好市場-2 留言30%支持文章 最終情緒分數為(-2*30%)=-0.6。在分析的最後依照以下格式輸出平均的情緒分數[今日的平均情緒分數為:(score)]
    #你是一個金融市場情緒分析師，你的任務是分析每天的文章，並且給出正面1、負面-1、中性0，以下為範例，範例一:文章多認為未來市場看好，且半數留言認同文章，輸出[今日的情緒分數為:1]。範例2:各個文章有不同的觀點，而留言有大量相反的意見，輸出[今日的情緒分數為:0]。在分析的最後依照以下格式輸出今日的情緒分數[今日的情緒分數為:(score)]
    #"你是一個金融市場情緒分析師，你的任務是分析每天的文章，並且給出正面1、負面-1、中性0，以下為範例。範例一:一天中有三篇文章看好未來股市，一篇文章不看好，一篇文章只是分享數據沒有特別偏好，3篇正面大於1篇負面，輸出[今日的情緒分數為:1]。範例2:兩篇文章看好股市，一篇不看好，2篇正面僅比1篇負面多一篇，輸出[今日的情緒分數為:0]。並在分析的最後依照以下格式輸出今日的情緒分數[今日的情緒分數為:(score)]"
    system_instruction="你是一個金融市場情緒分析師，你的任務是分析每天的文章，並且給出正面1、負面-1、中性0，以下為範例，範例一:文章多認為未來市場看好，且半數留言認同文章，輸出[今日的情緒分數為:1]。範例2:各個文章有不同的觀點，而留言有大量相反的意見，輸出[今日的情緒分數為:0]。在分析的最後依照以下格式輸出今日的情緒分數[今日的情緒分數為:(score)]"
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=system_instruction),
        ],
    )

    loop = " "
    for chunk in client.models.generate_content_stream(
    model=model,
    contents=contents,
    config=generate_content_config,
    ):
        loop += chunk.text
    end_time = time.perf_counter()
    print(f"程式碼執行時間：{end_time - start_time} 秒")
    #print(loop)
    return loop

# 讀取 CSV 檔案
read_class = '2603'
file_paths = ["filtered_" + read_class + ".csv"]
dfs = []

for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df['category'] = file.split('_')[1][0]
        dfs.append(df)

# 合併所有分類的資料
df_all = pd.concat(dfs, ignore_index=True)

df_all['date'] = pd.to_datetime(df_all['date'])

grouped_by_date = df_all.groupby('date')

loop_id = 0
while(loop_id == 0):
    loop_id = 1
    # 嘗試讀取已存檔的分析結果
    analysis_file = "analysis_ternary" + read_class + ".csv"
    if os.path.exists(analysis_file):
        df_existing = pd.read_csv(analysis_file)
        processed_dates = set(df_existing['date'])
        analysis_results = df_existing.to_dict('records')
    else:
        processed_dates = set()
        analysis_results = []

    # 依照日期處理文章
    for date, group in grouped_by_date:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in processed_dates:
            print(f"跳過 {date_str}，因為已經分析過。", end='\r')
            continue
        
        articles = group[['content', 'messages']].dropna().values.tolist()
        if not articles:
            continue
        
        print(f"分析 {date_str} 的文章...")
        looop = 0
        while(looop<5):
            try:
                analysis = analyze_articles(date, articles)
                looop = 5
            except Exception as e:
                print(f"錯誤（{date.strftime('%Y-%m-%d')}）：{e}")
                looop = looop + 1
        try:
            numbers = extract_numbers(analysis.split("情緒分數為:")[1])
            if numbers == 1:
                pass
            elif numbers == 0:
                pass
            elif numbers == -1:
                pass
            else:
                print('錯誤數字非1,0,-1')
                loop_id = 0
                break
            """
            if numbers > 5:
                print('錯誤數字過大')
                loop_id = 0
                break
            elif numbers < -5:
                print('錯誤數字過小')
                loop_id = 0
                break
            """
        except Exception as e:
            print(f"錯誤（{date.strftime('%Y-%m-%d')}）：{e}")
            loop_id = 0
            break

        analysis_results.append({"date": date_str, "analysis": numbers})
        pd.DataFrame(analysis_results).to_csv(analysis_file, index=False, encoding='utf-8-sig')
        print(f"已儲存 {date_str} 的分析結果到 {analysis_file}。")
        time.sleep(3)  # 避免 API 請求過於頻繁

print("每日文章分析完成，已儲存至 " + analysis_file)
