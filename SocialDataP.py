import pandas as pd
import json
import os

# 設定 JSON 檔案資料夾路徑
folder_path = 'ptt_stock_data/'  # 修改為您的資料夾路徑
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 讀取所有 JSON 檔案並合併
data_frames = []

for file in json_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.json_normalize(data, 'articles')  # 將 'articles' 欄位展開
        data_frames.append(df)

# 合併所有 DataFrame
df_all = pd.concat(data_frames, ignore_index=True)

# 確保 'article_title' 欄位不為 NaN
df_all['article_title'] = df_all['article_title'].fillna('')

# 文章分類函式
def categorize_article(title):
    C_keywords = ["中華電", "CHT", "2412"]
    A_keywords = ["長榮", "2603"]
    B_keywords = ["ASUS", "2357", "華碩"]
    
    if any(keyword in title for keyword in C_keywords):
        return '2412'
    elif any(keyword in title for keyword in A_keywords):
        return '2603'
    elif any(keyword in title for keyword in B_keywords):
        return '2357'
    return None

# 套用分類並過濾未分類的資料
df_all['category'] = df_all['article_title'].apply(categorize_article)
df_all = df_all[df_all['category'].notna()]

# 轉換日期格式
df_all['date'] = pd.to_datetime(df_all['date'], format='%a %b %d %H:%M:%S %Y').dt.strftime('%Y-%m-%d')

# 取得 message_count.all
df_all['message_count'] = df_all['message_count.all']

# 提取 messages 中的 push_content，使用 ' | ' 分隔
df_all['messages'] = df_all['messages'].apply(lambda x: ' | '.join([msg['push_content'] for msg in x if msg['push_content']]))

# 選擇需要的欄位
df_filtered = df_all[['article_title', 'date', 'message_count', 'messages', 'content', 'category']]

# 分類並存成不同檔案
for category in ['2412', '2603', '2357']:
    df_category = df_filtered[df_filtered['category'] == category].drop(columns=['category'])
    file_name = f'filtered_{category}.csv'
    df_category.to_csv(file_name, index=False, encoding='utf-8')
    print(f"已輸出 {file_name}")

print("所有分類資料處理完成！")
