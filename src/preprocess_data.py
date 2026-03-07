import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 数据预处理
def preprocess_data(data):
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 清洗文本
    df["dialogue"] = df["dialogue"].apply(lambda x: x.strip())
    
    # 去除空值
    df = df.dropna()
    
    return df

# 划分数据集
def split_data(df, test_size=0.2, val_size=0.1):
    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # 从训练集中划分验证集
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=42)
    
    return train_df, val_df, test_df

# 保存数据集
def save_data(train_df, val_df, test_df):
    train_df.to_json("/home/lc/bert-broken/data/train_data.json", orient="records", force_ascii=False, indent=2)
    val_df.to_json("/home/lc/bert-broken/data/val_data.json", orient="records", force_ascii=False, indent=2)
    test_df.to_json("/home/lc/bert-broken/data/test_data.json", orient="records", force_ascii=False, indent=2)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")

if __name__ == "__main__":
    # 加载示例数据
    data = load_data("/home/lc/bert-broken/data/sample_data.json")
    
    # 预处理数据
    df = preprocess_data(data)
    
    # 划分数据集
    train_df, val_df, test_df = split_data(df)
    
    # 保存数据集
    save_data(train_df, val_df, test_df)
