from datasets import load_dataset
import json
import random
import os

def main():
    print("正在加载 m-a-p/COIG-CQIA 数据集...")
    dataset = load_dataset("m-a-p/COIG-CQIA", name="zhihu", split="train")
    
    # 1. 过滤 600 字以内的对话
    print("正在过滤 600 字以内的样本...")
    filtered_data = []
    for item in dataset:
        dialogue = []
        if item.get("instruction"):
            dialogue.append(item["instruction"])
        if item.get("input"):
            dialogue.append(item["input"])
        if item.get("output"):
            dialogue.append(item["output"])
        
        full_text = "\n".join(dialogue)
        length = len(full_text)
        
        if 0 < length <= 600:
            filtered_data.append({
                "dialogue": dialogue,
                "length": length
            })
    
    print(f"符合条件的样本数: {len(filtered_data)}")

    # 2. 按照长度分布均匀采样 10 条
    # 将长度范围 (0, 600] 分成 10 个区间
    print("正在按照均匀长度分布采样 10 条...")
    num_samples = 10
    step = 600 / num_samples
    sampled_data = []
    
    for i in range(num_samples):
        min_len = i * step
        max_len = (i + 1) * step
        
        # 寻找落在此区间的样本
        candidates = [d for d in filtered_data if min_len < d["length"] <= max_len]
        
        if candidates:
            sampled_data.append(random.choice(candidates)["dialogue"])
        else:
            # 如果区间内没样本，从全局随机选一个
            print(f"警告: 区间 {min_len}-{max_len} 没有样本，随机选择一个。")
            sampled_data.append(random.choice(filtered_data)["dialogue"])

    # 3. 保存采样数据
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/samples.json", "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    # 统计采样后的长度
    sampled_lengths = [len("\n".join(d)) for d in sampled_data]
    print(f"成功保存 10 条样本到 data/raw/samples.json")
    print(f"采样长度分布: {sorted(sampled_lengths)}")

if __name__ == "__main__":
    main()
