import json
import random
import os

def main():
    print("正在从数据集筛选对话感强的样本...")
    
    from datasets import load_dataset
    dataset = load_dataset("m-a-p/COIG-CQIA", name="zhihu", split="train")

    filtered_data = []
    for item in dataset:
        instr = item.get("instruction", "")
        out = item.get("output", "")
        
        # 更加宽松的筛选条件：
        # 1. instruction 包含问句特征
        # 2. output 长度在合理范围内
        # 3. 总长度在 600 字以内
        full_text = instr + "\n" + out
        length = len(full_text)
        
        is_question = any(q in instr for q in ["？", "?", "如何", "为什么", "怎么", "什么", "吗"])
        
        if is_question and 10 < len(instr) < 150 and 20 < len(out) < 400 and length <= 600:
            filtered_data.append({
                "dialogue": [instr, out],
                "length": length
            })
            
    print(f"筛选出符合对话特征的样本数: {len(filtered_data)}")

    if not filtered_data:
        # 如果还是没有，就直接取前1000个样本中较短的
        print("警告: 未筛选出特定对话样本，将从较短样本中随机抽取。")
        for i in range(min(1000, len(dataset))):
            item = dataset[i]
            dialogue = [item.get("instruction", ""), item.get("output", "")]
            length = len("\n".join(dialogue))
            if 0 < length <= 600:
                filtered_data.append({"dialogue": dialogue, "length": length})

    # 均匀采样 10 条
    num_samples = 10
    filtered_data.sort(key=lambda x: x["length"])
    
    # 确保采样覆盖不同长度
    sampled_data = []
    if len(filtered_data) >= num_samples:
        indices = [int(i * (len(filtered_data)-1) / (num_samples-1)) for i in range(num_samples)]
        for idx in indices:
            sampled_data.append(filtered_data[idx]["dialogue"])
    else:
        sampled_data = [d["dialogue"] for d in filtered_data]

    # 保存
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/samples.json", "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存 {len(sampled_data)} 条对话感强的样本到 data/raw/samples.json")
    sampled_lengths = [len("\n".join(d)) for d in sampled_data]
    print(f"采样长度分布: {sampled_lengths}")

if __name__ == "__main__":
    main()
