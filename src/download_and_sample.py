from datasets import load_dataset
import json
import random

def main():
    print("正在加载 m-a-p/COIG-CQIA 数据集...")
    try:
        # 加载 zhihu 子集
        dataset = load_dataset("m-a-p/COIG-CQIA", name="zhihu", split="train")
        print("COIG-CQIA zhihu 数据集加载成功")
    except Exception as e:
        print(f"COIG-CQIA 加载失败: {e}, 尝试加载整个数据集...")
        dataset = load_dataset("m-a-p/COIG-CQIA", split="train")
        print("COIG-CQIA 数据集加载成功")

    # 采样10条
    print("正在采样10条对话...")
    # 随机采样 10 条
    sampled_indices = random.sample(range(len(dataset)), 10)
    samples = []
    
    for idx in sampled_indices:
        item = dataset[idx]
        # COIG-CQIA 格式通常是 {"instruction": ..., "input": ..., "output": ...}
        # 如果是对话，通常是在 instruction 和 output 中
        # 我们把它们组合成一个对话列表
        dialogue = []
        if item.get("instruction"):
            dialogue.append(item["instruction"])
        if item.get("input"):
            dialogue.append(item["input"])
        if item.get("output"):
            dialogue.append(item["output"])
            
        samples.append(dialogue)

    # 保存采样数据
    import os
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存 10 条样本到 data/raw/samples.json")

if __name__ == "__main__":
    main()
