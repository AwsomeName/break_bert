from datasets import load_dataset
import json
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
    
    # 2. 保存完整过滤后的数据集
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/filtered_dataset.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"符合条件的样本数: {len(filtered_data)}")
    print(f"完整过滤后的数据集已保存到 data/raw/filtered_dataset.json")

if __name__ == "__main__":
    main()
