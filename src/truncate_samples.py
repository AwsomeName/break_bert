import json
import random
import os

def truncate_dialogue(dialogue):
    """
    实现严格随机字符位置截断。
    """
    # 将对话列表合并为完整文本
    full_text = "\n".join(dialogue)
    if not full_text:
        return "", ""
    
    length = len(full_text)
    
    # 在 10% 到 90% 的长度范围内，严格随机选择一个字符索引进行截断
    if length < 5:
        truncate_point = length // 2
    else:
        start_min = int(length * 0.1)
        end_max = int(length * 0.9)
        truncate_point = random.randint(start_min, end_max)
    
    truncated_text = full_text[:truncate_point]
    return truncated_text, full_text

def main():
    # 读取原始采样样本
    input_file = "data/raw/samples.json"
    if not os.path.exists(input_file):
        print(f"原始采样数据不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    processed_samples = []
    for i, dialogue in enumerate(samples):
        truncated, full = truncate_dialogue(dialogue)
        processed_samples.append({
            "id": i,
            "truncated_text": truncated,
            "full_text": full
        })
    
    # 保存处理后的样本
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/truncated_samples.json", "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)
    
    print(f"成功实现严格随机截断，并保存到 data/processed/truncated_samples.json")

if __name__ == "__main__":
    main()
