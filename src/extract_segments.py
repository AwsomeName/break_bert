import json
import random
import os

def extract_random_segment(dialogue):
    """
    从对话中随机截取 100-300 字的一段
    """
    full_text = "\n".join(dialogue)
    length = len(full_text)
    
    if length <= 100:
        return full_text
    
    # 随机选择目标长度 100-300
    target_len = random.randint(100, min(300, length))
    
    # 随机选择起点
    start_pos = random.randint(0, length - target_len)
    
    segment = full_text[start_pos : start_pos + target_len]
    return segment

def main():
    input_file = "data/raw/samples.json"
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    print(f"读取到 {len(samples)} 条对话。正在提取 100-300 字片段...")
    
    new_samples = []
    for dialogue in samples:
        # 将原对话列表转为单段短文本
        segment = extract_random_segment(dialogue)
        # 这里包装成列表形式以兼容之前的脚本
        new_samples.append([segment])
    
    # 保存为新的原始数据
    output_file = "data/raw/samples.json" # 直接覆盖，因为它是后续处理的输入源
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_samples, f, ensure_ascii=False, indent=2)
    
    print(f"成功将 1000 条对话片段化（100-300字），已更新 {output_file}")

if __name__ == "__main__":
    main()
