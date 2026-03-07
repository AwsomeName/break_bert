import json
import random
import os

def truncate_text(text):
    """
    严格随机字符位置截断
    """
    length = len(text)
    if length <= 5:
        return text[:length//2], text
    
    # 在 10% 到 90% 的长度范围内随机选择截断点
    start_min = int(length * 0.1)
    end_max = int(length * 0.9)
    truncate_point = random.randint(start_min, end_max)
    
    truncated_text = text[:truncate_point]
    return truncated_text, text

def main():
    input_file = "data/raw/samples.json"
    output_file = "data/processed/truncated_samples.json"
    
    if not os.path.exists(input_file):
        print(f"原始数据文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    multiplier = 5
    processed_samples = []
    
    print(f"读取到 {len(samples)} 条原始短语料。正在进行 {multiplier} 倍采样随机截断...")
    
    for i, dialogue in enumerate(samples):
        # 兼容列表格式 [content]
        text = dialogue[0] if isinstance(dialogue, list) else dialogue
        for j in range(multiplier):
            truncated, full = truncate_text(text)
            processed_samples.append({
                "id": f"{i}_{j}",
                "original_id": i,
                "truncated_text": truncated,
                "full_text": full
            })
            
    os.makedirs("data/processed", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成 {len(processed_samples)} 条截断数据并保存至 {output_file}")

if __name__ == "__main__":
    main()
