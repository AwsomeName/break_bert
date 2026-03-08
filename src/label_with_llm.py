import json
import dashscope
from dashscope import Generation
import os

key = os.environ.get("DASHSCOPE_API_KEY")
if not key:
    raise RuntimeError("DASHSCOPE_API_KEY is not set")
dashscope.api_key = key

def judge_semantic_completeness(truncated_text):
    prompt = f"""
请判断以下对话片段在截断处语义是否已经完整。
截断后的对话内容：
---
{truncated_text}
---
请仅回答“完整”或“不完整”。
判断标准：如果截断处是一个完整的句子且意思表达清楚，则为“完整”；如果截断在句子中间或意思未表达完，则为“不完整”。
"""
    
    try:
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            result = response.output.choices[0].message.content.strip()
            # 转换为 0 (不完整) 或 1 (完整)
            if "不完整" in result:
                return 0
            elif "完整" in result:
                return 1
            else:
                return 0
        else:
            return 0
    except Exception:
        return 0

def main():
    # 读取截断后的样本
    input_file = "data/processed/truncated_samples.json"
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    labeled_samples = []
    print("开始调用阿里云大模型进行打标...")
    
    for sample in samples:
        print(f"正在处理样本 ID: {sample['id']}...")
        label = judge_semantic_completeness(sample['truncated_text'])
        sample['label'] = label
        labeled_samples.append(sample)
    
    # 保存打标后的样本
    output_file = "data/processed/labeled_samples.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled_samples, f, ensure_ascii=False, indent=2)
    
    print(f"成功保存打标后的样本到 {output_file}")

if __name__ == "__main__":
    main()
