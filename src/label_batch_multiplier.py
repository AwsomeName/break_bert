import json
import os
import time
import dashscope
from dashscope import Generation
import re

key = os.environ.get("DASHSCOPE_API_KEY")
if not key:
    raise RuntimeError("DASHSCOPE_API_KEY is not set")
dashscope.api_key = key

def judge_semantic_completeness_batch(texts):
    """
    批量打标，减少 API 调用次数
    """
    prompt = "请判断以下对话片段在截断处语义是否已经完整。\n\n"
    for i, text in enumerate(texts):
        prompt += f"【片段 {i+1}】：{text}\n"
    
    prompt += """
要求：
1. 仅输出每段对话的判断结果（“完整”或“不完整”），结果以 JSON 列表形式输出，例如：["不完整", "完整", "不完整"]
2. 判断标准：如果截断处是一个完整的句子且意思表达清楚，则为“完整”；如果截断在句子中间或意思未表达完，则为“不完整”。
3. 务必按顺序输出结果列表，且列表长度必须与输入片段数一致。
4. 只输出 JSON 数组，不要任何额外说明。
"""
    
    try:
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content.strip()
            # 提取 JSON 数组
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                results = json.loads(match.group(0))
                # 转换为 0 (不完整) 或 1 (完整)
                labels = [1 if str(r).strip() == "完整" else 0 for r in results]
                return labels
        return None
    except Exception as e:
        print(f"批量打标出错: {e}")
        return None

def main():
    input_file = "data/processed/truncated_samples.json"
    output_file = "data/processed/labeled_samples.json"
    
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # 初始化/加载已打标数据
    labeled_samples = []
    if os.path.exists(output_file):
        # 备份并强制重新开始，因为原始数据已经根据短片段重写了
        print("发现旧的打标文件，将备份并重新打标以匹配新的短片段数据...")
        os.rename(output_file, output_file + ".bak")

    # 确定待打标样本
    remaining_samples = samples
    
    print(f"总样本: {len(samples)}，开始批量打标...")
    
    batch_size = 5 # 批量打标大小
    
    for i in range(0, len(remaining_samples), batch_size):
        batch = remaining_samples[i : i + batch_size]
        texts = [s['truncated_text'] for s in batch]
        
        print(f"进度: {len(labeled_samples)}/{len(samples)}...", end="\r")
        labels = judge_semantic_completeness_batch(texts)
        
        if labels and len(labels) == len(batch):
            for sample, label in zip(batch, labels):
                sample['label'] = label
                labeled_samples.append(sample)
            
            # 每 100 条保存一次
            if len(labeled_samples) % 100 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(labeled_samples, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.3) # 避免限流
        else:
            print(f"\n批量打标失败，重试中 (索引: {i})...")
            time.sleep(1)
            # 重试一次
            labels = judge_semantic_completeness_batch(texts)
            if labels and len(labels) == len(batch):
                for sample, label in zip(batch, labels):
                    sample['label'] = label
                    labeled_samples.append(sample)
            else:
                print(f"跳过本批次: {i}")

    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n打标完成！总计 {len(labeled_samples)} 条数据保存至 {output_file}")

if __name__ == "__main__":
    main()
