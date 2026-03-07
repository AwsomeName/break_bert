from modelscope.msdatasets import MsDataset
import json
import random
import os

def main():
    print("正在尝试从 ModelScope 加载高质量中文对话数据集...")
    
    # 尝试多个可能的 ID
    datasets_to_try = [
        ('LCCC-base', 'thu-coai'),
        ('LCCC-base', 'modelscope'),
        ('CrossWOZ', 'thu-coai'),
        ('CrossWOZ', 'modelscope'),
        ('NaturalConv', 'modelscope'),
        ('JDDC', 'modelscope')
    ]
    
    ds = None
    for name, ns in datasets_to_try:
        try:
            print(f"尝试加载 {ns}/{name}...")
            # 某些数据集需要指定完整的 namespace/name
            ds = MsDataset.load(f'{ns}/{name}', split='test')
            print(f"{ns}/{name} 加载成功")
            break
        except Exception as e:
            # print(f"{ns}/{name} 加载失败: {e}")
            try:
                # 尝试不带 namespace
                ds = MsDataset.load(name, split='test')
                print(f"{name} 加载成功")
                break
            except:
                continue

    if ds is None:
        print("所有 ModelScope 数据集尝试加载失败。")
        return

    # 1. 过滤真正属于对话且在 600 字以内的样本
    print("正在过滤 600 字以内的真实对话样本...")
    filtered_data = []
    
    for item in ds:
        # 兼容不同数据集的格式
        dialogue = None
        if 'dialog' in item:
            dialogue = item['dialog']
        elif 'content' in item:
            dialogue = item['content']
        elif 'messages' in item:
            dialogue = [m['content'] for m in item['messages']]
        elif 'conversation' in item:
            dialogue = item['conversation']
            
        if not isinstance(dialogue, list) or len(dialogue) <= 1:
            continue
            
        full_text = "\n".join(dialogue)
        length = len(full_text)
        
        if 0 < length <= 600:
            filtered_data.append({
                "dialogue": dialogue,
                "length": length
            })
    
    print(f"符合条件的对话样本数: {len(filtered_data)}")

    if not filtered_data:
        print("未筛选出符合条件的对话样本。")
        return

    # 2. 均匀采样 10 条
    print("正在按照均匀长度分布采样 10 条真实对话...")
    num_samples = 10
    filtered_data.sort(key=lambda x: x["length"])
    
    sampled_data = []
    indices = [int(i * (len(filtered_data)-1) / (num_samples-1)) for i in range(num_samples)]
    for idx in indices:
        sampled_data.append(filtered_data[idx]["dialogue"])

    # 3. 保存
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/samples.json", "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    with open("data/raw/filtered_dataset.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data[:5000], f, ensure_ascii=False, indent=2)

    print(f"成功保存 {len(sampled_data)} 条真实对话样本到 data/raw/samples.json")
    sampled_lengths = [len("\n".join(d)) for d in sampled_data]
    print(f"采样长度分布: {sampled_lengths}")

if __name__ == "__main__":
    main()
