import json
import os

def main():
    input_file = "data/raw/samples.json"
    output_file = "data/raw_segments_preview.txt"
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== 总计 {len(samples)} 条 100-300 字短片段语料预览 ===\n\n")
        for i, sample in enumerate(samples):
            # 兼容列表嵌套或直接字符串
            content = sample[0] if isinstance(sample, list) else sample
            f.write(f"--- 样本 {i+1} (长度: {len(content)}) ---\n")
            f.write(content)
            f.write("\n\n" + "="*50 + "\n\n")
    
    print(f"成功导出预览文件至 {output_file}")

if __name__ == "__main__":
    main()
