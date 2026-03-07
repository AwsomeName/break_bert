import json
import random
import os
import time
import dashscope
from dashscope import Generation
import re

# 设置 API Key
dashscope.api_key = "sk-9945083bf30946c9b5fa32a899305c14"

def generate_single_dialogue(scenario, p1, p2):
    """
    生成单条对话，确保格式稳定
    """
    prompt = f"""
请根据以下场景和人物性格，生成一段高质量、真实、口语化的中文对话。

【场景】：{scenario}
角色 A：{p1['gender']}，{p1['age']}，来自{p1['region']}，职业是{p1['occupation']}，习惯：{p1['habit']}
角色 B：{p2['gender']}，{p2['age']}，来自{p2['region']}，职业是{p2['occupation']}，习惯：{p2['habit']}

要求：
1. 对话包含 4-8 轮交流。
2. 严格遵循人物习惯，对话要自然、口语化，像真实生活中的聊天。
3. 输出格式必须是 JSON 字符串列表，例如：["内容1", "内容2", "内容3"]
4. 不要包含角色名称前缀（如 A: 或 B:）。
5. 只输出 JSON 数组，不要任何解释说明。
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
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        return data
                except:
                    pass
        return None
    except Exception:
        return None

def main():
    if not os.path.exists("data/scenarios.txt") or not os.path.exists("data/personas.json"):
        print("场景或人物文件不存在！")
        return
        
    with open("data/scenarios.txt", "r", encoding="utf-8") as f:
        scenarios = [line.strip() for line in f if line.strip()]
    
    with open("data/personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)
    
    total_target = 1000
    output_file = "data/raw/samples.json"
    
    # 初始化/加载现有数据
    all_dialogues = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                all_dialogues = json.load(f)
                print(f"已加载现有 {len(all_dialogues)} 条对话。")
            except:
                all_dialogues = []

    print(f"开始生成 {total_target} 条对话语料...")
    
    # 连续失败计数
    fail_count = 0
    
    while len(all_dialogues) < total_target:
        scenario = random.choice(scenarios)
        p1, p2 = random.sample(personas, 2)
        
        print(f"进度: {len(all_dialogues)}/{total_target}...", end="\r")
        dialogue = generate_single_dialogue(scenario, p1, p2)
        
        if dialogue:
            all_dialogues.append(dialogue)
            fail_count = 0
            # 每 5 条保存一次，减少 IO
            if len(all_dialogues) % 5 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
        else:
            fail_count += 1
            if fail_count > 5:
                print("\n连续生成失败，暂停 2 秒...")
                time.sleep(2)
                fail_count = 0
            
    # 最后保存一次
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
            
    print(f"\n成功完成 1000 条对话合成，保存至 {output_file}")

if __name__ == "__main__":
    main()
