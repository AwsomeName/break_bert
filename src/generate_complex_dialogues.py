import json
import random
import os
import dashscope
from dashscope import Generation

# 设置 API Key
dashscope.api_key = "sk-9945083bf30946c9b5fa32a899305c14"

def generate_dialogue(scenario, p1, p2):
    prompt = f"""
请根据以下场景和人物性格，生成一段高质量、真实、口语化的中文对话。

场景：{scenario}

人物 1 (角色 A)：
- 性别：{p1['gender']}
- 年龄：{p1['age']}
- 地域：{p1['region']}
- 职业：{p1['occupation']}
- 发言习惯：{p1['habit']}

人物 2 (角色 B)：
- 性别：{p2['gender']}
- 年龄：{p2['age']}
- 地域：{p2['region']}
- 职业：{p2['occupation']}
- 发言习惯：{p2['habit']}

要求：
1. 对话要非常自然，像真实生活中的聊天或服务场景。
2. 严格遵循人物的发言习惯。
3. 对话轮数在 4-6 轮左右。
4. 输出格式为 JSON 列表，例如：["内容1", "内容2", ...]
5. 不要包含人物名称前缀，直接输出对话内容字符串列表。
"""
    
    try:
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content.strip()
            # 简单清洗，防止 LLM 输出非 JSON 字符
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        else:
            print(f"API 失败: {response.message}")
            return None
    except Exception as e:
        print(f"生成出错: {e}")
        return None

def main():
    # 1. 加载场景
    with open("data/scenarios.txt", "r", encoding="utf-8") as f:
        scenarios = [line.strip() for line in f if line.strip()]
    
    # 2. 加载人物
    with open("data/personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)
    
    # 3. 随机选择 10 个场景并生成
    selected_scenarios = random.sample(scenarios, 10)
    all_dialogues = []
    
    print("开始基于场景和性格合成对话...")
    for i, scenario in enumerate(selected_scenarios):
        p1, p2 = random.sample(personas, 2)
        print(f"正在生成第 {i+1}/10 条对话: {scenario[:30]}...")
        dialogue = generate_dialogue(scenario, p1, p2)
        if dialogue:
            all_dialogues.append(dialogue)
    
    # 4. 保存
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/samples.json", "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
    
    print(f"成功合成 10 条高质量对话并保存到 data/raw/samples.json")

if __name__ == "__main__":
    main()
