import json
import random
import os

def generate_personas(count=60):
    genders = ["男", "女"]
    ages = ["18-25岁", "26-35岁", "36-45岁", "46-60岁", "60岁以上"]
    regions = ["北京", "上海", "广东", "四川", "东北", "山东", "江苏", "浙江", "湖南", "湖北", "河南", "陕西"]
    occupations = ["学生", "程序员", "医生", "老师", "销售", "导游", "客服", "工人", "退休人员", "家庭主妇", "自媒体人", "厨师", "司机", "保安", "白领"]
    
    habits = [
        "说话喜欢带语气词，如‘嗯’、‘那个’、‘就是说’。",
        "说话非常简练，直奔主题。",
        "喜欢用成语或诗句，说话比较文绉绉。",
        "性格急躁，说话语速快，容易打断别人。",
        "性格温和，说话慢条斯理，喜欢用‘您’。",
        "说话喜欢带点方言口音，亲切自然。",
        "逻辑性极强，喜欢分点说明（第一、第二）。",
        "容易纠结，说话吞吞吐吐，常用‘可能’、‘大概’。",
        "喜欢开玩笑，说话幽默风趣。",
        "非常客气，经常说‘谢谢’、‘麻烦了’、‘不好意思’。"
    ]
    
    personas = []
    for i in range(count):
        persona = {
            "id": i + 1,
            "gender": random.choice(genders),
            "age": random.choice(ages),
            "region": random.choice(regions),
            "occupation": random.choice(occupations),
            "habit": random.choice(habits)
        }
        personas.append(persona)
    
    return personas

def main():
    personas = generate_personas(60)
    
    os.makedirs("data", exist_ok=True)
    with open("data/personas.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成 60 个人物性格档案并保存到 data/personas.json")

if __name__ == "__main__":
    main()
