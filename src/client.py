import requests
import json
import time

# 服务器地址
SERVER_URL = "http://localhost:8000/predict"

def test_inference(text):
    payload = {"text": text}
    try:
        start_time = time.time()
        # 禁用代理访问本地服务
        response = requests.post(SERVER_URL, json=payload, proxies={"http": None, "https": None})
        latency = (time.time() - start_time) * 1000 # 毫秒
        
        if response.status_code == 200:
            result = response.json()
            print(f"输入文本: {result['text']}")
            print(f"检测状态: {result['status']} (置信度: {result['confidence']:.2%})")
            print(f"推理时延: {latency:.2f} ms")
            print("-" * 50)
        else:
            print(f"请求失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # 测试集
    test_cases = [
        "您好，我想请问一下那个",
        "那就这么定了，我跟孩子们说说。",
        "医生，我的牙是不是得拔？",
        "那个，老师，我英语听力和口语都不太",
        "末班车是晚上九点，现在还有三趟。"
    ]
    
    print("=== BERT Semantic Break Detection Client 测试 ===\n")
    for text in test_cases:
        test_inference(text)
