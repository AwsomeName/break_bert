import json
import requests

# 阿里云API配置
API_KEY = "sk-9945083bf30946c9b5fa32a899305c14"
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

def test_api():
    """测试API连接"""
    print("测试API连接...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    prompt = "你好，请回复：API测试成功"
    
    data = {
        "model": "ep-20260306143243-dsfcx",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=30)
        print(f"API状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"API响应: {result}")
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"返回内容: {content}")
                return True
        else:
            print(f"API错误: {response.text}")
    except Exception as e:
        print(f"API调用失败: {e}")
    
    return False

if __name__ == "__main__":
    test_api()
