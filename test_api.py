import json
import requests
import os

API_KEY = os.environ.get("ARK_API_KEY") or ""
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

def test_api():
    """测试API是否正常工作"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": "ep-20260306143243-dsfcx",
        "messages": [
            {"role": "user", "content": "请回复数字1"}
        ],
        "temperature": 0.3
    }
    
    try:
        print("正在测试API...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"解析结果: {result}")
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                print(f"内容: {content}")
                return True
    except Exception as e:
        print(f"API测试失败: {e}")
    
    return False

if __name__ == "__main__":
    test_api()
