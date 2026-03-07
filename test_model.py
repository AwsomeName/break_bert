import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import joblib

# 加载模型和向量化器
model = joblib.load("/home/lc/bert-broken/models/model.pkl")
vectorizer = joblib.load("/home/lc/bert-broken/models/vectorizer.pkl")

# 测试文本
test_text = "用户：好的，就这些，谢谢。"

# 转换文本
X = vectorizer.transform([test_text])

# 预测
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0]

print(f"预测结果: {pred}")
print(f"对话是否结束: {bool(pred)}")
print(f"置信度: {proba[pred]:.4f}")
