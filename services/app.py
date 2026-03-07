from fastapi import FastAPI, HTTPException
import uvicorn
import json
import torch
import sys
import os
from transformers import BertTokenizer, BertForSequenceClassification

# 添加src目录到路径，确保能加载tokenizer模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("/home/lc/bert-broken/models/bert_tokenizer")
model = BertForSequenceClassification.from_pretrained("/home/lc/bert-broken/models/bert_model")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 创建FastAPI应用
app = FastAPI(
    title="语义中断检测服务",
    description="基于BERT模型的语义中断检测API",
    version="2.0.0"
)

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 语义中断检测端点
@app.post("/detect")
def detect_interruption(dialogue: str):
    try:
        # 转换文本
        encoding = tokenizer(
            dialogue,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # 解析结果
        result = {
            "dialogue": dialogue,
            "is_interrupted": bool(prediction),
            "confidence": confidence
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 批量检测端点
@app.post("/batch_detect")
def batch_detect_interruption(dialogues: list):
    try:
        # 转换文本
        encodings = tokenizer(
            dialogues,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        
        # 解析结果
        results = []
        for i, dialogue in enumerate(dialogues):
            prediction = predictions[i]
            confidence = float(probabilities[i][prediction])
            results.append({
                "dialogue": dialogue,
                "is_interrupted": bool(prediction),
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
