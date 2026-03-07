import onnxruntime as ort
from transformers import BertTokenizer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# 初始化 FastAPI
app = FastAPI(title="BERT Semantic Break Detection API")

# 模型路径
MODEL_PATH = "models/bert_model/model.onnx"
TOKENIZER_PATH = "models/bert_model"

# 全局加载推理引擎和分词器
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX 模型文件不存在: {MODEL_PATH}")

print(f"正在加载 ONNX 模型: {MODEL_PATH}...")
# 强制使用 CPU
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    text: str
    status: str
    label: int
    confidence: float

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # 1. 预处理 (Tokenization)
    inputs = tokenizer(
        request.text,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    
    # 2. 推理
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    try:
        ort_outputs = session.run(None, ort_inputs)
        logits = ort_outputs[0]
        
        # 3. 后处理 (Softmax & Argmax)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_label = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0][pred_label])
        
        status = "完整" if pred_label == 1 else "不完整"
        
        return InferenceResponse(
            text=request.text,
            status=status,
            label=pred_label,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "provider": session.get_providers()[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
