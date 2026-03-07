import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

def export_to_onnx():
    model_path = "models/bert_model"
    onnx_output_path = "models/bert_model/model.onnx"
    
    if not os.path.exists(model_path):
        print(f"模型目录不存在: {model_path}")
        return

    # 加载模型和分词器
    print(f"正在从 {model_path} 加载模型...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # 准备虚拟输入
    dummy_text = "您好，我想请问一下那个"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    
    # 导出 ONNX
    print(f"正在导出 ONNX 模型到 {onnx_output_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True
    )
    
    print("ONNX 导出成功！")

if __name__ == "__main__":
    export_to_onnx()
