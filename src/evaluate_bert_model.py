import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

# 自定义数据集类
class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dialogue = self.data[idx]["dialogue"]
        label = self.data[idx]["label"]
        
        encoding = self.tokenizer(
            dialogue,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 评估BERT模型
def evaluate_bert_model():
    # 加载数据
    test_data = load_data("/home/lc/bert-broken/data/test_data.json")
    
    print(f"测试集大小: {len(test_data)}")
    
    # 加载BERT tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained("/home/lc/bert-broken/models/bert_tokenizer")
    model = BertForSequenceClassification.from_pretrained("/home/lc/bert-broken/models/bert_model")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 创建数据集和数据加载器
    test_dataset = DialogueDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # 评估模型
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 检查是否达到要求
    if accuracy >= 0.95 and recall >= 0.95:
        print("模型性能达到要求：准确率和召回率均达到95%以上")
    else:
        print(f"模型性能未达到要求：准确率{accuracy:.2%}，召回率{recall:.2%}，需要达到95%以上")
    
    return accuracy, recall, precision, f1

if __name__ == "__main__":
    evaluate_bert_model()
