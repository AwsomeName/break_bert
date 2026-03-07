import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

# 自定义数据集类
class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def evaluate_base_model():
    # 1. 加载数据并进行与训练脚本一致的划分 (seed=42)
    input_file = "data/processed/labeled_samples.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["truncated_text"] for item in data]
    labels = [item["label"] for item in data]
    
    _, temp_texts, _, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    _, test_texts, _, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 2. 加载未经训练的预训练基座
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 3. 准备 DataLoader
    test_dataset = DialogueDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 4. 评估
    all_preds = []
    all_labels = []
    
    print(f"正在评估未经训练的基座模型 ({model_name}) 在测试集上的表现...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
            
    # 5. 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=["不完整 (0)", "完整 (1)"], output_dict=True)
    
    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/untrained_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"评估完成！基座模型准确率: {acc:.4f}")
    print(f"报告已保存至 results/untrained_evaluation_report.json")

if __name__ == "__main__":
    evaluate_base_model()
