import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
import numpy as np
from tqdm import tqdm
import datetime

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
            text,
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

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_preds, all_labels

def train():
    # 1. 加载打标后的数据
    input_file = "data/processed/labeled_samples.json"
    if not os.path.exists(input_file):
        print(f"数据文件不存在: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["truncated_text"] for item in data]
    labels = [item["label"] for item in data]
    
    print(f"加载了 {len(texts)} 条打标数据。")
    
    # 2. 划分数据集 (80% 训练, 10% 验证, 10% 测试)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 3. 准备 Tokenizer 和 DataLoader
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    train_dataset = DialogueDataset(train_texts, train_labels, tokenizer)
    val_dataset = DialogueDataset(val_texts, val_labels, tokenizer)
    test_dataset = DialogueDataset(test_texts, test_labels, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 4. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 5. 设置优化器和学习率调度器
    epochs = 3
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 6. 训练循环
    best_val_acc = 0
    metrics_history = []
    print("开始正式训练 BERT 模型...")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        }
        metrics_history.append(metrics)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models/bert_model", exist_ok=True)
            model.save_pretrained("models/bert_model")
            tokenizer.save_pretrained("models/bert_model")
            print(f"  发现更好的模型，已保存 (Val Acc: {val_acc:.4f})")
    
    # 7. 在测试集上评估最终模型
    print("\n--- 在测试集上进行最终评估 ---")
    best_model = BertForSequenceClassification.from_pretrained("models/bert_model")
    best_model.to(device)
    test_loss, test_acc, test_f1, test_preds, test_labels_list = evaluate(best_model, test_loader, device)
    
    report = classification_report(test_labels_list, test_preds, target_names=["不完整 (0)", "完整 (1)"], output_dict=True)
    
    # 8. 保存测评结果
    evaluation_results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_history": metrics_history,
        "test_results": {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "classification_report": report
        },
        "config": {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 2e-5,
            "max_length": 128
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/final_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练与测评完成！")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测评报告已保存至 results/final_evaluation_report.json")

if __name__ == "__main__":
    train()
