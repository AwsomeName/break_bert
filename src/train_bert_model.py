import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
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

def build_tiny_model(tokenizer, num_labels):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(os.getenv("TINY_HIDDEN_SIZE", "128")),
        num_hidden_layers=int(os.getenv("TINY_NUM_LAYERS", "2")),
        num_attention_heads=int(os.getenv("TINY_NUM_HEADS", "4")),
        intermediate_size=int(os.getenv("TINY_INTERMEDIATE_SIZE", "512")),
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=num_labels
    )
    return BertForSequenceClassification(config)

def train():
    # 1. 加载打标后的数据
    input_file = os.getenv("INPUT_FILE", "data/processed/labeled_samples.json")
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
    model_name = os.getenv("MODEL_NAME", "hfl/rbt3")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "models/bert_tokenizer")
    use_tiny_scratch = os.getenv("USE_TINY_SCRATCH", "0") == "1"
    local_files_only = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
    allow_scratch_fallback = os.getenv("ALLOW_SCRATCH_FALLBACK", "1") == "1"
    epochs = int(os.getenv("EPOCHS", "3"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
    output_model_dir = os.getenv("OUTPUT_MODEL_DIR", "models/bert_tiny_model")
    output_report_path = os.getenv("OUTPUT_REPORT_PATH", "results/final_evaluation_report_bert_tiny.json")
    tokenizer_source = model_name if model_name else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_files_only)
    
    train_dataset = DialogueDataset(train_texts, train_labels, tokenizer)
    val_dataset = DialogueDataset(val_texts, val_labels, tokenizer)
    test_dataset = DialogueDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 4. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if use_tiny_scratch:
        model = build_tiny_model(tokenizer, num_labels=2)
    else:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                local_files_only=local_files_only
            )
        except Exception as e:
            if not allow_scratch_fallback:
                raise
            print(f"预训练 tiny 加载失败，切换为 tiny-scratch：{e}")
            use_tiny_scratch = True
            model = build_tiny_model(tokenizer, num_labels=2)
    model.to(device)
    
    # 5. 设置优化器和学习率调度器
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
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
            os.makedirs(output_model_dir, exist_ok=True)
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)
            print(f"  发现更好的模型，已保存 (Val Acc: {val_acc:.4f})")
    
    # 7. 在测试集上评估最终模型
    print("\n--- 在测试集上进行最终评估 ---")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_model_dir, local_files_only=local_files_only)
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
            "input_file": input_file,
            "model_name": model_name if model_name else "tiny-bert-scratch",
            "tokenizer_source": tokenizer_source,
            "use_tiny_scratch": use_tiny_scratch,
            "local_files_only": local_files_only,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "max_length": 128
        }
    }
    
    report_dir = os.path.dirname(output_report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练与测评完成！")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测评报告已保存至 {output_report_path}")

if __name__ == "__main__":
    train()
