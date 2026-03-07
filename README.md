# BERT 语义中断检测 (Semantic Break Detection)

本项目旨在训练一个 BERT 模型，用于实时检测中文对话中的语义完整性。该模型可以作为语音活动检测 (VAD) 的后处理器，有效拦截由于 VAD 误触发导致的“话没说完就启动 AI 回复”的场景。

## 📊 训练收益报告

经过 5000+ 条高质量口语语料的微调，模型性能相比预训练基座有了质的飞跃。

| 指标 | 训练前 (Base Model) | 训练后 (Best Epoch) | 最终测试集 |
| :--- | :--- | :--- | :--- |
| **准确率 (Accuracy)** | 14.77% | **88.82%** | 85.03% |
| **不完整 (0) F1-Score** | 0.00 | **0.93** | 0.916 |

### 结论
- **高收益**: 训练直接将模型从不可用提升到了工业级初步可用水平。
- **核心优势**: 对“断句/没说完”的识别极其精准（Recall 0 > 94%），能有效起到“语义防火墙”的作用。

## 🚀 部署与性能

### 1. ONNX 导出与服务化
为了实现高性能、低依赖的部署，我们将 PyTorch 模型转换为了 ONNX 格式，并基于 **FastAPI** 和 **ONNX Runtime** 构建了推理服务。

- **模型格式**: `models/bert_model/model.onnx`
- **服务脚本**: `src/server.py`
- **客户端脚本**: `src/client.py`

### 2. CPU vs. GPU 性能对比
在相同的测试集上，我们对比了纯 CPU 与 GPU 的推理时延：

| 指标 | **CPU 推理时延** | GPU 推理时延 |
| :--- | :---: | :---: |
| **平均时延** | **73.92 ms** | **71.93 ms** |

## 🛠️ 使用方案

### 1. 环境准备
```bash
pip install "fastapi[all]" onnx onnxruntime requests transformers torch sklearn tqdm
```

### 2. 启动服务
```bash
python src/server.py
```

### 3. 客户端测试
```bash
python src/client.py
```

## 📂 项目结构
- `src/`: 核心逻辑脚本
  - `batch_generate_dialogues.py`: 基于 LLM 的语料合成
  - `train_bert_model.py`: BERT 微调与评估
  - `server.py`: 基于 ONNX 的 FastAPI 推理服务
  - `client.py`: 推理服务客户端
- `data/`: 语料数据
  - `raw/samples.json`: 1000 条原始对话片段
  - `processed/labeled_samples.json`: 5000+ 条带标签的训练数据
- `results/`: 测评报告与指标
