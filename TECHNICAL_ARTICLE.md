# BERT 中文语义中断检测：从数据到服务的端到端实现

本项目旨在训练一个基于 BERT 的二分类模型，实时判断中文对话片段在截断处的语义是否“完整”。它可作为 VAD（语音活动检测）的后处理“语义防火墙”，有效避免“用户还没说完系统就开始回复”的体验问题。

- 项目地址结构与核心脚本：
  - 训练与评测：[train_bert_model.py](file:///home/lc/bert-broken/src/train_bert_model.py)、[evaluate_untrained.py](file:///home/lc/bert-broken/src/evaluate_untrained.py)
  - 数据合成与预处理：[batch_generate_dialogues.py](file:///home/lc/bert-broken/src/batch_generate_dialogues.py)、[truncate_multiplier.py](file:///home/lc/bert-broken/src/truncate_multiplier.py)、[label_batch_multiplier.py](file:///home/lc/bert-broken/src/label_batch_multiplier.py)
  - ONNX 导出与服务：[export_onnx.py](file:///home/lc/bert-broken/src/export_onnx.py)、[server.py](file:///home/lc/bert-broken/src/server.py)、[client.py](file:///home/lc/bert-broken/src/client.py)
  - PyTorch 服务备选实现：[services/app.py](file:///home/lc/bert-broken/services/app.py)
  - 结果报告样例：results 目录（如 [final_evaluation_report.json](file:///home/lc/bert-broken/results/final_evaluation_report.json)）

---

**业务背景**

- VAD 常以静音阈值判断“说话结束”。真实对话中用户会出现思考、犹豫、语气词等短停顿，导致系统误判“已结束”而提前抢答。
- 我们通过“语义中断检测”在 VAD 之后再加一道判别：若文本在截断处语义未完成则阻断回复，提升交互自然度与满意度。

---

**系统架构**

- 数据层：合成多场景/多人设中文对话 → 抽取短片段 → 严格随机截断 → 大模型打标（完整/不完整）。
- 模型层：使用 bert-base-chinese 作为 backbone，在标注数据上微调二分类头。
- 服务层：导出 ONNX，基于 FastAPI + ONNX Runtime 部署高效推理；亦提供 PyTorch 直接推理服务备选。
- 客户端：提供简单 HTTP 客户端脚本用于连通性与时延验证。

数据流水线脚本一览：
- 合成/采样：
  - 多场景+人设合成：[batch_generate_dialogues.py](file:///home/lc/bert-broken/src/batch-generate-dialogues.py)
  - 公开数据再采样（可选）：[resample_from_modelscope.py](file:///home/lc/bert-broken/src/resample_from_modelscope.py)、[resample_dialogue.py](file:///home/lc/bert-broken/src/resample_dialogue.py)
  - 片段化（按 100–300 字）：[extract_segments.py](file:///home/lc/bert-broken/src/extract_segments.py)
- 截断与打标：
  - 多倍严格随机截断：[truncate_multiplier.py](file:///home/lc/bert-broken/src/truncate_multiplier.py)
  - 批量打标（Qwen-Turbo）：[label_batch_multiplier.py](file:///home/lc/bert-broken/src/label_batch_multiplier.py)

产物格式：
- 原始对话：data/raw/samples.json
- 截断样本：data/processed/truncated_samples.json（id、truncated_text、full_text）
- 打标样本：data/processed/labeled_samples.json（增加 label=0/1）

---

**模型训练与评测**

- 训练脚本：[train_bert_model.py](file:///home/lc/bert-broken/src/train_bert_model.py)
  - 数据拆分：train/val/test = 8/1/1（分层抽样）
  - 模型：bert-base-chinese + 分类头（num_labels=2）
  - 优化器与调度：AdamW + 线性 warmup/schedule
  - 度量：Accuracy、Macro F1，并在验证集上保存最优权重至 models/bert_model
- 评测脚本：
  - 基座对比：[evaluate_untrained.py](file:///home/lc/bert-broken/src/evaluate_untrained.py)
  - 最终测试报告：results/final_evaluation_report.json

核心结果（典型一次实验，详见结果文件）：
- 未训练基座：
  - 准确率 14.77%（见 [untrained_evaluation_report.json](file:///home/lc/bert-broken/results/untrained_evaluation_report.json)）
- 微调后（测试集）：
  - 准确率 84.83%、不完整类 F1≈0.915（见 [final_evaluation_report.json](file:///home/lc/bert-broken/results/final_evaluation_report.json)）
- 结论：对“未完成（0）”的识别显著增强，具备在 VAD 后端部署的工程价值。对“完整（1）”的召回偏保守，可通过增加正样本比例进一步优化。

---

**导出与服务**

- ONNX 导出：[export_onnx.py](file:///home/lc/bert-broken/src/export_onnx.py)
  - 输入：input_ids、attention_mask
  - 动态轴：batch_size、sequence_length
  - 导出到 models/bert_model/model.onnx
- ONNX 服务端：[server.py](file:///home/lc/bert-broken/src/server.py)
  - 框架：FastAPI + ONNX Runtime（CPUExecutionProvider）
  - 接口：
    - POST /predict
      - 请求：{"text": "您好，我想请问一下那个"}
      - 响应：{"text": "...", "status": "完整|不完整", "label": 1|0, "confidence": 0.92}
    - GET /health（健康检查）
- 客户端测试：[client.py](file:///home/lc/bert-broken/src/client.py)
  - 批量构造测试句，打印类别、置信度与时延
- PyTorch 服务备选：[services/app.py](file:///home/lc/bert-broken/services/app.py)
  - 接口：/detect、/batch_detect，便于灰度或对照测试
- Docker 化：[Dockerfile](file:///home/lc/bert-broken/Dockerfile)
  - 基于 python:3.10-slim，pip 安装依赖，默认启动 services/app.py

---

**性能与延迟**

- 在相同测试集上的一次对比（示例）：
  - CPU ONNX 推理平均时延 ≈ 73.92 ms
  - GPU PyTorch 推理平均时延 ≈ 71.93 ms
- 结论：ONNX CPU 已能满足多数在线场景；若在服务端已有 GPU 资源，PyTorch 直推亦可。

---

**快速开始**

环境安装（建议）：
```bash
# 方式一：按 README 的一键安装
pip install "fastapi[all]" onnx onnxruntime requests transformers torch sklearn tqdm

# 如需使用数据脚本（可选）：
pip install datasets dashscope modelscope
```

复现实验（最小闭环）：
```bash
# 1) 准备数据（已有示例）
#   data/raw/samples.json 已包含样例。也可使用脚本自行合成：
#   python src/batch_generate_dialogues.py
#   python src/extract_segments.py

# 2) 随机截断 5 倍扩增
python src/truncate_multiplier.py

# 3) 大模型打标（需配置 dashscope API Key 为环境变量）
#    export DASHSCOPE_API_KEY=xxx  # 推荐方式；脚本里为示例写法
python src/label_batch_multiplier.py

# 4) 训练与评测
python src/evaluate_untrained.py   # 基座对照（可选）
python src/train_bert_model.py     # 训练，产出 models/bert_model 与 results/...

# 5) 导出 ONNX（用于高性能部署）
python src/export_onnx.py

# 6) 启动 ONNX 服务并使用客户端测试
python src/server.py
python src/client.py
```

生产部署要点：
- 建议容器化部署，结合 liveness/readiness 探针与限流。
- 与 VAD 集成策略：当 label=0（不完整）或置信度<阈值（如 0.8）时继续等待用户；label=1 且置信度≥阈值时再触发回复。

---

**目录结构**

```
bert-broken/
├─ src/                      # 训练、数据、导出、服务核心脚本
│  ├─ train_bert_model.py
│  ├─ evaluate_untrained.py
│  ├─ export_onnx.py
│  ├─ server.py
│  ├─ client.py
│  ├─ batch_generate_dialogues.py / label_batch_multiplier.py / truncate_multiplier.py ...
├─ services/
│  └─ app.py                 # PyTorch 服务备选实现
├─ data/
│  ├─ raw/                   # 原始对话/抽样数据
│  └─ processed/             # 截断与打标后的训练数据
├─ models/
│  ├─ bert_model/            # 训练产物与 ONNX（导出后）
│  └─ bert_tokenizer/        # 分词器（如有拆分保存）
└─ results/                  # 各阶段指标与报告
```

---

**实现细节与设计选择**

- 数据截断策略：严格随机字符级截断（10%–90% 范围），高复现对话中断形态，提升“未完成”类的可分性。
- 标注方式：采用 LLM 进行自动化 0/1 判定，后续可叠加人工校验；实践中“未完成”类更易判定，正负样本比可动态调优。
- 模型选择：中文 BERT-base 足以胜任二分类语义边界识别；更大模型能进一步提升“完整”类召回。
- 导出与服务：ONNX 在 CPU 上具备稳定、轻依赖优势；API 简洁优化为便于上游模块（VAD/ASR/NLU）对接。

---

**常见问题**

- 依赖安装与版本：
  - 训练使用 PyTorch + transformers + sklearn + tqdm。
  - 服务使用 FastAPI + ONNX Runtime（或 PyTorch 直推）。
- 数据安全与密钥：
  - 请通过环境变量注入 DashScope/云厂商 Key，避免明文出现在脚本或仓库中。
- 类别不均衡：
  - 如需提升“完整（1）”的召回率，可增加正样本比例或引入 class weight/focal loss 等技巧。

---

**附：关键文件索引**

- 训练与评测：
  - [train_bert_model.py](file:///home/lc/bert-broken/src/train_bert_model.py)
  - [evaluate_untrained.py](file:///home/lc/bert-broken/src/evaluate_untrained.py)
- 数据流水线：
  - [batch_generate_dialogues.py](file:///home/lc/bert-broken/src/batch_generate_dialogues.py)
  - [truncate_multiplier.py](file:///home/lc/bert-broken/src/truncate_multiplier.py)
  - [label_batch_multiplier.py](file:///home/lc/bert-broken/src/label_batch_multiplier.py)
- 部署：
  - [export_onnx.py](file:///home/lc/bert-broken/src/export_onnx.py)
  - [server.py](file:///home/lc/bert-broken/src/server.py)
  - [client.py](file:///home/lc/bert-broken/src/client.py)
  - [services/app.py](file:///home/lc/bert-broken/services/app.py)
  - [Dockerfile](file:///home/lc/bert-broken/Dockerfile)

