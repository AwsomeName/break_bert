# BERT 语义中断检测（ONNX 服务）

## 快速开始
- 构建镜像：
```bash
docker build -t bert-onnx-service .
```
- 启动服务（默认 8000）：
```bash
docker run --rm -p 8000:8000 --name bert-onnx \
  -v $(pwd)/models:/app/models:ro \
  bert-onnx-service
```
- 健康检查：
```bash
curl http://localhost:8000/health
```
- 在线推理：
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"您好，我想请问一下那个"}'
```

## 镜像导出/导入
- 导出到本地：
```bash
sudo docker save -o bert-onnx-service.tar bert-onnx-service:latest
```
- 导入到其他机器：
```bash
sudo docker load -i bert-onnx-service.tar
```

## API
- GET /health → 服务健康状态与推理后端
- POST /predict → 语义中断检测
  - 请求体：{"text":"..."}
  - 响应体：{"text":"...","status":"完整|不完整","label":1|0,"confidence":0.xx}

## 开发者说明
- 本地运行服务：
```bash
pip install "fastapi[all]" onnx onnxruntime transformers numpy uvicorn
python src/server.py
```
- ONNX 模型路径：models/bert_model/model.onnx  
- 若未生成 ONNX，可执行：
```bash
python src/export_onnx.py
```

## 目录
- Dockerfile：容器构建文件
- start.sh：容器入口脚本（启动 FastAPI 服务）
- src/server.py：ONNX 推理服务
- models/bert_model：模型与分词器
