FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY src /app/src
COPY models /app/models
COPY start.sh /app/start.sh
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "fastapi[all]" onnx onnxruntime transformers uvicorn
EXPOSE 8000
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
