FROM python:3.12-slim
# cups-client = `lp` (submits to host cupsd over IPP); libreoffice = Office->PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
      cups-client libreoffice-nogui fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 poller

# Build llama-cpp-python from source when no wheel matches this Python/platform.
# Keep the native build toolchain out of the final image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir llama-cpp-python==0.3.36 pypdf \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download TinyLlama-1.1B quantized model (Q4_K_M: ~350MB, runs in ~500MB RAM)
RUN mkdir -p /models && \
    python3 -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', '/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')"

# Setup Python app environment
WORKDIR /app
COPY poller.py .
ENV PYTHONUNBUFFERED=1 HOME=/home/poller LLAMA_MODEL_PATH=/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
USER poller
CMD ["python3", "poller.py"]
