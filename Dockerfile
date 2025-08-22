FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ffmpeg \
    build-essential \
    cmake \
    libopenblas-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy source files
COPY . .

# Build whisper.cpp
RUN mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j4

# Copy models
RUN mkdir -p models \
    && cd models \
    && wget https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# Create NVIDIA-specific symlink for explicit model usage (if needed)
RUN if [ -f "models/ggml-large-v3-turbo.bin" ]; then \
        echo "Using NVIDIA-compatible ggml model"; \
    fi

# Install Python dependencies for proxy
RUN pip3 install requests

# Expose port
EXPOSE 8080

# Run the proxy server
CMD ["python3", "proxy.py"]
