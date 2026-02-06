# --- STAGE 1: Builder ---
FROM python:3.10-slim as builder

# Options: cpu, cuda, rocm, vulkan
ARG HARDWARE=cpu

# Install generic build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libvulkan-dev \
    glslc \
    && rm -rf /var/lib/apt/lists/*

# Install uv from Astral's UV image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

WORKDIR /app

# Copy the dependencies files
COPY pyproject.toml uv.lock* ./

# Create a virtual enviroment for isolation
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Install Dependencies with Hardware Flags
# We use 'uv pip install' directly here for speed
RUN if [ "HARDWARE" = "cuda" ] ; then \
        CMAKE_ARGS="-DLLAMA_CUDA=ON" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install -r pyproject.toml; \
    elif [ "HARDWARE" = "vulkan" ] ; then \
        CMAKE_ARGS="-DGGML_VULKAN=ON" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install -r pyproject.toml; \
    else \
        CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install -r pyproject.toml; \
    fi

# --- STAGE 2: Runtime ---
FROM python:3.10-slim

# 1. Install Runtime Vulkan Drivers
# 'libvulkan1' and 'mesa-vulkan-drivers' are needed to TALK to the GPU at runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the virtual environment from builder
# This works because we used 'uv venv' in the builder
ENV VIRTUAL_ENV=/app/.venv
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]