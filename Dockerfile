# --- STAGE 1: Builder ---
FROM python:3.10-slim as builder

ARG HARDWARE=cpu

# Install generic build tools
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl libvulkan-dev glslc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

# 1. UNIFY: Set Workdir to /app
WORKDIR /app

# 2. Copy dependency files to /app
COPY pyproject.toml uv.lock* ./

# 3. Create venv inside /app (.venv is now at /app/.venv)
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Install Dependencies (Using $HARDWARE correctly)
RUN if [ "$HARDWARE" = "cuda" ] ; then \
        CMAKE_ARGS="-DLLAMA_CUDA=ON" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install . ; \
    elif [ "$HARDWARE" = "vulkan" ] ; then \
        CMAKE_ARGS="-DGGML_VULKAN=ON" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install . ; \
    else \
        CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" uv pip install --no-cache-dir --force-reinstall llama-cpp-python && \
        uv pip install . ; \
    fi

# --- STAGE 2: Runtime ---
FROM python:3.10-slim

# 1. Install Runtime Drivers
RUN apt-get update && apt-get install -y \
    libgomp1 libvulkan1 mesa-vulkan-drivers vulkan-tools \
    libopenblas-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. UNIFY: Set Workdir to /app (Matches Builder)
WORKDIR /app

# 3. Copy the venv to the exact same location
COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 4. Copy Source Code
# This puts your local 'src' folder into '/app/src'
COPY src/ src/

# 5. Create models directory
RUN mkdir -p models

# 6. CRITICAL: Set PYTHONPATH to the location inside the container
# Since we copied to 'src/', the code is in '/app/src'
ENV PYTHONPATH="/app/src:$PYTHONPATH"

EXPOSE 8000

# 7. Start Command
# Because PYTHONPATH includes /app/src, we import 'app.main' directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]