# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

# Set environment variables to optimize Python runtime in Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (required for building certain Python wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files first to optimize layer caching
COPY pyproject.toml LICENSE README.md /app/
COPY src/ /app/src/

# Install the library and its dependencies
RUN pip install --no-cache-dir .

# Copy utilities and demo notebook
COPY WaveSeekerNet_Demo.ipynb /app/

# Expose Jupyter port if users want to run the demo notebook
EXPOSE 8888

# Default test command
CMD ["python", "-c", "import WaveSeekerNet; print('WaveSeekerNet successfully installed!')"]
