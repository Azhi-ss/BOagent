FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/submission/code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment and code
COPY environment.yml .
COPY . .

# Install dependencies via pip
RUN pip install --no-cache-dir \
    gpytorch>=1.15.0 \
    botorch>=0.18.0,<0.19.0 \
    pandas \
    scipy \
    scikit-learn \
    pandera \
    requests \
    pyyaml

CMD ["python", "main/run_submission.py"]
