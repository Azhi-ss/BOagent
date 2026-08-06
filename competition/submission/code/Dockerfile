FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/submission

# Install the same uv release used to generate the exported lock.
RUN python -m pip install --no-cache-dir uv==0.11.32

COPY pyproject.toml uv.lock ./
COPY packages/bo-core ./packages/bo-core
COPY datasets ./datasets
COPY code ./code

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--frozen", "--no-dev", "--no-sync", "python", "code/main/run_submission.py"]
