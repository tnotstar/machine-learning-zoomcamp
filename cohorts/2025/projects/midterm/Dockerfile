FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
COPY "predict.py" "pipeline_v1.bin" ./

RUN uv sync --locked

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-8080}"]
