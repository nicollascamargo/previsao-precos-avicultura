# Multi-stage: build wheels once, ship a slim runtime without a compiler.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build \
    && pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim AS runtime

# Never run the API as root.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels precos-avicultura \
    && rm -rf /wheels

COPY --chown=app:app models ./models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "precos_avicultura.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
