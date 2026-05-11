FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash coreutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sandbox

COPY container_entrypoint.py /sandbox/container_entrypoint.py

RUN useradd --create-home --shell /bin/bash sandboxuser \
    && chown -R sandboxuser:sandboxuser /sandbox

USER sandboxuser

ENTRYPOINT ["python", "/sandbox/container_entrypoint.py"]
