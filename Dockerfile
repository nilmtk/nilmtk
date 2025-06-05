FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/enfuego27826/nilmtk"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv globally via pip
RUN pip install uv

WORKDIR /app

# Copy pyproject.toml, uv.lock, and README.md so build backend can access metadata files
COPY pyproject.toml uv.lock* README.md ./

# Install nilmtk using uv pip install
RUN uv pip install --system .

# Copy all source files after install (optional, if you want the whole repo inside container)
COPY . .

CMD ["bash"]
