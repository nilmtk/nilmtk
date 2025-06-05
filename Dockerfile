# Base image with Python 3.11
FROM python:3.11-slim

# Link back to the source repo so GitHub auto-associates the package
LABEL org.opencontainers.image.source="https://github.com/nilmtk/nilmtk"

# Install build tools and Git for pip VCS installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Upgrade pip and install NILMTK (using setup.py)
RUN pip install --upgrade pip && \
    pip install .

# Default to a shell for interactive use
CMD ["bash"]
