# Slim, reproducible container for CI and demos (no GUI in-container).
FROM python:3.11-slim

# System libs OpenCV needs to decode images, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed first to leverage caching
COPY requirements.txt /app/requirements.txt

# Install runtime deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.26.4 \
      opencv-contrib-python-headless==4.10.0.84 \
      flask==3.0.3 \
      pytest==8.3.3 \
      ruff==0.6.8

# Copy the project
COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# Compose will override this with a command
CMD ["/bin/bash"]
