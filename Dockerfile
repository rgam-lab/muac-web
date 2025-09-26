# Slim, reproducible container for CI and demos (no GUI in-container).
# Uses headless OpenCV to keep the image small.

FROM python:3.11-slim

# System libs OpenCV needs to decode images, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install runtime deps
# Note: Use headless OpenCV in Docker (containers typically won't have GUI)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.26.4 \
      opencv-contrib-python-headless==4.10.0.84 \
      flask==3.0.3 \
      pytest==8.3.3 \
      ruff==0.6.8

# Now copy the project
COPY . /app

# Default working directory and environment
ENV PYTHONUNBUFFERED=1

# Examples (uncomment an ENTRYPOINT you prefer, or run commands explicitly):
# 1) Env check:
# ENTRYPOINT ["python", "scripts/check_env.py"]
# 2) Web app:
# ENTRYPOINT ["python", "-m", "flask", "--app", "src/webapp/app.py", "run", "--host", "0.0.0.0"]

# By default, just open a shell if someone uses `docker run -it`
CMD ["/bin/bash"]
