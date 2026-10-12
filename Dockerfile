FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY train.py evaluate.py predict.py ./

# Dataset and outputs are mounted at runtime (see docker-compose.yml)
VOLUME ["/app/data/gtsrb", "/app/outputs"]

CMD ["python", "train.py"]
