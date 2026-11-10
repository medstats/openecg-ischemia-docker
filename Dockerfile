FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    pillow \
    matplotlib \
    scikit-learn \
    scikit-image \
    torchvision \
    pyyaml \
    timm

RUN git clone https://github.com/c-w-k/Electrocardiogram-Digitization.git /workspace/Open-ECG-Digitizer

COPY models/ /workspace/models/
COPY pipeline.py /workspace/pipeline.py

# Download Gustafsson OMI weights
RUN mkdir -p /workspace/models/gustafsson && \
    (curl -L -s -o /tmp/gustafsson.tar.gz https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/gustafsson_models.tar.gz && \
     tar -xzf /tmp/gustafsson.tar.gz -C /workspace/models/gustafsson && \
     rm -f /tmp/gustafsson.tar.gz) || true

CMD ["python", "/workspace/pipeline.py", "--help"]
