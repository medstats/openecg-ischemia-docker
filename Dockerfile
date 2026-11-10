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

# Clone OpenECG Digitizer
RUN git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git /workspace/Open-ECG-Digitizer

# Download OpenECG weights from release
RUN mkdir -p /workspace/Open-ECG-Digitizer/weights && \
    curl -L -s -o /workspace/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/lead_name_unet_weights_07072025.pt && \
    curl -L -s -o /workspace/Open-ECG-Digitizer/weights/unet_weights_07072025.pt https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/unet_weights_07072025.pt

COPY models/ /workspace/models/
COPY pipeline.py /workspace/pipeline.py

# Download Gustafsson OMI weights
RUN mkdir -p /workspace/models/gustafsson && \
    (curl -L -s -o /tmp/gustafsson.tar.gz https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/gustafsson_models.tar.gz && \
     tar -xzf /tmp/gustafsson.tar.gz -C /workspace/models/gustafsson && \
     rm -f /tmp/gustafsson.tar.gz) || true

CMD ["python", "/workspace/pipeline.py", "--help"]
