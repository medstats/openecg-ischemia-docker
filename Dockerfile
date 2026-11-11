FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MPLBACKEND=Agg TMPDIR=/storage/emulated/0/tmp
WORKDIR /workspace
RUN mkdir -p /storage/emulated/0/tmp && apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgl1 libglib2.0-0 fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-inference.txt /workspace/requirements-inference.txt
RUN pip install --no-cache-dir -r requirements-inference.txt

# Pin the digitizer source to the revision tested with the coordinate/calibration patch.
ARG OPENECG_COMMIT=97a15087d4abcda843da8c58ee74b1d8f47e6f9a
RUN git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git /workspace/Open-ECG-Digitizer \
    && git -C /workspace/Open-ECG-Digitizer checkout "${OPENECG_COMMIT}"
COPY patches/openecg-inference.patch /workspace/openecg-inference.patch
RUN git -C /workspace/Open-ECG-Digitizer apply /workspace/openecg-inference.patch

# Fail the build on a failed/partial download or unexpected checkpoint checksum.
RUN mkdir -p /workspace/Open-ECG-Digitizer/weights && \
    curl -fL --retry 5 --retry-delay 3 --retry-all-errors \
      -o /workspace/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt \
      https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/lead_name_unet_weights_07072025.pt && \
    curl -fL --retry 5 --retry-delay 3 --retry-all-errors \
      -o /workspace/Open-ECG-Digitizer/weights/unet_weights_07072025.pt \
      https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/unet_weights_07072025.pt && \
    echo '840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2  /workspace/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt' | sha256sum -c - && \
    echo '17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b  /workspace/Open-ECG-Digitizer/weights/unet_weights_07072025.pt' | sha256sum -c -

COPY models/ /workspace/models/
RUN mkdir -p /workspace/models/gustafsson /workspace/downloads && \
    curl -fL --retry 5 --retry-delay 3 --retry-all-errors \
      -o /workspace/downloads/gustafsson.tar.gz \
      https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/gustafsson_models.tar.gz && \
    echo '6d6ecccaabb1a971e0445c8120705b05813fe34b24b8be0210c1665ae66aa0af  /workspace/downloads/gustafsson.tar.gz' | sha256sum -c - && \
    tar -xzf /workspace/downloads/gustafsson.tar.gz -C /workspace/models/gustafsson && \
    rm /workspace/downloads/gustafsson.tar.gz && \
    test -s /workspace/models/gustafsson/model_1.pth && \
    test -s /workspace/models/gustafsson/model_2.pth && \
    test -s /workspace/models/gustafsson/model_3.pth && \
    test -s /workspace/models/gustafsson/model_4.pth && \
    test -s /workspace/models/gustafsson/model_5.pth

COPY ecg_adapters.py model_inference.py audit.py pipeline.py run_all_ecg.py /workspace/
COPY tests/ /workspace/tests/
# CPU tests and real-weight smoke test are mandatory before the image can be published.
RUN python -m unittest discover -s tests -v && \
    python tests/smoke_models.py --models-dir /workspace/models && \
    python -c "from run_all_ecg import load_digitizer; from pathlib import Path; load_digitizer(Path('/workspace/Open-ECG-Digitizer'), 'cpu', 2000)" && \
    python /workspace/run_all_ecg.py --help
CMD ["python", "/workspace/run_all_ecg.py", "--help"]
