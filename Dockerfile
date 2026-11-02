# Epiloft SAM 2 worker for RunPod Serverless
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SAM2_BUILD_CUDA=0 \
    SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_b+.yaml \
    SAM2_CHECKPOINT=/models/sam2.1_hiera_base_plus.pt \
    SAM2_DECODER_ONNX=/models/sam2_decoder.onnx

RUN apt-get update && apt-get install -y --no-install-recommends git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake weights into the image so cold starts never download them.
RUN mkdir -p /models && wget -q -O /models/sam2.1_hiera_base_plus.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

COPY export_decoder.py handler.py ./
# Build the browser decoder from the exact same weights (CPU export).
RUN python export_decoder.py --checkpoint /models/sam2.1_hiera_base_plus.pt --out /models/sam2_decoder.onnx

CMD ["python", "-u", "handler.py"]
