From nvcr.io/nvidia/nemo:22.12

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir git+https://github.com/NVIDIA/NeMo.git@main \
 && pip install --no-cache-dir -U sagemaker-training
 
WORKDIR /