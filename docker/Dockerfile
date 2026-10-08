FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
COPY src /opt/brainmetaseg/src
RUN python -m pip install --no-deps /opt/brainmetaseg/src
# The whole portable dist is mounted at /bundle so large checkpoints remain external to the image.
WORKDIR /bundle
CMD ["/bin/bash"]
