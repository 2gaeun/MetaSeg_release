# Official SynthStrip 1.8 assets, pinned to the inspected amd64 image digest.
FROM docker.io/freesurfer/synthstrip@sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06 AS synthstrip

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
COPY --from=synthstrip /freesurfer/mri_synthstrip /opt/synthstrip/1.8/mri_synthstrip
COPY --from=synthstrip /freesurfer/models/synthstrip.1.pt /opt/synthstrip/1.8/models/synthstrip.1.pt
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/surfa /opt/synthstrip/1.8/python/surfa
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/surfa-0.6.1.dist-info /opt/synthstrip/1.8/python/surfa-0.6.1.dist-info
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/xxhash /opt/synthstrip/1.8/python/xxhash
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/xxhash-3.5.0.dist-info /opt/synthstrip/1.8/python/xxhash-3.5.0.dist-info
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/importlib_resources /opt/synthstrip/1.8/python/importlib_resources
COPY --from=synthstrip /freesurfer/env/lib/python3.10/site-packages/importlib_resources-6.5.2.dist-info /opt/synthstrip/1.8/python/importlib_resources-6.5.2.dist-info
ENV BMS_SYNTHSTRIP_HOME=/opt/synthstrip/1.8 FREESURFER_HOME=/opt/synthstrip/1.8
ENV PYTHONPATH=/opt/synthstrip/1.8/python
ENV PATH=/opt/synthstrip/1.8:${PATH}
COPY src /opt/brainmetaseg/src
RUN python -m pip install --no-deps --no-build-isolation /opt/brainmetaseg/src && \
    python -c "from bms_deploy.postprocess import SynthStrip; SynthStrip()"
ENV METASEG_IN_CONTAINER=1 HOME=/tmp
WORKDIR /bundle
CMD ["bash", "/bundle/run.sh"]
