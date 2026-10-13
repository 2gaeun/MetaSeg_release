# Third-party notices

This distribution uses third-party inference components. Their original licenses remain applicable.

- I2SB: NVIDIA Source Code License, retained in `src/bms_deploy/_vendor/LICENSE_I2SB`.
- OpenAI guided-diffusion: MIT license, retained in `src/bms_deploy/_vendor/guided_diffusion/LICENSE_GUIDED_DIFFUSION`.
  The vendored copy is reduced to forward inference helpers.
- nnU-Net and its installed dependencies retain their upstream licenses and package metadata.
- SynthStrip 1.8 is obtained from the official image. Script/weight provenance is in `docker/synthstrip1.8.json`;
  package license metadata is copied with the runtime components.
