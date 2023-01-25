[![CI](https://github.com/cta-observatory/ctapipe_io_nectarcam/workflows/CI/badge.svg?branch=master)](https://github.com/cta-observatory/ctapipe_io_nectarcam/actions?query=workflow%3ACI+branch%3Amaster)

EventSource Plugin for ctapipe, able to read NectarCam zfits files
and calibrate them to R1 as needed for ctapipe tools.

To install into an existing environment, just do:
```shell
pip install .
```

For development, create a new environment and run the development install:
```shell
conda env create -n nectar-dev -f environment.yml
conda activate nectar-dev
pip install -e .
```