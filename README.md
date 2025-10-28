[![CI](https://github.com/cta-observatory/ctapipe_io_nectarcam/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cta-observatory/ctapipe_io_nectarcam/actions/workflows/ci.yml?query=workflow%3ACI+branch%3Amain)
[![DOI](https://zenodo.org/badge/172958150.svg)](https://doi.org/10.5281/zenodo.17466683)

EventSource Plugin for ctapipe, able to read NectarCAM zfits files
and calibrate them to R1 as needed for ctapipe tools.

To install into an existing environment, just do:
```shell
pip install .
```

or:
```shell
mamba install -c conda-forge ctapipe-io-nectarcam 
```

For development, create a new environment and run the development install:
```shell
mamba env create -n nectar-dev -f environment.yml
mamba activate nectar-dev
pip install -e .
pre-commit install
```
