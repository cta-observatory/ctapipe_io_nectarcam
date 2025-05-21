import numpy as np
from astropy import units as u
from astropy.coordinates.earth import EarthLocation

N_GAINS = 2
N_MODULES = 265
N_PIXELS_MODULE = 7
N_PIXELS = N_MODULES * N_PIXELS_MODULE
N_SAMPLES = 60
HIGH_GAIN = 0
LOW_GAIN = 1

PIXEL_INDEX = np.arange(N_PIXELS)

# Dummy location for NectarCAM, need for ctapipe.instrument.subarray
nectarcam_location = EarthLocation(
    lon=0 * u.deg,
    lat=0 * u.deg,
    height=0 * u.m,
)
