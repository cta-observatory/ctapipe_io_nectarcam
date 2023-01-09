import numpy as np
import tables
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.containers import MonitoringContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Path, FloatTelescopeParameter, Bool, Float
)
from ctapipe.io import HDF5TableReader
from pkg_resources import resource_filename

from .constants import (
    N_GAINS, N_PIXELS, HIGH_GAIN, LOW_GAIN,
    PIXEL_INDEX,
)
from .containers import NectarCAMDataContainer

__all__ = [
    'NectarCAMR0Corrections',
]


class NectarCAMR0Corrections(TelescopeComponent):
    """
    The base R0-level calibrator. Changes the r0 container.
    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server.
    This calibrator exists in ctapipe_io_nectarcam for testing and prototyping purposes.
    """

    calibration_path = Path(
        default_value=resource_filename(
            'ctapipe_io_nectarcam',
            'resources/calibrationfile_run3255_pedrun3255_gainrun3155.hdf5'
        ),
        exists=True, directory_ok=False, allow_none=True,
        help='Path to calibration file',
    ).tag(config=True)

    calib_scale_high_gain = FloatTelescopeParameter(
        default_value=1.0,
        help='High gain waveform is multiplied by this number'
    ).tag(config=True)

    calib_scale_low_gain = FloatTelescopeParameter(
        default_value=1.0,
        help='Low gain waveform is multiplied by this number'
    ).tag(config=True)

    select_gain = Bool(
        default_value=False,
        help='Set to False to keep both gains.'
    ).tag(config=True)

    gain_selection_threshold = Float(
        default_value=3500.,
        help='Threshold for the ThresholdGainSelector.'
    ).tag(config=True)

    apply_flatfield = Bool(
        default_value=True,
        help='Apply flatfielding coefficients.'
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        The R0 calibrator for NectarCAM data.
        Fill the r1 container.
        Parameters
        ----------
        """
        super().__init__(
            subarray=subarray, config=config, parent=parent, **kwargs
        )

        self.mon_data = None

        if self.select_gain:
            self.gain_selector = ThresholdGainSelector(
                threshold=self.gain_selection_threshold,
                parent=self
            )
        else:
            self.gain_selector = None

        if self.calibration_path is not None:
            self.mon_data = self._read_calibration_file(self.calibration_path)

    def calibrate(self, event: NectarCAMDataContainer):
        for tel_id in event.r0.tel:
            r1 = event.r1.tel[tel_id]
            # check if waveform is already filled
            if r1.waveform is None:
                r1.waveform = event.r0.tel[tel_id].waveform

            r1.waveform = r1.waveform.astype(np.float32, copy=False)

            # do gain selection before converting to pe
            # like eventbuilder will do
            if self.select_gain and r1.selected_gain_channel is None:
                r1.selected_gain_channel = self.gain_selector(r1.waveform)
                r1.waveform = r1.waveform[r1.selected_gain_channel, PIXEL_INDEX]

            # apply monitoring data corrections,
            # subtract pedestal per sample and convert to pe
            if self.mon_data is not None:
                calibration = self.mon_data.tel[tel_id].calibration
                # pedestal subtraction and gain correction
                convert_to_pe(
                    waveform=r1.waveform,
                    calibration=calibration,
                    selected_gain_channel=r1.selected_gain_channel
                )
                # flatfielding
                if self.apply_flatfield:
                    flatfield = self.mon_data.tel[tel_id].flatfield
                    apply_flatfield(
                        waveform=r1.waveform,
                        flatfield=flatfield,
                        selected_gain_channel=r1.selected_gain_channel
                    )

            # do not use pixels that are broken in this run or have unusable calibration
            unusable_pixels = np.logical_and(
                event.mon.tel[tel_id].pixel_status.hardware_failing_pixels,
                self.mon_data.tel[tel_id].calibration.unusable_pixels
            )
            # set r1 waveforms to 0 for broken pixels
            if r1.selected_gain_channel is None:
                r1.waveform[unusable_pixels] = 0.0
            else:
                r1.waveform[unusable_pixels[r1.selected_gain_channel, PIXEL_INDEX]] = 0.0

            # needed for charge scaling in ctpaipe dl1 calib
            if r1.selected_gain_channel is not None:
                relative_factor = np.empty(N_PIXELS)
                relative_factor[r1.selected_gain_channel == HIGH_GAIN] = \
                    self.calib_scale_high_gain.tel[tel_id]
                relative_factor[r1.selected_gain_channel == LOW_GAIN] = \
                    self.calib_scale_low_gain.tel[tel_id]
            else:
                relative_factor = np.empty((N_GAINS, N_PIXELS))
                relative_factor[HIGH_GAIN] = self.calib_scale_high_gain.tel[tel_id]
                relative_factor[LOW_GAIN] = self.calib_scale_low_gain.tel[tel_id]

            event.calibration.tel[tel_id].dl1.relative_factor = relative_factor

    @staticmethod
    def _read_calibration_file(path):
        """
        Read the correction from hdf5 calibration file
        Only calibration and flatfield containers are filled
        """
        mon = MonitoringContainer()

        with tables.open_file(path) as f:
            tel_ids = [
                int(key[4:]) for key in f.root._v_children.keys()
                if key.startswith('tel_')
            ]

        for tel_id in tel_ids:
            with HDF5TableReader(path) as h5_table:
                base = f'/tel_{tel_id}'
                # read the calibration data
                table = base + '/calibration'
                next(h5_table.read(table, mon.tel[tel_id].calibration))

                # read flat-field data
                table = base + '/flatfield'
                next(h5_table.read(table, mon.tel[tel_id].flatfield))

        return mon


def convert_to_pe(waveform, calibration, selected_gain_channel):
    if selected_gain_channel is None:
        waveform -= calibration.pedestal_per_sample
        waveform *= calibration.dc_to_pe[:, :, np.newaxis]
    else:
        waveform = waveform - calibration.pedestal_per_sample[selected_gain_channel]
        waveform *= calibration.dc_to_pe[selected_gain_channel, PIXEL_INDEX, np.newaxis]


def apply_flatfield(waveform, flatfield, selected_gain_channel):
    if selected_gain_channel is None:
        waveform *= flatfield.relative_gain_mean[:, :, np.newaxis]
    else:
        waveform *= flatfield.relative_gain_mean[
            selected_gain_channel, PIXEL_INDEX, np.newaxis]
