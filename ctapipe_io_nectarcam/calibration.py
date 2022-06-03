from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Path, IntTelescopeParameter,
    TelescopeParameter, FloatTelescopeParameter, Bool, Float

from ctapipe.calib.camera.gainselection import ThresholdGainSelector

class NectarCAMR0Corrections(TelescopeComponent):
    """
    The base R0-level calibrator. Changes the r0 container.
    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server.
    This calibrator exists in ctapipe_io_nectarcam for testing and prototyping purposes.
    """
    offset = IntTelescopeParameter(
        default_value=250,
        help=(
            'Define offset to be subtracted from the waveform *additionally*'
            ' to the pedestal. This only needs to be given when'
            ' the pedestal calibration is not applied or the offset of the'
            ' pedestal file is different from the data run'
        )
    ).tag(config=True)

    r1_sample_start = IntTelescopeParameter(
        default_value=0,
        help='Start sample for r1 waveform',
        allow_none=True,
    ).tag(config=True)

    r1_sample_end = IntTelescopeParameter(
        default_value=59,
        help='End sample for r1 waveform',
        allow_none=True,
    ).tag(config=True)

    pedestal_path = TelescopeParameter(
        trait=Path(exists=True, directory_ok=False, allow_none=True),
        allow_none=True,
        default_value=None,
        help=(
            'Path to pedestal file'
            ', required when `apply_pedestal_subtraction=True`'
        ),
    ).tag(config=True)

    calibration_path = Path(
        None, exists=True, directory_ok=False, allow_none=True,
        help='Path to calibration file',
    ).tag(config=True)

    # TODO: verify if we need these factors for NectarCAM
    calib_scale_high_gain = FloatTelescopeParameter(
        default_value=1.0,
        help='High gain waveform is multiplied by this number'
    ).tag(config=True)

    calib_scale_low_gain = FloatTelescopeParameter(
        default_value=1.0,
        help='Low gain waveform is multiplied by this number'
    ).tag(config=True)

    select_gain = Bool(
        default_value=True,
        help='Set to False to keep both gains.'
    ).tag(config=True)

    apply_pedestal_subtraction = Bool(
        default_value=True,
        help=(
            'Set to False to disable pedestal subtraction.'
            ' Providing the pedestal_path is required to perform this calibration'
        ),
    ).tag(config=True)

    gain_selection_threshold = Float(
        default_value=3500, # TODO: set default value appropriate for NectarCAM
        help='Threshold for the ThresholdGainSelector.'
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