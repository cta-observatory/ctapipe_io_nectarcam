"""
Container structures for data that should be read or written to disk
"""

from ctapipe.core import Container, Field, Map
from ctapipe.containers import ArrayEventContainer

__all__ = [
    'NectarCAMContainer',
    'NectarCAMCameraContainer',
]


class NectarCAMServiceContainer(Container):
    """
    Container for Fields that are specific to each NectarCAM camera configuration
    """

    # Data from the CameraConfig table
    telescope_id = Field(-1, "telescope id")
    cs_serial = Field(None, "serial number of the camera server")
    configuration_id = Field(None, "id of the CameraConfiguration")
    date = Field(None, "NTP start of run date")
    num_pixels = Field(-1, "number of pixels")
    num_samples = Field(-1, "num samples")
    pixel_ids = Field([], "id of the pixels in the waveform array")
    data_model_version = Field(None, "data model version")

    idaq_version = Field(0o0, "idaq version")
    cdhs_version = Field(0o0, "cdhs version")
    acquisition_mode = Field(-1, "acquisition mode")
    algorithms = Field(None, "algorithms")
    # pre_proc_algorithms = Field(None, "pre processing algorithms")
    module_ids = Field([], "module ids")
    num_modules = Field(-1, "number of modules")


class NectarCAMEventContainer(Container):
    """
    Container for Fields that are specific to each NectarCAM event
    """

    # Data from the CameraEvent table
    configuration_id = Field(None, "id of the CameraConfiguration")
    event_id = Field(None, "local id of the event")
    tel_event_id = Field(None, "global id of the event")
    pixel_status = Field([], "status of the pixels")
    ped_id = Field(None, "tel_event_id of the event used for pedestal substraction")
    module_status = Field([], "status of the modules")
    extdevices_presence = Field(None, "presence of data for external devices")
    #tib_data = Field([], "TIB data array")
    #cdts_data = Field([], "CDTS data array")

    tib_event_counter = Field(None, "TIB event counter")
    tib_pps_counter = Field(None, "TIB pps counter")
    tib_tenMHz_counter = Field(None, "TIB 10 MHz counter")
    tib_stereo_pattern = Field(None, "TIB stereo pattern")
    tib_masked_trigger = Field(None, "TIB trigger mask")

    ucts_event_counter =  Field(None, "UCTS event counter")
    ucts_pps_counter = Field(None, "UCTS pps counter")
    ucts_clock_counter = Field(None, "UCTS clock counter")
    ucts_timestamp = Field(None, "UCTS timestamp")
    ucts_camera_timestamp = Field(None, "UCTS camera timestamp")
    ucts_trigger_type = Field(None, "UCTS trigger type")
    ucts_white_rabbit_status = Field(None, "UCTS White Rabbit status")
    ucts_address = Field(None, "UCTS address")
    ucts_busy_counter = Field(None, "UCTS busy counter")
    ucts_stereo_pattern = Field(None, "UCTS stereo pattern")
    ucts_num_in_bunch = Field(None, "UCTS num in bunch (for debugging)")
    cdts_version = Field(None, "CDTS version")

    swat_data = Field([], "SWAT data array")
    counters = Field([], "counters")

    # FEB counters and trigger pattern
    feb_abs_event_id = Field(None, "FEB absolute id of the event")
    feb_event_id = Field(None, "FEB event count since last PPS")
    feb_pps_cnt = Field(None, "FEB count of PPS")
    feb_ts1 = Field(None, "FEB TS1")
    feb_ts2_trig = Field(None, "FEB TS2 Trigger Shift")
    feb_ts2_pps = Field(None, "FEB TS2 PPS Shift")
    trigger_pattern =  Field(None, "FEB L0 trigger pattern")
    native_charge = Field(None, "FEB native charge")

class NectarCAMCameraContainer(Container):
    """
    Container for Fields that are specific to each NectarCAM camera
    """
    evt = Field(NectarCAMEventContainer(), "NectarCAM specific event Information")
    svc = Field(NectarCAMServiceContainer(), "NectarCAM specific camera_config "
                                             "Information")


class NectarCAMContainer(Container):
    """
    Storage for the NectarCAMCameraContainer for each telescope
    """
    tels_with_data = Field([], "list of telescopes with data")

    # create the camera container
    tel = Field(
        Map(NectarCAMCameraContainer),
        "map of tel_id to NectarCAMCameraContainer")


class NectarCAMDataContainer(ArrayEventContainer):
    """
    Data container including NectarCAM information
    """
    nectarcam = Field(NectarCAMContainer(), "NectarCAM specific Information")
