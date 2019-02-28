"""
Container structures for data that should be read or written to disk
"""

from ctapipe.core import Container, Field, Map
from ctapipe.io.containers import DataContainer

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
    tib_data = Field([], "TIB data array")
    cdts_data = Field([], "CDTS data array")
    swat_data = Field([], "SWAT data array")
    counters = Field([], "counters")


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


class NectarCAMDataContainer(DataContainer):
    """
    Data container including NectarCAM information
    """
    nectarcam = Field(NectarCAMContainer(), "NectarCAM specific Information")
