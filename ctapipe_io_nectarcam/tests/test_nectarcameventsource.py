from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import N_GAINS, N_SAMPLES, N_PIXELS
from traitlets.config import Config

FIRST_EVENT_NUMBER_IN_FILE = 1
example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz")


def test_loop_over_events():
    from ctapipe_io_nectarcam import NectarCAMEventSource

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=n_events
    )

    waveform_shape = (N_GAINS, N_PIXELS, N_SAMPLES)
    for i, event in enumerate(inputfile_reader):
        assert event.trigger.tels_with_trigger == [0]
        for telid in event.trigger.tels_with_trigger:
            assert event.index.event_id == FIRST_EVENT_NUMBER_IN_FILE + i
            assert event.r0.tel[telid].waveform.shape == waveform_shape

    # make sure max_events works
    assert i == n_events - 1


def test_is_compatible():
    from ctapipe_io_nectarcam import NectarCAMEventSource

    assert NectarCAMEventSource.is_compatible(example_file_path)


def test_factory_for_nectarcam_file():
    from ctapipe.io import EventSource

    reader = EventSource(input_url=example_file_path)

    # explicit import after event_source, to test if this
    # package is detected by ctapipe
    from ctapipe_io_nectarcam import NectarCAMEventSource
    assert isinstance(reader, NectarCAMEventSource)

def test_subarray():
    from ctapipe_io_nectarcam import NectarCAMEventSource

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=n_events
        )
    subarray = inputfile_reader.subarray
    subarray.info()
    subarray.to_table()

    n_camera_pixels = inputfile_reader.subarray.tel[0].camera.geometry.n_pixels
    assert n_camera_pixels == N_PIXELS

def test_time_precision():
    """Make sure UCTS time precision is not lost
    during conversion to astropy Time, even if timestamp is cast to an int."""
    from ctapipe_io_nectarcam import time_from_unix_tai_ns
    from astropy import units as u
    from math import fabs

    tticks1 = int(1659704534052529046)
    delta = 741
    time1 = time_from_unix_tai_ns(tticks1)
    time2 = time_from_unix_tai_ns(tticks1 + delta)
    assert fabs((time2 - time1).to_value(u.ns) - delta) < 1

def test_r1_waveforms():
    from ctapipe_io_nectarcam import NectarCAMEventSource

    # without gain selection
    config = Config(dict(
        NectarCAMEventSource=dict(
            NectarCAMR0Corrections=dict(
                select_gain = False,
            )
        )
    ))

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events = n_events,
        config = config
    )

    waveform_shape = (N_GAINS, N_PIXELS, N_SAMPLES)
    for event in inputfile_reader:
        for telid in event.trigger.tels_with_trigger:
            assert event.r1.tel[telid].waveform.shape == waveform_shape

    # with gain selection
    config = Config(dict(
        NectarCAMEventSource=dict(
            NectarCAMR0Corrections=dict(
                select_gain = True,
            )
        )
    ))

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events = n_events,
        config = config
    )

    waveform_shape = (N_PIXELS, N_SAMPLES)
    for event in inputfile_reader:
        for telid in event.trigger.tels_with_trigger:
            assert event.r1.tel[telid].waveform.shape == waveform_shape

