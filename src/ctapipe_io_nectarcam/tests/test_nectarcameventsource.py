from ctapipe.utils import get_dataset_path
from traitlets.config import Config

from ctapipe_io_nectarcam.constants import N_GAINS, N_PIXELS, N_SAMPLES

# Beware: the current test file, NectarCAM.Run4717.0000.fits.fz, only contains even
# events. The test_loop_over_events test and FIRST_EVENT_NUMBER_IN_FILE should match
# even events !
EXAMPLE_FILE_PATH = get_dataset_path("NectarCAM.Run4717.0000.fits.fz")
FIRST_EVENT_NUMBER_IN_FILE = 2
EVENT_JUMP = 2
TOTAL_EVENT_IN_FILE = 15

EXAMPLE_FILE_PATH_V6 = get_dataset_path("NectarCAM.Run5288.0001.fits.fz")
FIRST_EVENT_NUMBER_IN_FILE_V6 = 3
EVENT_JUMP_V6 = 4
TOTAL_EVENT_IN_FILE_V6 = 13

# Regroupe the infos in order to avoid too much copy/paste in the tests
EXAMPLE_FILE_PATHS = [EXAMPLE_FILE_PATH, EXAMPLE_FILE_PATH_V6]
FIRST_EVENT_NUMBER_IN_FILES = [
    FIRST_EVENT_NUMBER_IN_FILE,
    FIRST_EVENT_NUMBER_IN_FILE_V6,
]
EVENT_JUMPS = [EVENT_JUMP, EVENT_JUMP_V6]
TOTAL_EVENTS = [TOTAL_EVENT_IN_FILE, TOTAL_EVENT_IN_FILE_V6]


def test_loop_over_events():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        for example_file_path, first_event_number, evt_jump in zip(
            EXAMPLE_FILE_PATHS, FIRST_EVENT_NUMBER_IN_FILES, EVENT_JUMPS
        ):
            n_events = 10
            inputfile_reader = event_source(
                input_url=example_file_path, max_events=n_events
            )

            waveform_shape = (N_GAINS, N_PIXELS, N_SAMPLES)
            for i, event in enumerate(inputfile_reader):
                assert event.trigger.tels_with_trigger == [0]
                for telid in event.trigger.tels_with_trigger:
                    # assert event.index.event_id == FIRST_EVENT_NUMBER_IN_FILE + i*2
                    assert event.index.event_id == first_event_number + i * evt_jump
                    assert event.r0.tel[telid].waveform.shape == waveform_shape

            # make sure max_events works
            assert i == n_events - 1


def test_length():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        for example_file_path, total_event in zip(EXAMPLE_FILE_PATHS, TOTAL_EVENTS):
            inputfile_reader = event_source(input_url=example_file_path)
            assert len(inputfile_reader) == total_event
            assert inputfile_reader.get_entries() == total_event


def test_length_with_limit():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    n_events = 10
    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        for example_file_path in EXAMPLE_FILE_PATHS:
            inputfile_reader = event_source(
                input_url=example_file_path, max_events=n_events
            )
            assert len(inputfile_reader) == n_events
            assert inputfile_reader.get_entries() == n_events


def test_identification():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        inputfile_reader = event_source(
            input_url=EXAMPLE_FILE_PATH,
        )
        assert inputfile_reader.pre_v6_data == True


def test_v6_identification():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        inputfile_reader = event_source(
            input_url=EXAMPLE_FILE_PATH_V6,
        )
        assert inputfile_reader.pre_v6_data == False


def test_is_compatible():
    from ctapipe_io_nectarcam import LightNectarCAMEventSource, NectarCAMEventSource

    for example_file_path in EXAMPLE_FILE_PATHS:
        assert NectarCAMEventSource.is_compatible(example_file_path)

    for example_file_path in EXAMPLE_FILE_PATHS:
        assert LightNectarCAMEventSource.is_compatible(example_file_path) == False


def test_factory_for_nectarcam_file():
    from ctapipe.io import EventSource

    for example_file_path in EXAMPLE_FILE_PATHS:
        reader = EventSource(input_url=example_file_path)

        # explicit import after event_source, to test if this
        # package is detected by ctapipe
        from ctapipe_io_nectarcam import NectarCAMEventSource

        assert isinstance(reader, NectarCAMEventSource)


def test_subarray():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        for example_file_path in EXAMPLE_FILE_PATHS:
            n_events = 10
            inputfile_reader = event_source(
                input_url=example_file_path, max_events=n_events
            )
            subarray = inputfile_reader.subarray
            subarray.info()
            subarray.to_table()

            n_camera_pixels = inputfile_reader.subarray.tel[0].camera.geometry.n_pixels
            assert n_camera_pixels == N_PIXELS


def test_time_precision():
    """Make sure UCTS time precision is not lost
    during conversion to astropy Time, even if timestamp is cast to an int."""
    from math import fabs

    from astropy import units as u

    from ctapipe_io_nectarcam import time_from_unix_tai_ns

    tticks1 = int(1659704534052529046)
    delta = 741
    time1 = time_from_unix_tai_ns(tticks1)
    time2 = time_from_unix_tai_ns(tticks1 + delta)
    assert fabs((time2 - time1).to_value(u.ns) - delta) < 1


def test_r1_waveforms():
    from ctapipe_io_nectarcam import (
        BlockNectarCAMEventSource,
        LightNectarCAMEventSource,
        NectarCAMEventSource,
    )

    #    from ctapipe_io_nectarcam import NectarCAMEventSource

    for event_source in [
        NectarCAMEventSource,
        LightNectarCAMEventSource,
        BlockNectarCAMEventSource,
    ]:
        for example_file_path in EXAMPLE_FILE_PATHS:
            # without gain selection
            config = Config(
                dict(
                    NectarCAMEventSource=dict(
                        NectarCAMR0Corrections=dict(
                            select_gain=False,
                        )
                    )
                )
            )

            n_events = 10
            inputfile_reader = event_source(
                input_url=example_file_path, max_events=n_events, config=config
            )

            waveform_shape = (N_GAINS, N_PIXELS, N_SAMPLES)
            for event in inputfile_reader:
                for telid in event.trigger.tels_with_trigger:
                    assert event.r1.tel[telid].waveform.shape == waveform_shape

            # with gain selection
            config = Config(
                dict(
                    NectarCAMEventSource=dict(
                        NectarCAMR0Corrections=dict(
                            select_gain=True,
                        )
                    )
                )
            )

            n_events = 10
            inputfile_reader = event_source(
                input_url=example_file_path, max_events=n_events, config=config
            )

            waveform_shape = (N_PIXELS, N_SAMPLES)
            for event in inputfile_reader:
                for telid in event.trigger.tels_with_trigger:
                    assert event.r1.tel[telid].waveform.shape == waveform_shape


def test_blockdetection():
    from ctapipe_io_nectarcam import BlockNectarCAMEventSource

    assert (
        BlockNectarCAMEventSource.guess_block_size_from_file(EXAMPLE_FILE_PATH_V6) == 4
    )
    assert BlockNectarCAMEventSource.guess_block_size_from_file(EXAMPLE_FILE_PATH) == 2
