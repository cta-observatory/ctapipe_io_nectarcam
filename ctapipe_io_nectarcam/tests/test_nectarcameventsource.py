from ctapipe.utils import get_dataset_path

FIRST_EVENT_NUMBER_IN_FILE = 1
example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz")


def test_loop_over_events():
    from ctapipe_io_nectarcam import NectarCAMEventSource

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=n_events
    )

    for i, event in enumerate(inputfile_reader):
        assert event.trigger.tels_with_trigger == [0]
        for telid in event.trigger.tels_with_trigger:
            assert event.index.event_id == FIRST_EVENT_NUMBER_IN_FILE + i
            n_gain = 2
            n_camera_pixels = inputfile_reader.subarray.tel[0].camera.geometry.n_pixels
            num_samples = event.nectarcam.tel[telid].svc.num_samples
            waveform_shape = (n_gain, n_camera_pixels, num_samples)
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

