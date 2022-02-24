# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for NectarCAM protobuf-fits.fz-files.

Needs protozfits v1.5.0 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
import glob
import struct
from astropy import units as u
from pkg_resources import resource_filename
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraDescription,
    CameraReadout,
    CameraGeometry,
    OpticsDescription,
)
from enum import IntFlag, auto
from ctapipe.coordinates import CameraFrame
from ctapipe.io import EventSource
from ctapipe.containers import (
    PixelStatusContainer,
    EventType,
)
from ctapipe.core.traits import Int, Bool, Enum
from ctapipe.core import Provenance
from astropy.io import fits
from .containers import NectarCAMDataContainer
from .constants import (
    HIGH_GAIN, N_GAINS, N_PIXELS, N_SAMPLES
)

__all__ = ['NectarCAMEventSource']

class TriggerBits(IntFlag):
    '''
    See TIB User manual
    '''
    UNKNOWN = 0
    MONO = auto()
    STEREO = auto()
    CALIBRATION = auto()
    SINGLE_PE = auto()
    SOFTWARE = auto()
    PEDESTAL = auto()
    SLOW_CONTROL = auto()

    PHYSICS = MONO | STEREO
    OTHER = CALIBRATION | SINGLE_PE | SOFTWARE | PEDESTAL | SLOW_CONTROL

OPTICS = OpticsDescription(
    # https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/verification/verification-process/mst-structure/-/blob/master/Appendix-MST-Structure.pdf
    # version from 20 Jan 2022
    'MST',
    equivalent_focal_length=u.Quantity(16., u.m),
    num_mirrors=1,
    mirror_area=u.Quantity(106., u.m**2), #no shadowing, uncertainty is 0.5 m2
    num_mirror_tiles=86, # Garczarczyk 2017
)

def load_camera_geometry(version=3):
    ''' Load camera geometry from bundled resources of this repo '''
    f = resource_filename(
        'ctapipe_io_nectarcam', f'resources/NectarCam-{version:03d}.camgeom.fits.gz'
    )
    Provenance().add_input_file(f, role="CameraGeometry")
    geom = CameraGeometry.from_table(f)
    geom.frame = CameraFrame(focal_length=OPTICS.equivalent_focal_length)

    return geom

def read_pulse_shapes():

    '''
    Reads in the data on the pulse shapes from an external file
    Returns
    -------
    (daq_time_per_sample, pulse_shape_time_step, pulse shapes)
        daq_time_per_sample: time between samples in the actual DAQ (ns, astropy quantity)
        pulse_shape_time_step: time between samples in the returned single-p.e pulse shape (ns, astropy
    quantity)
        pulse shapes: Single-p.e. pulse shapes, ndarray of shape (2, 1640)
    '''

    # https://gitlab.cta-observatory.org/cta-consortium/aswg/simulations/simulation-model/simulation-model-description/-/blob/master/datFiles/Pulse_template_nectarCam_17042020.dat
    infilename = resource_filename(
        'ctapipe_io_nectarcam',
        'resources/Pulse_template_nectarCam_17042020.dat'
    )

    data = np.genfromtxt(infilename, dtype='float', comments='#')
    Provenance().add_input_file(infilename, role="PulseShapes")
    pulse_shape_time_step = 0.125 * u.ns # file specific, change if model file is changed
    # TODO read automatically from file

    # https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/verification/verification-process/mst-nectarcam/-/blob/master/Appendix-NectarCam.pdf
    # version from 13 Jan 2022
    daq_time_per_sample = 1. * u.ns

    # Note we have to transpose the pulse shapes array to provide what ctapipe
    # expects:
    return daq_time_per_sample, pulse_shape_time_step, data[:,1:].T


class NectarCAMEventSource(EventSource):
    """
    EventSource for NectarCam r0 data.
    """

    baseline = Int(
        250,
        help='r0 waveform baseline '
    ).tag(config=True)

    trigger_information = Bool(
        default_value=True,
        help='Fill trigger information.'
    ).tag(config=True)

    default_trigger_type = Enum(
        ['ucts', 'tib'], default_value='ucts',
        help=(
            'Default source for trigger type information.'
        )
    ).tag(config=True)

    def __init__(self, **kwargs):
        """
        Constructor
        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool: ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """
        # EventSource can not handle file wild cards as input_url
        # To overcome this we substitute the input_url with first file matching
        # the specified file mask (copied from  MAGICEventSourceROOT).

        if 'input_url' in kwargs.keys():
            self.file_list = glob.glob(str(kwargs['input_url']))
            self.file_list.sort()
            kwargs['input_url'] = self.file_list[0]
            super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
            self.file_list = [self.input_url]

        self.multi_file = MultiFiles(self.file_list)
        self.camera_config = self.multi_file.camera_config
        self.data = None
        self.log.info("Read {} input files".format(self.multi_file.num_inputs()))
        self._tel_id = self.camera_config.telescope_id
        self.geometry_version = 3
        self._subarray = self.create_subarray(self.geometry_version, self._tel_id)

    @property
    def is_simulation(self):
        return False

    @property
    def datalevels(self):
        return (DataLevel.R0,)

    @property
    def subarray(self):
        return self._subarray

    @staticmethod
    def create_subarray(geometry_version, tel_id=0):
        """
        Obtain the subarray from the EventSource
        Returns
        -------
        ctapipe.instrument.SubarrayDescription
        """

        # camera info from NectarCAM-[geometry_version].camgeom.fits.gz file
        camera_geom = load_camera_geometry(version=geometry_version)

        # get info on the camera readout:
        daq_time_per_sample, pulse_shape_time_step, pulse_shapes = read_pulse_shapes()

        camera_readout = CameraReadout(
            'NectarCam',
            1 / daq_time_per_sample,
            pulse_shapes,
            pulse_shape_time_step,
            )

        camera = CameraDescription('NectarCam', camera_geom, camera_readout)

        mst_tel_descr = TelescopeDescription(
            name='NectarCam', tel_type='MST', optics=OPTICS, camera=camera
        )

        tel_descriptions = {tel_id: mst_tel_descr}

        # MST telescope position
        tel_positions = {tel_id: [0., 0., 0] * u.m}

        subarray = SubarrayDescription(
            name=f"MST-{tel_id} subarray",
            tel_descriptions=tel_descriptions,
            tel_positions=tel_positions,
        )

        return subarray

    @property
    def obs_ids(self):
        # currently no obs id is available from the input files
        return [self.camera_config.configuration_id, ]


    def _generator(self):

        # container for NectarCAM data
        self.data = NectarCAMDataContainer()
        self.data.meta['input_url'] = self.input_url
        self.data.meta['origin'] = 'NectarCAM'

        # fill data from the CameraConfig table
        self.fill_nectarcam_service_container_from_zfile()

        # initialize general monitoring container
        self.initialize_mon_container(self.data)

        # loop on events
        for count, event in enumerate(self.multi_file):

            self.data.count = count

            # fill specific NectarCAM event data
            self.fill_nectarcam_event_container_from_zfile(event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(event)

            # copy r0 to r1
            # current method does not yield actual r1 data
            # self.fill_r1_container()

            if self.trigger_information:
                self.fill_trigger_info(self.data)

            # fill general monitoring data
            self.fill_mon_container_from_zfile(event)

            yield self.data

    @staticmethod
    def is_compatible(file_path):
        try:
            # The file contains two tables:
            #  1: CameraConfig
            #  2: Events
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_nectarcam_file = 'nectarcam_counters' in ttypes
        return is_protobuf_zfits_file & is_nectarcam_file

    def fill_nectarcam_service_container_from_zfile(self):

        self.data.nectarcam.tels_with_data = [self.camera_config.telescope_id, ]
        svc_container = self.data.nectarcam.tel[self.camera_config.telescope_id].svc

        svc_container.telescope_id = self.camera_config.telescope_id
        svc_container.cs_serial = self.camera_config.cs_serial
        svc_container.configuration_id = self.camera_config.configuration_id
        svc_container.acquisition_mode = self.camera_config.nectarcam.acquisition_mode
        svc_container.date = self.camera_config.date
        svc_container.num_pixels = self.camera_config.num_pixels
        svc_container.num_samples = self.camera_config.num_samples
        svc_container.pixel_ids = self.camera_config.expected_pixels_id
        svc_container.data_model_version = self.camera_config.data_model_version

        svc_container.num_modules = self.camera_config.nectarcam.num_modules
        svc_container.module_ids = self.camera_config.nectarcam.expected_modules_id
        svc_container.idaq_version = self.camera_config.nectarcam.idaq_version
        svc_container.cdhs_version = self.camera_config.nectarcam.cdhs_version
        svc_container.algorithms = self.camera_config.nectarcam.algorithms
        # svc_container.pre_proc_algorithms = camera_config.nectarcam.pre_proc_algorithms

    def fill_nectarcam_event_container_from_zfile(self, event):
        event_container = self.data.nectarcam.tel[self.camera_config.telescope_id].evt

        event_container.configuration_id = event.configuration_id
        event_container.event_id = event.event_id
        event_container.tel_event_id = event.tel_event_id
        event_container.pixel_status = event.pixel_status
        event_container.ped_id = event.ped_id
        event_container.module_status = event.nectarcam.module_status
        event_container.extdevices_presence = event.nectarcam.extdevices_presence
        #event_container.tib_data = event.nectarcam.tib_data
        #event_container.cdts_data = event.nectarcam.cdts_data
        event_container.swat_data = event.nectarcam.swat_data
        event_container.counters = event.nectarcam.counters

        # unpack TIB data
        rec_fmt = '=IHIBB'
        unpacked_tib = struct.unpack(rec_fmt, event.nectarcam.tib_data)
        event_container.tib_event_counter = unpacked_tib[0]
        event_container.tib_pps_counter = unpacked_tib[1]
        event_container.tib_tenMHz_counter = unpacked_tib[2]
        event_container.tib_stereo_pattern = unpacked_tib[3]
        event_container.tib_masked_trigger = unpacked_tib[4]
        event_container.swat_data = event.lstcam.swat_data

        # unpack CDTS data
        is_old_cdts = len(event.nectarcam.cdts_data) < 36
        rec_fmt = '=IIIQQBBB' if is_old_cdts else '=QIIIIIBBBBI'
        unpacked_cdts =  struct.unpack(rec_fmt, event.nectarcam.cdts_data)
        if is_old_cdts:
            event_container.ucts_event_counter = unpacked_cdts[0]
            event_container.ucts_pps_counter = unpacked_cdts[1]
            event_container.ucts_clock_counter = unpacked_cdts[2]
            event_container.ucts_timestamp = unpacked_cdts[3]
            event_container.ucts_camera_timestamp = unpacked_cdts[4]
            event_container.ucts_trigger_type = unpacked_cdts[5]
            event_container.ucts_white_rabbit_status = unpacked_cdts[6]
        else:
            event_container.ucts_timestamp = unpacked_cdts[0]
            event_container.ucts_address = unpacked_cdts[1]
            event_container.ucts_event_counter = unpacked_cdts[2]
            event_container.ucts_busy_counter = unpacked_cdts[3]
            event_container.ucts_pps_counter = unpacked_cdts[4]
            event_container.ucts_clock_counter = unpacked_cdts[5]
            event_container.ucts_trigger_type = unpacked_cdts[6]
            event_container.ucts_white_rabbit_status = unpacked_cdts[7]
            event_container.ucts_stereo_pattern = unpacked_cdts[8]
            event_container.ucts_num_in_bunch = unpacked_cdts[9]
            event_container.cdts_version = unpacked_cdts[10]

        # Unpack FEB counters and trigger pattern
        self.unpack_feb_data(event)

    def fill_trigger_info(self, array_event):
        tel_id = self._tel_id

        nectarcam = array_event.nectarcam.tel[tel_id]
        tib_available = nectarcam.evt.extdevices_presence & 1
        ucts_available = nectarcam.evt.extdevices_presence & 2

        # fill trigger time using UCTS timestamp
        trigger = array_event.trigger
        trigger.time = nectarcam.evt.ucts_timestamp
        trigger.tels_with_trigger = [tel_id]
        trigger.tel[tel_id].time = trigger.time

        # decide which source to use, if both are available,
        # the option decides, if not, fallback to the avilable source
        # if no source available, warn and do not fill trigger info
        if tib_available and ucts_available:
            if self.default_trigger_type == 'ucts':
                trigger_bits = nectarcam.evt.ucts_trigger_type
            else:
                trigger_bits = nectarcam.evt.tib_masked_trigger

        elif tib_available:
            trigger_bits = nectarcam.evt.tib_masked_trigger

        elif ucts_available:
            trigger_bits = nectarcam.evt.ucts_trigger_type

        else:
            self.log.warning('No trigger info available.')
            trigger.event_type = EventType.UNKNOWN
            return

        if (
                ucts_available
                and nectarcam.evt.ucts_trigger_type == 42 #TODO check if it's correct
                and self.default_trigger_type == "ucts"
        ) :
            self.log.warning(
                'Event with UCTS trigger_type 42 found.'
                ' Probably means unreliable or shifted UCTS data.'
                ' Consider switching to TIB using `default_trigger_type="tib"`'
            )

        # first bit mono trigger, second stereo.
        # If *only* those two are set, we assume it's a physics event
        # for all other we only check if the flag is present
        if (trigger_bits & TriggerBits.PHYSICS) and not (trigger_bits & TriggerBits.OTHER):
            trigger.event_type = EventType.SUBARRAY
        elif trigger_bits & TriggerBits.CALIBRATION:
            trigger.event_type = EventType.FLATFIELD
        elif trigger_bits & TriggerBits.PEDESTAL:
            trigger.event_type = EventType.SKY_PEDESTAL
        elif trigger_bits & TriggerBits.SINGLE_PE:
            trigger.event_type = EventType.SINGLE_PE
        else:
            self.log.warning(f'Event {array_event.index.event_id} has unknown event type, trigger: {trigger_bits:08b}')
            trigger.event_type = EventType.UNKNOWN

    def unpack_feb_data(self, event):
        '''Unpack FEB counters and trigger pattern'''
        event_container = self.data.nectarcam.tel[self.camera_config.telescope_id].evt

        # Deduce data format version
        bytes_per_module = len(event.nectarcam.counters)//self.camera_config.nectarcam.num_modules
        # Remain compatible with data before addition of trigger pattern
        module_fmt = 'IHHIBBBBBBBB' if bytes_per_module > 16 else 'IHHIBBBB'
        n_fields = len(module_fmt)
        rec_fmt = '=' + module_fmt*self.camera_config.nectarcam.num_modules
        # Unpack
        unpacked_feb =  struct.unpack(rec_fmt, event.nectarcam.counters)
        # Initialize field containers
        n_camera_modules = N_PIXELS//7
        event_container.feb_abs_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
        event_container.feb_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
        event_container.feb_pps_cnt = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
        event_container.feb_ts1 = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
        event_container.feb_ts2_trig  = np.zeros(shape=(n_camera_modules,), dtype=np.int16)
        event_container.feb_ts2_pps  = np.zeros(shape=(n_camera_modules,), dtype=np.int16)
        if bytes_per_module > 16:
            n_patterns = 4
            event_container.trigger_pattern = np.zeros(shape=(n_patterns, N_PIXELS), dtype=bool)

        # Unpack absolute event ID
        event_container.feb_abs_event_id[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[0::n_fields]
        # Unpack PPS counter
        event_container.feb_pps_cnt[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[1::n_fields]
        # Unpack relative event ID
        event_container.feb_event_id[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[2::n_fields]
        # Unpack TS1 counter
        event_container.feb_ts1[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[3::n_fields]
        # Unpack TS2 counters
        ts2_decimal = lambda bits: bits - (1 << 8) if bits & 0x80 != 0 else bits
        ts2_decimal_vec = np.vectorize(ts2_decimal)
        event_container.feb_ts2_trig[self.camera_config.nectarcam.expected_modules_id] = ts2_decimal_vec(unpacked_feb[4::n_fields])
        event_container.feb_ts2_pps[self.camera_config.nectarcam.expected_modules_id] = ts2_decimal_vec(unpacked_feb[5::n_fields])
        # Loop over modules
        for module_idx, module_id in enumerate(self.camera_config.nectarcam.expected_modules_id):
            offset = module_id*7
            if bytes_per_module > 16:
                field_id = 8
                # Decode trigger pattern
                for pattern_id in range(n_patterns):
                    value = unpacked_feb[n_fields*module_idx+field_id+pattern_id]
                    module_pattern = [int(digit) for digit in reversed(bin(value)[2:].zfill(7))]
                    event_container.trigger_pattern[pattern_id, offset:offset+7] = module_pattern

        # Unpack native charge
        if len(event.nectarcam.charges_gain1) > 0:
            event_container.native_charge = np.zeros(shape=(N_GAINS, N_PIXELS), dtype=np.uint16)
            rec_fmt = '=' + 'H'*self.camera_config.num_pixels
            for gain_id in range(N_GAINS):
                unpacked_charge = struct.unpack(rec_fmt, getattr(event.nectarcam, f'charges_gain{gain_id + 1}'))
                event_container.native_charge[gain_id, self.camera_config.expected_pixels_id] = unpacked_charge
        

    def fill_r0_camera_container_from_zfile(self, container, event):

        self.data.trigger.time = event.trigger_time_s
        self.data.trigger.tels_with_trigger = [self._tel_id]
        #container.trigger_type = event.trigger_type
        self.data.trigger.event_type = self.data.nectarcam.tel[self.camera_config.telescope_id].evt.tib_masked_trigger

        # verify the number of gains

        if event.waveform.shape[0] != self.camera_config.num_pixels * self.camera_config.num_samples * N_GAINS:
            raise ValueError(f"Number of gains not correct, waveform shape is {event.waveform.shape[0]}"
                             f" instead of "
                             f"{self.camera_config.num_pixels * self.camera_config.num_samples * N_GAINS}")

        reshaped_waveform = np.array(
            event.waveform
             ).reshape(N_GAINS,
                       self.camera_config.num_pixels,
                       self.camera_config.num_samples)

        # initialize the waveform container to zero
        container.waveform = np.zeros([N_GAINS,
                                       N_PIXELS,
                                       self.camera_config.num_samples])

        # re-order the waveform following the expected_pixels_id values (rank = pixel id)
        container.waveform[:, self.camera_config.expected_pixels_id, :] \
            = reshaped_waveform

    def fill_r0_container_from_zfile(self, event):
        """fill the event r0 container"""
        self.data.index.obs_id = self.obs_ids[0]
        self.data.index.event_id = event.event_id

        container = self.data.r0

        r0_camera_container = container.tel[self.camera_config.telescope_id]

        self.fill_r0_camera_container_from_zfile(
            r0_camera_container,
            event
        )

    def fill_r1_container(self):
        """
           fill the event r1 container
           In the case of nectarCAM:
           r1 waveform = r0 waveform - self.baseline

        """

        r1_camera_container = self.data.r1.tel[self.camera_config.telescope_id]
        r1_camera_container.waveform = self.data.r0.tel[self.camera_config.telescope_id].waveform - self.baseline
        #r1_camera_container.trigger_type = self.data.r0.tel[self.camera_config.telescope_id].trigger_type
        #r1_camera_container.trigger_time = self.data.r0.tel[self.camera_config.telescope_id].trigger_time

    def initialize_mon_container(self, array_event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = array_event.mon
        mon_camera_container = container.tel[self._tel_id]

        # initialize the container
        status_container = PixelStatusContainer()

        shape = (N_GAINS, N_PIXELS)
        status_container.hardware_failing_pixels = np.zeros(shape, dtype=bool)
        status_container.pedestal_failing_pixels = np.zeros(shape, dtype=bool)
        status_container.flatfield_failing_pixels = np.zeros(shape, dtype=bool)

        mon_camera_container.pixel_status = status_container

    def fill_mon_container_from_zfile(self, event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """

        status_container = self.data.mon.tel[self.camera_config.telescope_id].pixel_status

        # reorder the array
        pixel_status = np.zeros(N_PIXELS)
        pixel_status[self.camera_config.expected_pixels_id] = event.pixel_status
        status_container.hardware_failing_pixels[:] = pixel_status == 0
        '''
        for gain in(np.arange(N_GAINS)):
             pixel_status[self.camera_config.expected_pixels_id] = event.pixel_status

             # initialize the hardware mask
             status_container.hardware_failing_pixels[gain] = pixel_status == 0
        '''


class MultiFiles:

    """
    This class open all the files in file_list and read the events following
    the event_id order
    """

    def __init__(self, file_list):

        self._file = {}
        self._events = {}
        self._events_table = {}
        self._camera_config = {}
        self.camera_config = None


        paths = []
        for file_name in file_list:
            paths.append(file_name)
            Provenance().add_input_file(file_name, role='r0.sub.evt')

        # open the files and get the first fits Tables
        from protozfits import File

        for path in paths:

            try:
                self._file[path] = File(path)
                self._events_table[path] = File(path).Events
                self._events[path] = next(self._file[path].Events)

                # verify where the CameraConfig is present
                if 'CameraConfig' in self._file[path].__dict__.keys():
                    self._camera_config[path] = next(self._file[path].CameraConfig)

                # for the moment it takes the first CameraConfig it finds (to be changed)
                    if(self.camera_config is None):
                        self.camera_config = self._camera_config[path]


            except StopIteration:
                pass

        # verify that somewhere the CameraConfing is present
        assert (self.camera_config)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_event()

    def next_event(self):
        # check for the minimal event id
        if not self._events:
            raise StopIteration

        min_path = min(
            self._events.items(),
            key=lambda item: item[1].event_id,
        )[0]

        # return the minimal event id
        next_event = self._events[min_path]
        try:
            self._events[min_path] = next(self._file[min_path].Events)
        except StopIteration:
            del self._events[min_path]

        return next_event

    def __len__(self):
        total_length = sum(
            len(table)
            for table in self._events_table.values()
        )
        return total_length

    def num_inputs(self):
        return len(self._file)
