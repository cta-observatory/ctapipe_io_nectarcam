# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.

Needs protozfits v1.5.0 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
import glob
import struct
from astropy import units as u
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraGeometry,
    OpticsDescription,
)
from ctapipe.io import EventSource
from ctapipe.io.containers import PixelStatusContainer
from ctapipe.core.traits import Int
from ctapipe.core import Provenance
from astropy.io import fits
from .containers import NectarCAMDataContainer

__all__ = ['NectarCAMEventSource']


class NectarCAMEventSource(EventSource):
    """
    EventSource for NectarCam r0 data.
    """
    n_gains = Int(
        2,
        help='Number of gains at r0/r1 level'
    ).tag(config=True)

    baseline = Int(
        250,
        help='r0 waveform baseline '
    ).tag(config=True)

    geometry_version = Int(
        3,
        help='Version of the camera geometry to be used '
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
            self.file_list = glob.glob(kwargs['input_url'])
            self.file_list.sort()
            kwargs['input_url'] = self.file_list[0]
            super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
            self.file_list = [self.input_url]

        self.multi_file = MultiFiles(self.file_list)
        self.camera_config = self.multi_file.camera_config
        self.n_camera_pixels = 1855
        self.data = None
        self.log.info("Read {} input files".format(self.multi_file.num_inputs()))


    def subarray(self):
        return self.prepare_subarray_info()


    def prepare_subarray_info(self):
        """
        Constructs a SubarrayDescription object from the
        ``telescope_descriptions`` given by ``SimTelFile``
        Parameters
        ----------
        Returns
        -------
        SubarrayDescription :
            instrumental information
        """
        tel_descriptions = {}  # tel_id : TelescopeDescription
        tel_positions = {}  # tel_id : TelescopeDescription

        for tel_id in self.data.nectarcam.tels_with_data:
            # optics info from standard optics.fits.gz file
            optics = OpticsDescription.from_name("MST")
            optics.tel_subtype = ''  # to correct bug in reading

            # camera info from NectarCam-[geometry_version].camgeom.fits.gz file
            camera = CameraGeometry.from_name("NectarCam", self.geometry_version)

            tel_descr = TelescopeDescription(name='MST', tel_type='NectarCam', optics=optics, camera=camera)
            tel_descr.optics.tel_subtype = ''  # to correct bug in reading

            self.n_camera_pixels = tel_descr.camera.n_pixels

            # MST telescope position
            tel_positions[tel_id] = [0., 0., 0] * u.m
            tel_descriptions[tel_id] = tel_descr

        return SubarrayDescription(
            "Adlershof",
            tel_positions=tel_positions,
            tel_descriptions=tel_descriptions,
        )

    def _generator(self):

        # container for NectarCAM data
        self.data = NectarCAMDataContainer()
        self.data.meta['input_url'] = self.input_url
        self.data.meta['origin'] = 'NectarCAM'

        # fill data from the CameraConfig table
        self.fill_nectarcam_service_container_from_zfile()

        self.data.inst.subarray = self.subarray()

        # initialize general monitoring container
        self.initialize_mon_container()

        # loop on events
        for count, event in enumerate(self.multi_file):

            self.data.count = count

            # fill specific NectarCAM event data
            self.fill_nectarcam_event_container_from_zfile(event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(event)

            # copy r0 to r1
            self.fill_r1_container()

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
            event_container.ucts_event_counter = unpacked_cdts[2]
            event_container.ucts_pps_counter = unpacked_cdts[4]
            event_container.ucts_clock_counter = unpacked_cdts[5]
            event_container.ucts_trigger_type = unpacked_cdts[6]

        # Unpack FEB counters and trigger pattern
        self.unpack_feb_data(event)

    def unpack_feb_data(self, event):
        '''Unpack FEB counters and trigger pattern'''
        event_container = self.data.nectarcam.tel[self.camera_config.telescope_id].evt

        # Deduce data format version
        bytes_per_module = len(event.nectarcam.counters)//self.camera_config.nectarcam.num_modules
        # Remain compatible with data before addition of trigger pattern
        module_fmt = 'IHHIHBBBBBB' if bytes_per_module > 16 else 'IHHIHBB'
        n_fields = len(module_fmt)
        rec_fmt = '=' + module_fmt*self.camera_config.nectarcam.num_modules
        # Unpack
        unpacked_feb =  struct.unpack(rec_fmt, event.nectarcam.counters)
        # Initialize field containers
        n_camera_modules = self.n_camera_pixels//7
        event_container.feb_abs_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
        event_container.feb_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
        event_container.feb_pps_cnt = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
        event_container.feb_ts1 = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
        if bytes_per_module > 16:
            n_patterns = 4
            event_container.trigger_pattern = np.zeros(shape=(n_patterns, self.n_camera_pixels), dtype=bool)

        # Unpack absolute event ID
        event_container.feb_abs_event_id[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[0::n_fields]
        # Unpack relative event ID
        event_container.feb_event_id[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[1::n_fields]
        # Unpack PPS counter
        event_container.feb_pps_cnt[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[2::n_fields]
        # Unpack TS1 counter
        event_container.feb_ts1[self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[3::n_fields]
        # Loop over modules
        for module_idx, module_id in enumerate(self.camera_config.nectarcam.expected_modules_id):
            offset = module_id*7
            if bytes_per_module > 16:
                field_id = 7
                # Decode trigger pattern
                for pattern_id in range(n_patterns):
                    value = unpacked_feb[n_fields*module_idx+field_id+pattern_id]
                    module_pattern = [int(digit) for digit in reversed(bin(value)[2:].zfill(7))]
                    event_container.trigger_pattern[pattern_id, offset:offset+7] = module_pattern

        # Unpack native charge
        event_container.native_charge = np.zeros(shape=(self.n_gains, self.n_camera_pixels), dtype=np.uint16)
        rec_fmt = '=' + 'H'*self.camera_config.num_pixels
        for gain_id in range(self.n_gains):
            unpacked_charge = struct.unpack(rec_fmt, getattr(event.nectarcam, f'charges_gain{gain_id + 1}'))
            event_container.native_charge[gain_id, self.camera_config.expected_pixels_id] = unpacked_charge
        

    def fill_r0_camera_container_from_zfile(self, container, event):

        container.trigger_time = event.trigger_time_s
        #container.trigger_type = event.trigger_type
        container.trigger_type = self.data.nectarcam.tel[self.camera_config.telescope_id].evt.tib_masked_trigger

        # verify the number of gains

        if event.waveform.shape[0] != self.camera_config.num_pixels * self.camera_config.num_samples * self.n_gains:
            raise ValueError(f"Number of gains not correct, waveform shape is {event.waveform.shape[0]}"
                             f" instead of "
                             f"{self.camera_config.num_pixels * self.camera_config.num_samples * self.n_gains}")

        reshaped_waveform = np.array(
            event.waveform
             ).reshape(self.n_gains,
                       self.camera_config.num_pixels,
                       self.camera_config.num_samples)

        # initialize the waveform container to zero
        container.waveform = np.zeros([self.n_gains,
                                       self.n_camera_pixels,
                                       self.camera_config.num_samples])

        # re-order the waveform following the expected_pixels_id values (rank = pixel id)
        container.waveform[:, self.camera_config.expected_pixels_id, :] \
            = reshaped_waveform

    def fill_r0_container_from_zfile(self, event):
        """fill the event r0 container"""

        container = self.data.r0
        container.obs_id = -1
        container.event_id = event.event_id

        container.tels_with_data = [self.camera_config.telescope_id, ]
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
        self.data.r1.tels_with_data = [self.camera_config.telescope_id, ]

        r1_camera_container = self.data.r1.tel[self.camera_config.telescope_id]
        r1_camera_container.waveform = self.data.r0.tel[self.camera_config.telescope_id].waveform - self.baseline
        r1_camera_container.trigger_type = self.data.r0.tel[self.camera_config.telescope_id].trigger_type
        r1_camera_container.trigger_time = self.data.r0.tel[self.camera_config.telescope_id].trigger_time

    def initialize_mon_container(self):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = self.data.mon
        container.tels_with_data = [self.camera_config.telescope_id, ]
        mon_camera_container = container.tel[self.camera_config.telescope_id]

        # initialize the container
        status_container = PixelStatusContainer()
        status_container.hardware_failing_pixels = np.zeros((self.n_gains, self.n_camera_pixels), dtype=bool)
        status_container.pedestal_failing_pixels = np.zeros((self.n_gains, self.n_camera_pixels), dtype=bool)
        status_container.flatfield_failing_pixels = np.zeros((self.n_gains, self.n_camera_pixels), dtype=bool)

        mon_camera_container.pixel_status = status_container

    def fill_mon_container_from_zfile(self, event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """

        status_container = self.data.mon.tel[self.camera_config.telescope_id].pixel_status

        # reorder the array
        pixel_status = np.zeros(self.n_camera_pixels)
        pixel_status[self.camera_config.expected_pixels_id] = event.pixel_status
        status_container.hardware_failing_pixels[:] = pixel_status == 0
        '''
        for gain in(np.arange(self.n_gains)):
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
