import numpy as np

TIB_DTYPE = np.dtype(
    [
        ("event_counter", np.uint32),
        ("pps_counter", np.uint16),
        ("tenMHz_counter", np.uint32),
        ("stereo_pattern", np.uint8),
        ("masked_trigger", np.uint8),
    ]
).newbyteorder("<")

CDTS_AFTER_37201_DTYPE = np.dtype(
    [
        ("timestamp", np.uint64),
        ("address", np.uint32),
        ("event_counter", np.uint32),
        ("busy_counter", np.uint32),
        ("pps_counter", np.uint32),
        ("clock_counter", np.uint32),
        ("trigger_type", np.uint8),
        ("white_rabbit_status", np.uint8),
        ("stereo_pattern", np.uint8),
        ("num_in_bunch", np.uint8),
        ("cdts_version", np.uint32),
    ]
).newbyteorder("<")

CDTS_BEFORE_37201_DTYPE = np.dtype(
    [
        ("event_counter", np.uint32),
        ("pps_counter", np.uint32),
        ("clock_counter", np.uint32),
        ("timestamp", np.uint64),
        ("camera_timestamp", np.uint64),
        ("trigger_type", np.uint8),
        ("white_rabbit_status", np.uint8),
        ("unknown", np.uint8),  # called arbitraryInformation in C-Struct
    ]
).newbyteorder("<")

SWAT_DTYPE = np.dtype(
    [
        ("timestamp", np.uint64),
        ("counter1", np.uint32),
        ("counter2", np.uint32),
        ("event_type", np.uint8),
        ("camera_flag", np.uint8),
        ("camera_event_num", np.uint32),
        ("array_flag", np.uint8),
        ("event_num", np.uint32),
    ]
).newbyteorder("<")
