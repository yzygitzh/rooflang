"""rooflang.language.hardware — hardware component model.

Currently re-exports from spec.py (the legacy HardwareSpec). Will be
decomposed into component classes (CPU, GPU, NIC, Switch, Fabric) in
subsequent commits.
"""

from rooflang.language.hardware.spec import (
    HardwareSpec,
    LinkSpec,
    InterNodeSpec,
    HW_B300,
    hardware_spec,
    collective_bytes,
    collective_time,
)
