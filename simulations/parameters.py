from dataclasses import dataclass
from typing import Optional

# FIXED FIBER PARAMETERS.
# Propagation delay to be considered in the analyses of the neural responses - heavily depends on FIBERS_LENGTH_UM
FIBERS_LENGTH_UM = 40 * 10000  # 40 cm
PROPAGATION_DELAY_MS = 6
FASCICLE_RADIUS_UM = 100
MEAN_DIAMETER_UM = 10.
STD_DIAMETER = 2.


@dataclass
class FiberParameters(object):
    """
    Holds the parameters of the modelled fiber population.
    """
    n_fibers: int
    length_um: float
    mean_diameter_um: float
    std_diameter: float
    # Propagation delay to be considered in the analyses of the neural responses.
    # This heavily depends on length_um below
    propagation_delay_ms: float
    fascicle_radius_um: float


@dataclass
class StimParameters(object):
    """
    Holds the parameters of the modelled fiber population.
    """
    frequency: float  # in Hz
    pulse_width_ms: float
    amplitude_ma: float
    bios: bool
    burst_frequency: Optional[float] = None
    burst_duration_ms: Optional[float] = None
    min_amplitude_ma: Optional[float] = None
