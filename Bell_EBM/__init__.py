# Author: Taylor Bell
# Last Update: 2018-12-20
# Description: This is a simple energy balance model that can be used to model planets both with and without atmospheres which assumes solid body rotation. Any Keplerian orbit can be used for the planet, although larger eccentricity orbits will take longer to run due to the greater difficulty in solving Kepler's equation for the eccentric anomaly

__version__ = 1.1
name = "Bell_EBM"

from .Star import Star
from .Planet import Planet
from .StarPlanetSystem import System
