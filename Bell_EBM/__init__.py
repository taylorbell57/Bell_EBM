# Author: Taylor Bell
# Last Update: 2024-02-22
# Description: This is a simple energy balance model that can be used to model planets both with and without atmospheres which assumes solid body rotation. Any Keplerian orbit can be used for the planet, although larger eccentricity orbits will take longer to run due to the greater difficulty in solving Kepler's equation for the eccentric anomaly

import os
try:
    from .__version__ import __version__
except ModuleNotFoundError:
    from setuptools_scm import get_version
    __version__ = get_version(root=f'..{os.sep}..{os.sep}',
                              relative_to=__file__)

from .Star import Star
from .Planet import Planet
from .StarPlanetSystem import System
