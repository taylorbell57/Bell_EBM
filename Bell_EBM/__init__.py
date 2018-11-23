# Author: Taylor Bell
# Last Update: 2018-11-20
# Description: This is a simple energy balance model that can be used to model planets both with and without atmospheres which assumes solid body rotation. Any Keplerian orbit can be used for the planet, although larger eccentricity orbits will take longer to run due to the greater difficulty in solving Kepler's equation for the eccentric anomaly

# Make plots look pretty
import matplotlib
preamble = [
    r'\usepackage{fontspec}',
    r'\setmainfont{Linux Libertine O}',
]
fontsize = 17
ticklen = 5
params = {
    'font.size': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.labelsize': fontsize,
    'legend.fontsize': fontsize,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': preamble,
    'xtick.major.size' : ticklen,
    'ytick.major.size' : ticklen,
    'xtick.minor.size' : ticklen/2,
    'ytick.minor.size' : ticklen/2
}
matplotlib.rcParams.update(params)

from .Star import Star
from .Planet import Planet
from .StarPlanetSystem import System
