"""
This file contains the configuration parameters for the simulation.

Parameters
----------
POPULATION_SIZE : int
    The number of drivers in the simulation.

ROAD_LENGTH : float
    The length of the road in meters.

TIME_STEPS : int
    Time steps to run the simulation for.

LANES : int
    The number of lanes on the road.
"""

import numpy as np
from lib.driver import LanePriority

POPULATION_SIZE = 25

TIME_STEPS = 1000

ROAD_LENGTH = 1e3  # In meters

MAX_SPEED = 130  # In km/h

N_LANES = 3

LANES_PRIORITY = LanePriority.LEFT

LANES_DENSITY = np.array([0.6, 0.3, 0.1])

SAFE_DISTANCE = 4.  # In meters
