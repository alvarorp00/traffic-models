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

POPULATION_SIZE = 100

TIME_STEPS = 1000

ROAD_LENGTH = 1e4

MAX_SPEED = 130  # In km/h

LANES = 2

LANES_PRIORITY = np.array([.7, .3])  # From right to left
