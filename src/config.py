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

# import numpy as np
from lib.driver import LanePriority

POPULATION_SIZE = 100

TIME_STEPS = 1000

ROAD_LENGTH = 1e4

MAX_SPEED = 130  # In km/h

LANES = 2

LANES_PRIORITY = LanePriority.LEFT

LANES_DENSITY = [0.8, 0.2]

SAFE_DISTANCE = 4.  # In meters
