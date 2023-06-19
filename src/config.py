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


class Config:
    TIME_STEPS = 100

    POPULATION_SIZE = 15

    MINIMUM_LOAD_FACTOR = 1.00

    START_WITH_POPULATION = False

    SPAWN_EVERY_N_STEPS = 10

    ROAD_LENGTH = 10000  # In meters

    SECTION_LENGTH = 100  # In meters

    ACCIDENTS = False  # division by zero, check this # TODO

    ACCIDENT_CLEARANCE_TIME = 15  # In seconds

    # In km/h
    MAX_SPEED = 130
    # Gap between max speed of cars of different types (consecutively)
    MAX_SPEED_GAP = 10

    MIN_SPEED = 60
    # Gap between min speed of cars of different types (consecutively)
    MIN_SPEED_GAP = 5

    N_LANES = 3

    # Just visual representation while plotting
    LANES_PRIORITY = LanePriority.LEFT

    # From lesser priority to higher priority
    LANES_DENSITY = np.array([0.6, 0.3, 0.1])

    # From more cautious to more aggressive
    DRIVER_TYPE_DENSITY = [.4, .3, .15, .1, .05]

    # From more quick to more slow
    DRIVER_REACTION_DENSITY = [.45, .3, .15, .1]

    # Car types
    CAR_TYPE_DENSITY = [.4, .3, .2, .1]

    # Minimum safe distance between two cars for the most aggressive driver
    SAFE_DISTANCE = 4.  # In meters

    # Print info
    VERBOSE = False
