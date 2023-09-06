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
    N_LANES = 2

    TIME_STEPS = 100

    POPULATION_SIZE = 50

    MINIMUM_LOAD_FACTOR = 0.25

    START_WITH_POPULATION = False

    SPAWN_EVERY_N_STEPS = 100

    ROAD_LENGTH = 500  # In meters

    SECTION_LENGTH = 100  # In meters

    ACCIDENTS = True

    ACCIDENT_CLEARANCE_TIME = 15  # In seconds

    # Percentage threshold to validate a simulation
    ACCIDENT_MAX_THRESHOLD = 0.05

    # Cars max speed
    CARS_MAX_SPEED = [80, 100, 120, 130]

    # Cars min speed
    CARS_MIN_SPEED = [40, 65, 75, 90]

    # Modifiers for max speed
    SPEED_MODIFIERS = [0.8, 1.0, 1.1, 1.2, 1.3]

    # Just visual representation while plotting
    LANES_PRIORITY = LanePriority.LEFT

    # From lesser priority to higher priority
    LANES_DENSITY = np.array([0.85, 0.15])

    # From more cautious to more aggressive
    DRIVER_TYPE_DENSITY = [.4, .3, .15, .1, .05]
    # DRIVER_TYPE_DENSITY = [.05, .1, .15, .3, .4]

    # From more quick to more slow
    DRIVER_REACTION_DENSITY = [.075, .25, .35, .25, .075]

    # Car types
    CAR_TYPE_DENSITY = [.10, .15, .55, .20]

    # Safe distance
    SAFE_DISTANCE = 25.0  # In meters

    # Multiplier for the safe distance for each driver type
    SAFE_DISTANCE_FACTOR = [1.5, 1.25, 1.0, 0.75, 0.5]

    # View distance
    VIEW_DISTANCE = 100.0  # In meters

    # Multiplier for the view distance for each driver type
    VIEW_DISTANCE_FACTOR = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Overtaking time
    TIME_IN_LANE = 10

    # Multiplier for the overtaking time for each driver type
    TIME_IN_LANE_FACTOR = [1.0, 1.1, 1.2, 1.3, 1.4]

    # Print info
    VERBOSE = False

    # Plot the simulation
    PLOT = False
