"""
    This module contains the Engine class,
    which is the main class of the simulation.

    The Engine class is responsible for running the simulation.
    It is responsible for creating the drivers, and road.
    It is also responsible for running the simulation and
    updating the state of the simulation.

    Attributes
    ----------
    run_config : RunConfig
        The configuration of the simulation.
"""


from typing import List
from lib.Driver import Driver, DriverConfig
from lib.Road import Road
import numpy as np
import scipy.stats as st


def initialize_drivers(population_size: int) -> List[Driver]:
    """
    Initializes the drivers for the simulation.

    Take the population size and obtain a random sample from
    the log-normal distribution. Then create an histogram
    of the sample (5 bins), where each bin represents a
    driver type. Then create the drivers based on the
    histogram.

    Then, the car selection follows a multinomial distribution
    with the following probabilities:
    - 20% of the drivers have a car type 0  (MOTORCYCLE)
    - 40% of the drivers have a car type 1  (SEDAN)
    - 30% of the drivers have a car type 2  (SUV)
    - 10% of the drivers have a car type 3  (TRACK)

    NOTE: Drivers & cars selection might change in the future.

    Parameters
    ----------
    population_size : int
        The number of drivers in the simulation.
    """
    stats = st.lognorm.rvs(0.5, size=population_size)

    # Create the histogram
    hist, _ = np.histogram(stats, bins=5)

    # Create the drivers
    drivers = []

    for i in range(len(hist)):
        for _ in range(hist[i]):
            dconfig = DriverConfig(
                driver_type=i,
                car_type=np.random.choice(
                    a=np.arange(0, 4),
                    p=[.2, .4, .3, .1]
                )
            )
            drivers.append(Driver(config=dconfig))

    return drivers


class RunConfig:
    def __init__(self, **kwargs):
        """
        Constructor for the RunConfig class.

        Parameters
        ----------
        population_size : int
            The number of drivers in the simulation. Defaults to 100.
        road_length : float
            The length of the road in meters. Defaults to 1e4.
        time_steps : int
            Time steps to run the simulation for. Defaults to 1000.
        lanes : int
            The number of lanes on the road. Defaults to 1.
        """

        if 'population_size' in kwargs:
            self.population_size = kwargs['population_size']
        else:
            self.population_size = 100  # default value
        if 'road_length' in kwargs:
            self.road_length = kwargs['road_length']
        else:
            self.road_length = 1e4  # default value --> in meters
        if 'time_steps' in kwargs:
            self.time_steps = kwargs['time_steps']
        else:
            self.time_steps = 1000
        if 'lanes' in kwargs:
            self.lanes = kwargs['lanes']
        else:
            self.lanes = 1


class Model:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.road = Road(self.run_config.road_length, self.run_config.lanes)
        self.drivers = initialize_drivers(self.run_config.population_size)

    @property
    def run_config(self) -> "RunConfig":
        return self.run_config

    @run_config.setter
    def run_config(self, run_config):
        self.run_config = run_config

    @property
    def road(self) -> "Road":
        return self.road

    @road.setter
    def road(self, road):
        self.road = road

    @property
    def drivers(self) -> list:
        return self.drivers

    @drivers.setter
    def drivers(self, drivers):
        self.drivers = drivers


class Engine:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.trace = Trace()

        # Initialize the model
        self.model = Model(self.run_config)

    @property
    def run_config(self) -> "RunConfig":
        return self.run_config

    @run_config.setter
    def run_config(self, run_config):
        self.run_config = run_config

    @property
    def trace(self) -> "Trace":
        return self.trace

    @trace.setter
    def trace(self, trace):
        self.trace = trace

    @property
    def model(self) -> "Model":
        return self.model

    @model.setter
    def model(self, model):
        self.model = model

    def run(self):
        for t in range(self.run_config.time_steps):
            # TODO: update the state of the simulation
            self.trace.add(self.model)  # Record the state of the simulation


class Trace:
    def __init__(self, *args, **kwargs):
        self.data = []

    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, data):
        self.data = data

    def add(self, data):
        self.data.append(data)
