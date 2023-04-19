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


from typing import Dict
from lib.Driver import Driver, DriverConfig, CarType,\
    DriverDistributions, DriverType
from lib.Road import Road
import numpy as np
import scipy.stats as st


def initialize_drivers(population_size: int) -> Dict[int, Driver]:
    """
    Initializes the drivers for the simulation.

    Take the population size and obtain a random sample from
    the log-normal distribution. Then create an histogram
    of the sample (5 bins), where each bin represents a
    driver type. Then create the drivers based on the
    histogram.

    The id assigned to the drivers is the index of the
    driver in the list of drivers. They are unique and are
    assigned in the order of creation.

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
    drivers = {}

    id = 0
    for i in range(len(hist)):
        for _ in range(hist[i]):
            car_type = CarType.random()[0]  # random car type list
            dconfig = DriverConfig(
                id=id,
                driver_type=DriverType(i + 1),
                car_type=car_type,
                speed=DriverDistributions.speed_initialize(
                    car_type=car_type,
                    size=1
                )
            )
            drivers[id] = Driver(config=dconfig)
            id += 1

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
            assert isinstance(kwargs['population_size'], int)
            self.population_size = kwargs['population_size']
        else:
            self.population_size = 100  # default value
        if 'road_length' in kwargs:
            assert isinstance(kwargs['road_length'], float) or \
                        isinstance(kwargs['road_length'], int)
            self.road_length = kwargs['road_length']
        else:
            self.road_length = 1e4  # default value --> in meters
        if 'time_steps' in kwargs:
            assert isinstance(kwargs['time_steps'], int)
            self.time_steps = kwargs['time_steps']
        else:
            self.time_steps = 1000
        if 'lanes' in kwargs:
            assert isinstance(kwargs['lanes'], int)
            self.lanes = kwargs['lanes']
        else:
            self.lanes = 1
        if 'lane_priority' in kwargs:
            assert len(kwargs['lane_priority']) == self.lanes
            assert isinstance(kwargs['lane_priority'], np.ndarray)
            self.lane_priority = kwargs['lane_priority']
        else:
            self.lane_priority = np.array([1])  # By default just one lane
        if 'max_speed' in kwargs:
            assert isinstance(kwargs['max_speed'], float) or \
                     isinstance(kwargs['max_speed'], int)
            self.max_speed = kwargs['max_speed']
        else:
            self.max_speed = 120  # In km/h


class Model:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

        self.info = {}

        self.info['drivers'] = initialize_drivers(
            population_size=self.run_config.population_size
        )

        # Change the location of the drivers, speed and lane
        for d in self.info['drivers'].values():
            d.config.location = np.random.uniform(
                low=0,
                high=self.run_config.road_length
            )
            d.config.speed = np.random.uniform(
                low=0,
                high=30
            )
            d.config.lane = np.random.choice(
                a=np.arange(0, self.run_config.lanes),
                p=self.run_config.lane_priority
            )

        self.info['road'] = Road(
            length=self.run_config.road_length,
            lanes=self.run_config.lanes,
            max_speed=self.run_config.max_speed
        )

        self.info['locations'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.location for d in self.info['drivers'].values()]
        ))

        self.info['speeds'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.speed for d in self.info['drivers'].values()]
        ))

        self.info['lanes'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.lane for d in self.info['drivers'].values()]
        ))

    @property
    def run_config(self) -> "RunConfig":
        return self._run_config

    @run_config.setter
    def run_config(self, run_config: "RunConfig"):
        self._run_config = run_config

    @property
    def info(self) -> Dict:
        return self._info

    @info.setter
    def info(self, info: Dict):
        self._info = info

    @property
    def road(self) -> "Road":
        return self.info['road']

    @property
    def drivers(self) -> Dict[int, "Driver"]:
        return self.info['drivers']

    @property
    def locations(self) -> Dict[int, float]:
        return self.info['locations']

    @property
    def speeds(self) -> Dict[int, float]:
        return self.info['speeds']

    @property
    def lanes(self) -> Dict[int, int]:
        return self.info['lanes']


class Engine:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.trace = Trace()

        # Initialize the model
        self.model = Model(self.run_config)

    @property
    def run_config(self) -> "RunConfig":
        return self._run_config

    @run_config.setter
    def run_config(self, run_config):
        self._run_config = run_config

    @property
    def trace(self) -> "Trace":
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    @property
    def model(self) -> "Model":
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def run(self):
        for t in range(self.run_config.time_steps):
            # TODO: update the state of the model
            self.trace.add(self.model)  # Record the state of the model


class Trace:
    def __init__(self, *args, **kwargs):
        self.data = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def add(self, data):
        self.data.append(data)
