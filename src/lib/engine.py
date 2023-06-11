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


import copy
import logging
from typing import Dict, List, Set, Tuple
from lib.driver import Driver, DriverConfig, CarType,\
    DriverType, LanePriority, Accident, DriverReactionTime
import lib.driver_distributions as driver_distributions
from lib.road import Road
import numpy as np
# import scipy.stats as st


def initialize_drivers(run_config: 'RunConfig',
                       lognormal_mode: bool = False,
                       debug=False) -> Dict[int, Driver]:
    """
    Initializes the drivers for the simulation.

    If lognormal_mode is set to True, the method
    will take the population size and obtain a random sample from
    the log-normal distribution. Then create an histogram
    of the sample (5 bins), where each bin represents a
    driver type. Then create the drivers based on the
    histogram.

    The id assigned to the drivers is the index of the
    driver in the list of drivers. They are unique and are
    assigned in the order of creation.

    The car seleccion is retrieved from CarType.random().

    NOTE: Drivers & cars selection might change in the future.

    Parameters
    ----------
    population_size : int
        The number of drivers in the simulation.
    lognormal_mode : bool
        If True, the drivers are selected from a log-normal
        distribution. If False, the drivers are selected
        from a multinomial distribution. Check the
        function `DriverType.random_lognormal` for more
        information.

    Returns
    -------
    Dict[int, Driver]
        A dictionary of drivers, where the key is the id
        of the driver and the value is the driver object.
    """
    if lognormal_mode:
        driver_types = DriverType.random_lognormal(
            size=run_config.population_size
        )
    else:
        driver_types = DriverType.random(
            size=run_config.population_size,
            probs=run_config.driver_type_density,
        )

    driver_reaction_times = DriverReactionTime.random(
        size=run_config.population_size,
    )

    drivers = {}

    # Experimental:
    (locations_lane, safe) =\
        driver_distributions.lane_location_initialize_biased(
            start=0, end=run_config.road_length,
            size=run_config.population_size,
            n_lanes=run_config.n_lanes,
            safe_distance=run_config.safe_distance,
            lane_density=np.array(run_config.lane_density),
            safe=True,
        )

    if not safe:
        logging.warning("The drivers were not initialized safely. "
                        "The location of the drivers might be too "
                        "close to other drivers.")

        if debug is False:
            logging.critical("Cannot continue with the simulation. "
                             "Location of drivers is not safe to proceed.")
            raise Exception()

    lanes_array = []
    for k in locations_lane.keys():
        lanes_array.extend(
            [k for _ in range(len(locations_lane[k]))]
        )

    locations_array = []
    for k in locations_lane.keys():
        # print(f'locations_lane[{k}] = {locations_lane[k].tolist()}')
        locations_array.extend(locations_lane[k].tolist())

    for i in range(len(driver_types)):
        car_type = CarType.random()[0]
        dconfig = DriverConfig(
            id=i,
            road=Road(
                length=run_config.road_length,
                n_lanes=run_config.n_lanes,
                max_speed=run_config.max_speed
            ),
            driver_type=driver_types[i],
            driver_reaction_time=driver_reaction_times[i],
            car_type=car_type,
            lane=int(lanes_array[i]),
            location=float(locations_array[i]),
            speed=driver_distributions.speed_initialize(
                car_type=car_type,
                driver_type=driver_types[i],
                size=1,
                max_speed_fixed=run_config.max_speed,
                max_speed_gap=run_config.max_speed_gap,
                min_speed_fixed=run_config.min_speed,
                min_speed_gap=run_config.min_speed_gap,
            )
        )
        drivers[i] = Driver(config=dconfig)

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
        lane_priority : LanePriority
            The priority of the lanes. Defaults to LanePriority.LEFT.
            It is just a visual representation of the lanes used while
            plotting the simulation.
        safe_distance : float
            The safe distance between drivers in meters.
            Defaults to 2 meters.
        lane_density : List[float]
            The density of the drivers in each lane.
            Defaults to equal distribution.
        driver_type_density : List[float]
            The density of the drivers in each driver type.
            Defaults to equal distribution.
        driver_reaction_density : List[float]
            The density of the drivers in each driver reaction time.
            Defaults to equal distribution.
        max_speed : float
            The maximum speed of the drivers in the simulation.
            Defaults to 120 km/h. Must be greater or eq. than min_speed & 60.
        max_speed_gap : float
            The gap between the maximum speed of cars of
            different types. Defaults to 10 km/h.
        min_speed : float
            The minimum speed of the drivers in the simulation.
            Defaults to 60 km/h. Must be less or eq. than max_speed.
        min_speed_gap : float
            The gap between the minimum speed of cars of
            different types. Defaults to 5 km/h.
        start_with_population : bool
            If True, the simulation will start with the population
            specified in the population_size parameter. If False,
            the simulation will start with an empty road.
            Defaults to False.
        minimum_load_factor : float
            The minimum load factor of the simulation.
            The load factor is the ratio of the number of drivers
            in the simulation to the population size.
            If the load factor is less than the minimum load factor,
            the simulation will spawn new drivers until the load
            factor is greater than the minimum load factor.
            Defaults to 0.5.
        accident_clearance_time : float
            The minimum time that an accident will take to be cleared.
            Defaults to 10 seconds.
        debug : bool
            If True, the simulation will run in debug mode.
            Defaults to False.

            For example, when initializing the drivers, the
            locations of the drivers are checked to make sure
            that they are not too close to each other. If they
            are too close, the simulation will not continue unless
            debug is set to True.
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

        if 'n_lanes' in kwargs:
            assert isinstance(kwargs['n_lanes'], int)
            self.n_lanes = kwargs['n_lanes']
        else:
            self.n_lanes = 1

        if 'lane_priority' in kwargs:
            assert isinstance(kwargs['lane_priority'], LanePriority)
            self.lane_priority = kwargs['lane_priority']
        else:
            self.lane_priority = LanePriority.LEFT

        if 'safe_distance' in kwargs:
            assert isinstance(kwargs['safe_distance'], float)
            self.safe_distance = kwargs['safe_distance']
        else:
            self.safe_distance = 2.0  # In meters

        if 'lane_density' in kwargs:
            assert isinstance(kwargs['lane_density'], list) or \
                        isinstance(kwargs['lane_density'], np.ndarray)
            assert np.isclose(np.sum(kwargs['lane_density']), 1.0)
            self.lane_density = kwargs['lane_density']
        else:
            self.lane_density = [1.0 / self.n_lanes] * self.n_lanes

        if 'driver_type_density' in kwargs:
            assert isinstance(kwargs['driver_type_density'], list)
            assert np.isclose(np.sum(kwargs['driver_type_density']), 1.0)
            self.driver_type_density = kwargs['driver_type_density']
        else:
            self.driver_type_density = [1.0 / 5] * 5

        if 'driver_reaction_density' in kwargs:
            assert isinstance(kwargs['driver_reaction_density'], list)
            assert np.isclose(np.sum(kwargs['driver_reaction_density']), 1.0)
            self.driver_reaction_density = kwargs['driver_reaction_density']
        else:
            self.driver_reaction_density = [1.0 / len(
                DriverReactionTime
            )] * len(
                DriverReactionTime
            )

        if 'max_speed' in kwargs:
            assert isinstance(kwargs['max_speed'], float) or \
                     isinstance(kwargs['max_speed'], int)
            self.max_speed = kwargs['max_speed']
        else:
            self.max_speed = 120  # In km/h

        if 'max_speed_gap' in kwargs:
            assert isinstance(kwargs['max_speed_gap'], float) or \
                     isinstance(kwargs['max_speed_gap'], int)
            self.max_speed_gap = kwargs['max_speed_gap']
        else:
            self.max_speed_gap = 10

        if 'min_speed' in kwargs:
            assert isinstance(kwargs['min_speed'], float) or \
                     isinstance(kwargs['min_speed'], int)
            self.min_speed = kwargs['min_speed']
            if self.min_speed > self.max_speed:
                logging.critical("min_speed must be less or equal than "
                                 "max_speed")
                raise Exception()
        else:
            self.min_speed = 60

        if 'min_speed_gap' in kwargs:
            assert isinstance(kwargs['min_speed_gap'], float) or \
                     isinstance(kwargs['min_speed_gap'], int)
            self.min_speed_gap = kwargs['min_speed_gap']
        else:
            self.min_speed_gap = 5

        if 'start_with_population' in kwargs:
            assert isinstance(kwargs['start_with_population'], bool)
            self.start_with_population = kwargs['start_with_population']
        else:
            self.start_with_population = False

        if 'minimum_load_factor' in kwargs:
            assert isinstance(kwargs['minimum_load_factor'], float)
            self.minimum_load_factor = kwargs['minimum_load_factor']
        else:
            self.minimum_load_factor = 0.5

        if 'accident_clearance_time' in kwargs:
            assert isinstance(kwargs['accident_clearance_time'], int)
            self.accident_clearance_time = kwargs['accident_clearance_time']
        else:
            self.accident_clearance_time = 10

        if 'debug' in kwargs:
            assert isinstance(kwargs['debug'], bool)
            self.debug = kwargs['debug']
        else:
            self.debug = False

        # add the rest of parameters here without checking
        # if they are valid or not
        for key in kwargs.keys():
            if key not in self.__dict__:
                self.__dict__[key] = kwargs[key]


class Model:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

        self.info = {}

        # Initialize the drivers with 100% load (all drivers)
        if self.run_config.start_with_population:
            __drivers_by_id = initialize_drivers(
                run_config=self.run_config,
                debug=self.run_config.debug
            )
        else:
            __drivers_by_id = {}

        # Check if there are no drivers
        if len(__drivers_by_id) == 0:
            # Initialize with empty drivers
            # in each lane
            drivers_by_lane_sorted = {
                _lane: {} for _lane in range(self.run_config.n_lanes)
            }
        else:
            # Classify drivers by lane & sort them by position
            drivers_by_lane_sorted_list: Dict[int, List[Driver]] =\
                Driver.sort_by_position_in_lane(
                    drivers_by_lane=Driver.classify_by_lane(
                        list(__drivers_by_id.values())
                    )
                )

            # Convert the list to a dict
            drivers_by_lane_sorted: Dict[int, Dict[int, Driver]] = {}

            for _lane in drivers_by_lane_sorted_list:
                __idx = 0
                drivers_by_lane_sorted[_lane] = {}
                for _driver in drivers_by_lane_sorted_list[_lane]:
                    drivers_by_lane_sorted[_lane][__idx] = _driver
                    _driver.config.index = __idx
                    __idx += 1

        self.info['active_drivers'] = drivers_by_lane_sorted

        self.info['inactive_drivers'] = {}

        self.info['road'] = Road(
            length=self.run_config.road_length,
            n_lanes=self.run_config.n_lanes,
            max_speed=self.run_config.max_speed
        )

        self.id_counter = len(__drivers_by_id)

        # print(f'Initial population: {self.id_counter}')

        # # print all the drivers
        # for _lane in self.info['active_drivers']:
        #     print(f'lane: {_lane}')
        #     for _driver in self.info['active_drivers'][_lane].values():
        #         print(f'\tdriver@{_driver}[{_driver.config.location}]')

        # Initialize dict with the time taken by each driver
        # which is 0 at the beginning
        self.info['time_taken'] = {
            _id: 0 for _id in range(self.id_counter)
        }

        # TODO
        self.load = self.id_counter / self.run_config.population_size

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
    def road(self) -> Road:
        return self.info['road']

    @property
    def active_drivers(self) -> Dict[int, Dict[int, Driver]]:
        """
        Returns a dict with the active drivers in the simulation.
        The dict is a mapping from lane to a dict of drivers in that lane,
        where the key is the driver index in the lane and the value is the
        driver object.

        The index of the driver in the dict is the position of the driver
        in the lane, and so it is the value of the index attribute of the
        driver's config object.

        e.g. driver#2 in lane#1: active_drivers[1][2].config.index == 2
        """
        return self.info['active_drivers']

    @property
    def inactive_drivers(self) -> Dict[int, List[Driver]]:
        """
        Returns a dict with the inactive drivers in the simulation.
        The dict is a mapping from driver id to a list of drivers in that lane.
        """
        return self.info['inactive_drivers']

    @property
    def time_taken(self) -> Dict[int, float]:
        return self.info['time_taken']

    @time_taken.setter
    def time_taken(self, time_taken: Dict[int, float]):
        self.info['time_taken'] = time_taken

    @property
    def id_counter(self) -> int:
        return self._id_counter

    @id_counter.setter
    def id_counter(self, id_counter: int):
        self._id_counter = id_counter

    @property
    def load(self) -> float:
        return self._load

    @load.setter
    def load(self, load: float):
        self._load = load

    def all_active_drivers(self) -> List[Driver]:
        """
        Returns a list of all the active drivers in the simulation.
        """
        drivers = []
        for _lane in self.active_drivers:
            drivers += list(self.active_drivers[_lane].values())
        return drivers

    def set_active(self, driver: 'Driver') -> None:
        """
        Sets a driver as active.
        """
        if driver not in self.active_drivers[
            driver.config.lane
        ].values():
            if driver in self.inactive_drivers:
                logging.warning(
                    f'Driver {driver.config.id} is already inactive.'
                )
                return  # Nothing to do
            # Search the position of the driver in the lane
            # and insert it in the correct position
            __lane = driver.config.lane
            if len(self.active_drivers[__lane]) == 0:
                # If there are no drivers in the lane
                # insert the driver in the first position
                self.active_drivers[__lane][0] = driver
                driver.config.index = 0
            else:
                # Add the driver sorted to the new lane
                __index = 0  # Farthest are first
                __dict_temp = {}
                driver.config.index = -1
                for __idx in self.active_drivers[driver.config.lane]:
                    __driver = self.active_drivers[
                        driver.config.lane
                    ][__idx]
                    if __driver.config.location > driver.config.location:
                        # Sorted
                        # print('---> A')
                        __dict_temp[__index] = __driver
                        __index += 1
                    else:
                        driver.config.index = __index
                        __dict_temp[__index] = driver
                        # print('---> B')
                        # print(f'New driver index: {__index}')
                        break
                # Check if driver is the last one
                if driver.config.index == -1:
                    # print('---> C1')
                    driver.config.index = __index
                    __dict_temp[__index] = driver
                else:
                    # Add the rest of the drivers
                    while __index < len(self.active_drivers[driver.config.lane]):
                        # print('---> C2')
                        __driver = self.active_drivers[
                            driver.config.lane
                        ][__index]
                        __dict_temp[__index + 1] = __driver
                        __index += 1
                # Update the dict
                self.active_drivers[__lane].update(__dict_temp)

        # print(f'Drivers in lane {driver.config.lane}:')
        # print(f'{self.active_drivers[driver.config.lane]}')

    def set_inactive(self, driver: Driver) -> None:
        """
        Sets a driver as inactive.
        """
        if driver not in self.inactive_drivers:
            if driver in self.active_drivers[
                driver.config.lane
            ].values():
                self.info['active_drivers'][
                    driver.config.lane
                ].pop(driver.config.index)

    def generate_driver(self) -> Driver:
        """
        Spawns a new driver in the simulation.
        Location will be 0 (start of the road).
        """
        car_type = CarType.random()[0]
        drv_type = DriverType.random(
            size=1,
            probs=self.run_config.driver_type_density
        )[0]
        drv_react = DriverReactionTime.random(
            size=1,
            probs=self.run_config.driver_reaction_density
        )
        new_driver = Driver(
            config=DriverConfig(
                id=self.id_counter,
                driver_type=drv_type,
                reaction_time=drv_react,
                road=self.road,
                car_type=car_type,
                lane=int(driver_distributions.lane_initialize_weighted(
                    n_lanes=self.run_config.n_lanes,
                    probs=self.run_config.lane_density  # type: ignore
                )),
                location=0,  # Start at the beginning of the road
                speed=driver_distributions.speed_initialize(
                    driver_type=drv_type,
                    car_type=car_type,
                    size=1,
                    max_speed_fixed=self.run_config.max_speed,
                    max_speed_gap=self.run_config.max_speed_gap,
                    min_speed_fixed=self.run_config.min_speed,
                    min_speed_gap=self.run_config.min_speed_gap,
                ),
            ),
        )

        new_driver.config.index = len(self.active_drivers[new_driver.config.lane])

        return new_driver

    def spawn_driver(self, driver: Driver, engine: 'Engine') -> None:
        """
        Adds a given driver to the simulation.
        """
        # print(f'Spawning driver {driver.config.id} in lane {driver.config.lane}')
        # print(f'\t current id_counter: {self.id_counter}')
        if self.id_counter != driver.config.id:
            driver.config.id = self.id_counter  # id must be unique
        self.id_counter += 1
        engine.driver_enters(driver)

    def reject_driver(self, driver: Driver) -> None:
        """
        Rejects a driver from the simulation.
        """
        self.id_counter -= 1

    def update_load(self):
        # Count the number of active drivers
        __active_drivers = 0
        for lane in self.active_drivers:
            __active_drivers += len(self.active_drivers[lane])
        # print(f'Load: {__active_drivers} / {self.run_config.population_size}')
        self.load = __active_drivers / self.run_config.population_size

    def check_accident(self) -> Tuple[bool, List[Accident]]:
        """
        Checks if there is an accident in the simulation.

        Returns
        -------
        Tuple[bool, List[Driver]]
            A tuple with a boolean indicating if there was an
            accident and a list of the drivers involved in the
            accident.

        NOTE: this function might be inneficient and should
        be tested & improved. An idea might be
        checking the drivers in order once they are
        ordered by location. # TODO
        """
        accidents = []

        # TODO
        return (False, accidents)

    def driver_updates(
        self,
        old_driver: Driver,
        new_driver: Driver,
    ) -> Driver:
        """
        Called whenever a driver updates its state, namely
        it's location, speed or/and lane.

        For now, it'll be used to mantain the active drivers
        dictionary sorted
        """
        # TODO: TEST THIS FUNCTION

        # print(f'\t Old driver: {old_driver}')
        # print(f'\t New driver: {new_driver}')

        # print(f'All active drivers[pre-update]: {self.all_active_drivers()}')

        # Compare the old and new driver to check
        # how much location & lane changed

        print(f'\t[{old_driver.config.id}] Old driver pos/lane: {old_driver.config.location}/{old_driver.config.lane}')
        print(f'\t[{new_driver.config.id}] New driver pos/lane: {new_driver.config.location}/{new_driver.config.lane}')

        # Only active drivers can update their state

        if old_driver == self.active_drivers[
            old_driver.config.lane
        ][old_driver.config.index]:
            # Check if we need to change the lane
            if old_driver.config.lane != new_driver.config.lane:
                # Remove the driver from the old lane
                self.active_drivers[
                    old_driver.config.lane
                ].pop(old_driver.config.index)

                # Update the index of the drivers in the old lane
                __dict_temp = {}
                # +1 because we want to start from the next driver
                # and len() + 1 because as we removed the driver
                # from the old lane, the last index is now len() - 1
                for __idx in range(
                    old_driver.config.index + 1,
                    len(self.active_drivers[old_driver.config.lane]) + 1
                ):
                    print(self.active_drivers, old_driver.config.lane, __idx)
                    __driver = self.active_drivers[
                        old_driver.config.lane
                    ][__idx]
                    __driver.config.index -= 1
                    __dict_temp[__driver.config.index] = __driver

                # Update the active drivers dictionary
                self.active_drivers[old_driver.config.lane].update(__dict_temp)

                # Check number of drivers in the new lane
                if len(self.active_drivers[new_driver.config.lane]) == 0:
                    # If there are no drivers in the new lane,
                    # just add the driver to the new lane
                    self.active_drivers[new_driver.config.lane][0] = new_driver
                    new_driver.config.index = 0
                else:
                    # Add the driver sorted to the new lane
                    __index = 0  # Farthest are first
                    __dict_temp = {}
                    new_driver.config.index = -1
                    for __idx in self.active_drivers[new_driver.config.lane]:
                        __driver = self.active_drivers[
                            new_driver.config.lane
                        ][__idx]
                        if __driver.config.location > new_driver.config.location:
                            # Sorted by location, keep adding
                            # print('---> A')
                            __dict_temp[__index] = __driver
                            __index += 1
                        else:
                            # Found the position
                            new_driver.config.index = __index
                            __dict_temp[__index] = new_driver
                            # print('---> B')
                            # print(f'New driver index: {__index}')

                            # Add the rest of the drivers
                            while __index < len(
                                self.active_drivers[new_driver.config.lane]
                            ):
                                # print('---> C2')
                                __driver = self.active_drivers[
                                    new_driver.config.lane
                                ][__index]
                                __dict_temp[__index + 1] = __driver
                                __driver.config.index += 1
                                __index += 1
                            # Exit the loop
                            break

                    # Check if driver is the last one
                    if new_driver.config.index == -1:
                        # print('---> C1')
                        new_driver.config.index = __index
                        __dict_temp[__index] = new_driver

                    # print(f'Len_temp: {len(__dict_temp)}')
                    # print(f'Temp dict: {__dict_temp}')
                    # print(f'Len_active: {len(self.active_drivers[new_driver.config.lane])}')
                    assert len(__dict_temp) == len(
                        self.active_drivers[new_driver.config.lane]
                    ) + 1
                    self.active_drivers[
                        new_driver.config.lane
                    ].update(__dict_temp)

                # TODO: check accidents
            else:
                # Update the driver in the same lane
                self.active_drivers[
                    old_driver.config.lane
                ][old_driver.config.index] = new_driver
                new_driver.config.index = old_driver.config.index
                pass  # Same lane, # TODO check accidents

        return new_driver


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
        __wait_to_spawn = False
        for t in range(self.run_config.time_steps):
            # Set a minimum population
            print(f'[INFO] Time step {t} - {self.model.load}')
            if self.model.load < self.run_config.minimum_load_factor and \
                    __wait_to_spawn is False:
                __candidate = self.model.generate_driver()
                # Check if the new driver is too close to other drivers
                # If it is, discard the driver and continue
                __drivers_close = Driver.drivers_close(
                    __candidate,
                    self.model.active_drivers[__candidate.config.lane],
                )
                # print(f'[INFO] Time step {t} - {__drivers_close}')
                if len(__drivers_close) > 0:
                    __candidates_close = [
                        driver for driver in __drivers_close
                        if Driver.collision(driver, __candidate)
                    ]
                    if len(__candidates_close) > 0:
                        # We need to wait to spawn the driver
                        # so that the drivers can move
                        # and the new driver can be spawned
                        # as free space will be available
                        self.model.reject_driver(__candidate)
                        __wait_to_spawn = True
                    else:
                        # It's safe to spawn the driver
                        self.model.spawn_driver(__candidate, self)
                        self.model.update_load()
                else:
                    # It's safe to spawn the driver
                    self.model.spawn_driver(__candidate, self)
                    self.model.update_load()

            print(f'[INFO] Time step {t} - {self.model.active_drivers}]')

            # Copy the state of the simulation
            # to the trace before updating
            self.trace.add(copy.deepcopy(self.model))

            __state = self.trace.last.active_drivers
            # print(f'[INFO] Time step {t} - {__state}')
            __new_state_drivers = []
            __finished_drivers = []  # Track drivers that finished

            for __lane in self.model.active_drivers:
                for __driver in __state[__lane].values():
                    # Update the time taken by the driver
                    self.model.time_taken[__driver.config.id] += 1
                    # Update the speed & location of the driver
                    __driver.action(
                        state=__state,  # type: ignore
                        update_fn=driver_distributions.speed_update,
                        callback_fn=self.model.driver_updates,
                        max_speed_fixed=self.run_config.max_speed,
                        max_speed_gap=self.run_config.max_speed_gap,
                        min_speed_fixed=self.run_config.min_speed,
                        min_speed_gap=self.run_config.min_speed_gap,
                    )
                    # Check if driver has reached the end of the road
                    if __driver.config.location >= self.model.road.length:
                        __finished_drivers.append(__driver)
                    else:
                        __new_state_drivers.append(__driver)

            for __driver in __finished_drivers:
                self.driver_finishes(__driver)
                self.model.update_load()

            # Check accident
            (accident, accidents) = self.model.check_accident()
            if accident:
                # Stop that cars involved in the accident
                # in their current location for a while
                # For each accident, the drivers will be stopped
                # for a random number of time steps given
                # by driver_distributions.collision_wait_time()
                # TODO
                pass

            # Print model
            print(f'[INFO - END STEP] Time step {t} - {self.model.active_drivers}')

            # Print all locations
            # for __lane in self.model.active_drivers:
            #     for __driver in self.model.active_drivers[__lane].values():
            #         print(f'[INFO] Time step {t} - {__driver}.{__driver.config.location}')

            # print(f'Finished time step {t}')
            # print(f'\t Active drivers: {self.model.all_active_drivers()}')
            __wait_to_spawn = False

    def driver_finishes(self, driver: Driver) -> None:
        """
        This method is called whenever a driver reaches
        the end of the road.
        """
        if driver in self.model.active_drivers[driver.config.lane]:
            self.model.set_inactive(driver)

    def driver_enters(self, driver: Driver) -> None:
        """
        This method is called whenever a driver enters
        the road.

        Requires that the driver is not already in the road
        and hadn't been in the road before.

        Driver should be generated by the spawn_driver method.

        Parameters
        ----------
        driver : Driver
            The driver that enters the road.
        """
        print(f'[INFO] Adding driver {driver} to the road.')
        if driver in self.model.inactive_drivers:
            logging.critical("Cannot add driver to the road. "
                             "Driver was already in the road.")
            raise Exception()
        elif driver in self.model.active_drivers[driver.config.lane].values():
            logging.critical("Cannot add driver to the road. "
                             "Driver is already in the road.")
            raise Exception()

        # self.model.info['active_drivers'][driver.config.id] = driver
        self.model.set_active(driver)
        self.model.time_taken[driver.config.id] = 0  # Reset time taken

        # __len_drivers = sum(
        #     [len(self.model.active_drivers[lane]) for lane in self.model.active_drivers]
        # )
        # print(f'Length of active drivers: {__len_drivers}')


class Trace:
    def __init__(self, *args, **kwargs):
        self.data = []

    @property
    def data(self) -> List['Model']:
        return self._data

    @data.setter
    def data(self, data: List['Model']):
        self._data = data

    @property
    def last(self) -> 'Model':
        return self.data[-1]

    def add(self, data: 'Model'):
        self.data.append(data)


class Stats:
    def __init__(self, *args, **kwargs):
        if 'engine' not in kwargs:
            logging.critical("Cannot create ModelStats without a model.")
            raise Exception()
        self.engine = kwargs['engine']

    @property
    def engine(self) -> Engine:
        return self._engine

    @engine.setter
    def engine(self, engine: Engine):
        self._engine = engine

    def _get_lane_changes(self):
        """
        Returns a dictionary with the number of lane changes
        for each driver.
        """
        drivers =\
            list(self.engine.model.inactive_drivers)

        for lane in self.engine.model.active_drivers:
            drivers += list(self.engine.model.active_drivers[lane].values())

        # Initialize as dict of lists
        lane_changes = {driver.config.id: [] for driver in drivers}

        # print(f'Trace length: {len(self.engine.trace.data)}')

        d_finished: List[Driver] = []
        for t in range(1, len(self.engine.trace.data)):
            for driver in drivers:
                # Check if the driver has finished
                trace_dict_t: Dict[int, Driver] = {}
                for lane in self.engine.trace.data[t].active_drivers:
                    trace_dict_t.update(
                        self.engine.trace.data[t].active_drivers[lane]
                    )

                trace_dict_t_1: Dict[int, Driver] = {}
                for lane in self.engine.trace.data[t - 1].active_drivers:
                    trace_dict_t_1.update(
                        self.engine.trace.data[t - 1].active_drivers[lane]
                    )
                # print(f'Drivers in trace_list_t: {[d.config.id for d in trace_list_t]}')
                # print(f'Drivers in trace_list_t_1: {[d.config.id for d in trace_list_t_1]}')
                # print(f'{t} / {len(self.engine.trace.data)}')
                if driver.config.id in\
                        self.engine.trace.data[t].inactive_drivers:
                    # It means that the driver finished
                    if driver not in d_finished:
                        lane_changes[driver.config.id].append(
                            self.engine.run_config.n_lanes + 1
                        )
                    d_finished.append(driver)
                    break
                elif driver.config.id not in trace_dict_t:
                    # It means that the driver was not in the road
                    # in the current time step
                    continue
                elif driver.config.id not in trace_dict_t_1:
                    # It means that the driver was not in the road
                    # in the previous time step
                    lane_changes[driver.config.id].append(
                        trace_dict_t[driver.config.id].config.lane
                    )
                else:
                    # Check if the driver changed lanes
                    # with respect to the previous time step
                    if trace_dict_t[driver.config.id].config.lane !=\
                         trace_dict_t_1[driver.config.id].config.lane:
                        # Add the lane change as trace
                        lane_changes[driver.config.id].append(
                            trace_dict_t[driver.config.id].config.lane
                        )

        # Sanitize those drivers that didn't change lanes
        for driver in drivers:
            if len(lane_changes[driver.config.id]) == 0:
                lane_changes[driver.config.id].append(
                    driver.config.lane
                )

        return lane_changes

    def _get_avg_time_taken(self) -> Dict[DriverType, float]:
        """
        Returns the average time taken by each driver type.

        Returns
        -------
        Dict[DriverType, float]
            A dictionary with the average time taken by each
            driver type.
        """
        avg_time_taken = {}
        for driver in self.engine.model.inactive_drivers:
            if driver.config.driver_type not in avg_time_taken:
                avg_time_taken[driver.config.driver_type] = 0
            avg_time_taken[driver.config.driver_type] +=\
                self.engine.model.time_taken[driver.config.id]
        for driver_type in avg_time_taken:
            avg_time_taken[driver_type] /=\
                len(Driver.classify_by_type(self.engine.model.inactive_drivers)[
                    driver_type
                ])
        return avg_time_taken

    def _avg_starting_position(self) -> Dict[DriverType, float]:
        """
        Returns the average starting position of each driver type.

        Returns
        -------
        Dict[DriverType, float]
            A dictionary with the average starting position of each
            driver type.
        """
        avg_starting_position = {}
        for driver in self.engine.model.inactive_drivers:
            if driver.config.driver_type not in avg_starting_position:
                avg_starting_position[driver.config.driver_type] = 0
            avg_starting_position[driver.config.driver_type] +=\
                driver.config.origin
        for driver in self.engine.model.all_active_drivers():
            if driver.config.driver_type not in avg_starting_position:
                avg_starting_position[driver.config.driver_type] = 0
            avg_starting_position[driver.config.driver_type] +=\
                driver.config.origin
        classified_by_type =\
            Driver.classify_by_type(self.engine.model.inactive_drivers)
        for driver_type in avg_starting_position:
            if len(classified_by_type[driver_type]) == 0:
                avg_starting_position[driver_type] = 0
            else:
                avg_starting_position[driver_type] /=\
                    len(classified_by_type[
                        driver_type
                    ])
        return avg_starting_position

    def get_stats(self):
        """
        Returns a dictionary of statistics about the model.

        Returns
        -------
        Dict
            A dictionary of statistics about the model.

            'avg_time_taken' : Dict[DriverType, float]
                The average time taken by each driver type.

            'avg_starting_position' : Dict[DriverType, float]
                The average starting position of each driver type.
                It should be close to 0, so if it is not, it means
                that the burn-in period was not long enough.
        """
        stats = {}

        stats['avg_starting_position'] = self._avg_starting_position()
        stats['avg_time_taken'] = self._get_avg_time_taken()
        stats['lane_changes'] = self._get_lane_changes()

        return stats
