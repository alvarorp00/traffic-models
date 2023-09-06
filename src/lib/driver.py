"""
This class defines the interface that all drivers must implement. It
also provides a few utility methods that are useful for all drivers.

The driver interface is designed to be as simple as possible. The
action() method will be called to act on a discrete time step,
where the time step is defined by the simulation.

The action() method would depend on the parameters of the driver
which will be set depending on the type of driver. For example, a
driver that takes risks will consider overtaking other vehicles
within less distance than a driver that is more cautious.

NOTE: above statement might change.

Almost all the methods in this class will be implemented by itself,
but the parameters of the driver will be set by the simulation
and so are required to be passed as arguments to the constructor.
"""

import bisect
import copy
import enum
import random
from typing import Dict, List, Optional, Union
import numpy as np
import scipy.stats as st


class LanePriority(enum.Enum):
    """
    Represents the priority of a lane. 2 priorities are defined:
    LEFT, RIGHT

    It is just a visual representation that will be used to
    plot the drivers according to their lane priority, i.e.
    the drivers in the first lane will be plotted at the right
    whenever the priority is LEFT, so the highest priority
    lane will be plotted at the left (the last lane).
    """
    LEFT = 0,
    RIGHT = 1

    @staticmethod
    def random(size=1) -> list['LanePriority']:
        """
        Returns a random lane priority.

        Probability distribution is defined by a multinomial
        distribution.

        Probability of LEFT is 0.5 and probability of RIGHT is 0.5.

        Returns
        -------
        LanePriority
            A random lane priority.
        """
        choice = random.choices(
            population=list(LanePriority),
            weights=[.5, .5],
            k=size
        )

        return choice


class CarType(enum.Enum):
    """
    Enum for the type of car. 4 types are defined:
    """
    TRUCK = 1
    VAN = 2
    SEDAN = 3
    MOTORCYCLE = 4

    @staticmethod
    def random(
        probs: List[float] = [.2, .4, .3, .1],
        size=1
    ) -> list['CarType']:
        """
        Returns a random car type.

        Probability distribution is defined by a multinomial
        distribution.

        Returns
        -------
        CarType
            A random car type.
        """
        choice = random.choices(
            population=list(CarType),
            weights=probs,
            k=size
        )

        return choice

    @staticmethod
    def random_lognormal(size=1) -> list['CarType']:
        """
        Returns a random car type.

        Probability distribution is defined by a log-normal
        distribution.

        Returns
        -------
        CarType
            A random car type.
        """
        stats = st.lognorm.rvs(0.5, size=int(1e5))

        # Create the histogram
        hist, _ = np.histogram(stats, bins=5)

        # Get probs
        probs = np.linalg.norm(np.array(hist))

        choice = random.choices(
            population=list(CarType),
            weights=probs,
            k=size
        )

        # return choice
        return choice

    @staticmethod
    def get_max_speed(
        car_type: 'CarType',
        car_max_speeds: List[float]
    ) -> float:
        """
        Returns the maximum speed of the car in meters per second.

        Parameters
        ----------
        car_type : CarType
            The type of car.
        car_max_speeds : List[float]
            The list of maximum speeds for each car type.

        Returns
        -------
        float
            The maximum speed of the car.
        """
        return car_max_speeds[car_type.value - 1]

    @staticmethod
    def get_min_speed(
        car_type: 'CarType',
        car_min_speeds: List[float]
    ) -> float:
        """
        Returns the minimum speed of the car in meters per second.

        Parameters
        ----------
        car_type : CarType
            The type of car.
        car_min_speeds : List[float]
            The list of minimum speeds for each car type.

        Returns
        -------
        float
            The minimum speed of the car.
        """
        return car_min_speeds[car_type.value - 1]

    @staticmethod
    def get_length(car_type: 'CarType') -> float:
        """
        Returns the size of the car in meters.

        Parameters
        ----------
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The size of the car.
        """
        if car_type == CarType.MOTORCYCLE:
            return 2
        elif car_type == CarType.SEDAN:
            return 3.5
        elif car_type == CarType.VAN:
            return 4.5
        elif car_type == CarType.TRUCK:
            return 8
        else:
            raise ValueError(f"Invalid car type @ {car_type}")


class DriverType(enum.Enum):
    """
    Enum for the type of driver. 5 types are defined:
    """
    CAUTIOUS = 1
    NORMAL = 2
    RISKY = 3
    AGGRESSIVE = 4
    RECKLESS = 5

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def driver_safe_distance(
        driver_type: 'DriverType',
        safe_distance: float,
        safe_distance_factor: List[float]
    ) -> float:
        """
        Returns the safe distance of the driver based on the type of driver.
        """
        _multiplier = safe_distance_factor[driver_type.value - 1]
        return safe_distance * _multiplier

    @staticmethod
    def max_close_distance(
        driver_type: 'DriverType',
        driver_safe_distance: float
    ) -> float:
        """
        Returns the maximum a car can be from the car in front of it
        depending on the driver type.
        """
        # The safe distance is the minimum distance between cars
        SAFE_DISTANCE = driver_safe_distance
        # The multiplier for the gap between cars
        MULTIPLIER = 10
        # Add some noise to the distance so it's not always the same
        NOISE = np.random.uniform(
            -SAFE_DISTANCE / MULTIPLIER, SAFE_DISTANCE / MULTIPLIER)
        return SAFE_DISTANCE + MULTIPLIER * (
            DriverType.RECKLESS.value - driver_type.value) + NOISE

    @staticmethod
    def driver_view_distance(
        driver_type: 'DriverType',
        vision_distance: float,
        vision_distance_factor: List[float]
    ) -> float:
        """
        Returns the vision distance of the driver based on the type of driver.
        """
        _multiplier = vision_distance_factor[driver_type.value - 1]
        return vision_distance * _multiplier

    @staticmethod
    def driver_keep_overtaking(
        driver_type: 'DriverType',
        time_in_lane: float,
        running_time: float,  # of the driver
    ) -> bool:
        """
        Returns whether the driver will keep overtaking or not.
        """
        return False

    @staticmethod
    def get_speed_modifier(
        driver_type: 'DriverType',
        modifiers: List[float] = [0.8, 1.0, 1.1, 1.2, 1.3]
    ) -> float:
        """
        Returns the speed modifier for the driver.
        """
        return modifiers[driver_type.value - 1]

    @staticmethod
    def get_time_in_lane(
        driver_type: 'DriverType',
        time_in_lane: float,
        time_in_lane_factor: List[float]
    ) -> int:
        """
        Returns the time the driver will keep in lane.
        """
        mean = time_in_lane
        scale = time_in_lane_factor[
            driver_type.value - 1] + np.random.uniform(0, driver_type.value)
        return abs(int(st.norm.rvs(mean, scale)))

    @staticmethod
    def get_max_speed(
        driver_type: 'DriverType',
        modifiers: List[float],
        car_type: 'CarType',
        cars_max_speeds: List[float],
        **kwargs
    ) -> float:
        """
        Returns the maximum speed of the driver.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The maximum speed of the driver.
        """
        base_max_speed = CarType.get_max_speed(
            car_type, cars_max_speeds)
        speed_modifier = DriverType.get_speed_modifier(
            driver_type, modifiers)

        return base_max_speed * speed_modifier

    @staticmethod
    def get_min_speed(
        driver_type: 'DriverType',
        modifiers: List[float],
        car_type: 'CarType',
        cars_min_speeds: List[float],
        **kwargs
    ) -> float:
        """
        Returns the minimum speed of the driver.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The minimum speed of the driver.
        """
        base_min_speed = CarType.get_min_speed(
            car_type, cars_min_speeds)
        speed_modifier = DriverType.get_speed_modifier(
            driver_type, modifiers)

        diff = abs(base_min_speed * speed_modifier - base_min_speed)

        return base_min_speed - diff

    @staticmethod
    def random(
        size: int = 1,
        probs: List[float] = [.4, .3, .15, .1, .05],
    ) -> list['DriverType']:
        """
        Returns a weighted random driver type.

        Probabilities:
        - Given or default (see below)

            - CAUTIOUS: 0.4
            - NORMAL: 0.3
            - RISKY: 0.15
            - AGGRESSIVE: 0.1
            - RECKLESS: 0.05

        - Probabilities given must be of length 5.

        Returns
        -------
        DriverType
            A random driver type.
        """
        choice = random.choices(
            population=list(DriverType),
            weights=probs,
            k=size
        )

        return choice

    @staticmethod
    def random_lognormal(size=1) -> list['DriverType']:
        """
        Returns a random driver type.

        Probability distribution is defined by a log-normal
        distribution.

        Returns
        -------
        DriverType
            A random driver type.
        """
        stats = st.lognorm.rvs(0.5, size=int(1e5))

        # Create the histogram
        hist, _ = np.histogram(stats, bins=5)

        # Get probs by normalizing
        probs = np.array(hist) / np.sum(np.array(hist))

        choice = random.choices(
            population=list(DriverType),
            weights=probs,
            k=size
        )

        # return choice
        return choice

    @staticmethod
    def from_int(int) -> 'DriverType':
        return DriverType(int)

    @staticmethod
    def as_index(driver_type: 'DriverType') -> int:
        """
        Returns the index of the driver type.

        Parameters
        ----------
        driver_type : DriverType
            The driver type.

        Returns
        -------
        int
            The index of the driver type.
        """
        return driver_type.value - 1


class DriverReactionTime(enum.Enum):
    FAST = 1
    NORMAL_FAST = 2
    NORMAL = 3
    NORMAL_SLOW = 4
    SLOW = 5

    def __new__(cls, value, *args, **kwargs):
        member = object.__new__(cls)
        member._value_ = value
        member._args_ = args
        member._kwargs_ = kwargs
        return member

    def __int__(self):
        return self.value

    @staticmethod
    def random(
        size: int = 1,
        probs: List[float] = [.4, .3, .15, .1, .05],
    ) -> list['DriverReactionTime']:
        """
        Returns a weighted random driver reaction time.

        Probabilities:
        - Given or default (see below)

            - QUICK: 0.4
            - NORMAL: 0.3
            - SLOW: 0.2
            - SNAIL: 0.1

        - Probabilities given must be of length 4.
        """
        choice = random.choices(
            population=list(DriverReactionTime),
            weights=probs,
            k=size
        )

        if size == 1:
            return choice[0]
        else:
            return choice


class Accident:
    def __init__(self, **kwargs):
        """
        Constructor for the Accident class.

        Parameters
        ----------
        drivers : set(Driver)
            List of drivers that caused the accident.
        accident_clearance_time : int
            The time to wait per driver in the accident
            before the accident is resolved.
            Default: 1 second
        """
        if 'drivers' not in kwargs:
            self.drivers = set()
        else:
            if isinstance(kwargs['drivers'], set):
                self.drivers = kwargs['drivers']
            else:
                self.drivers = set(kwargs['drivers'])

        if 'accident_clearance_time' in kwargs:
            assert isinstance(kwargs['accident_clearance_time'], int)
            self.wait_time = kwargs['accident_clearance_time']
        else:
            self.wait_time = 1

        self.expire_time = len(self.drivers) * self.wait_time

    @property
    def drivers(self) -> set['Driver']:
        return self._drivers

    @drivers.setter
    def drivers(self, drivers: set['Driver']):
        self._drivers = drivers

    def add_driver(self, driver: 'Driver'):
        self.drivers.add(driver)

    @property
    def wait_time(self) -> int:
        return self._wait_time

    @wait_time.setter
    def wait_time(self, wait_time: int):
        self._wait_time = wait_time

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Accident):
            return False
        return self.drivers == __value.drivers

    def __hash__(self) -> int:
        return sum([hash(d) for d in self.drivers]) + self.wait_time

    def count_down(self):
        self.expire_time -= 1

    def is_expired(self) -> bool:
        return self.expire_time <= 0


class DriverConfig:
    """
    Configuration for a driver.

    This class is used to configure the parameters of a driver. It
    is used by the simulation to create a new driver.

    Attributes
    ----------
    driver_type : DriverType
        The type of driver.
    reaction_time : DriverReactionTime
        The reaction time of the driver.
    car_type : CarType
        The type of car.
    location : float
        The location of the driver.
    origin : int
        Set automatically by the simulation, is the initial position
    speed : float
        The initial speed of the driver.
    lane : int
        The lane of the driver.
    index : int
        The index of the driver. Represents the relative position of
        the driver in the lane. Defaults to -1 (out of the road).
    accidented : bool
        Whether the driver is accidented or not.
    running_time : int
        The time the driver has been running.
    time_in_lane : int
        The time left in the lane until considering changing lanes.
    brake_counter : int
        The times the driver has braked without changing lanes
        and with the same driver in front.
    driver_in_front_id : int
        The id of the driver in front.
    """
    def __init__(self, id, **kwargs):
        """
        Constructor for the DriverConfig class.
        """
        self.id = id

        if 'driver_type' not in kwargs:
            raise ValueError("driver_type not passed to the constructor")
        self.driver_type = kwargs['driver_type']

        if 'reaction_time' not in kwargs:
            raise ValueError("reaction_time not passed to the constructor")
        self.reaction_time = kwargs['reaction_time']

        if 'car_type' not in kwargs:
            raise ValueError("car_type not passed to the constructor")
        self.car_type = kwargs['car_type']

        if 'road' not in kwargs:
            raise ValueError("road not passed to the constructor")
        self.road = kwargs['road']

        if 'location' in kwargs:
            # assert isinstance(kwargs['location'], float)
            self.location = kwargs['location']
        else:
            self.location = 0

        self.origin = self.location

        if 'speed' in kwargs:
            assert isinstance(kwargs['speed'], float)
            self.speed = kwargs['speed']
        else:
            self.speed = 0

        if 'lane' in kwargs:
            assert isinstance(kwargs['lane'], int)
            self.lane = kwargs['lane']
        else:
            self.lane = 0

        if 'index' in kwargs:
            self.index = kwargs['index']
        else:
            self.index = 0

        self.accidented = False
        self.running_time = 0
        self.time_in_lane = 0
        self.brake_counter = 0
        self.driver_in_front_id = -1

    @property
    def driver_type(self) -> DriverType:
        return self._driver_type

    @driver_type.setter
    def driver_type(self, driver_type: DriverType):
        self._driver_type = driver_type

    @property
    def reaction_time(self) -> DriverReactionTime:
        return self._reaction_time

    @reaction_time.setter
    def reaction_time(self, reaction_time: DriverReactionTime):
        self._reaction_time = reaction_time

    @property
    def car_type(self) -> CarType:
        return self._car_type

    @car_type.setter
    def car_type(self, car_type: CarType):
        self._car_type = car_type

    @property
    def location(self) -> float:
        return self._location

    @location.setter
    def location(self, location: float):
        self._location = location

    @property
    def origin(self) -> int:
        return self._origin

    @origin.setter
    def origin(self, origin: int):
        self._origin = origin

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, speed: float):
        self._speed = speed

    @property
    def lane(self) -> int:
        return self._lane

    @lane.setter
    def lane(self, lane: int):
        self._lane = lane

    @property
    def accidented(self) -> bool:
        return self._accidented

    @accidented.setter
    def accidented(self, accidented: bool):
        self._accidented = accidented

    @property
    def running_time(self) -> int:
        return self._running_time

    @running_time.setter
    def running_time(self, running_time: int):
        self._running_time = running_time

    @property
    def time_in_lane(self) -> int:
        return self._time_in_lane

    @time_in_lane.setter
    def time_in_lane(self, time_in_lane: int):
        self._time_in_lane = time_in_lane

    @property
    def brake_counter(self) -> int:
        return self._brake_counter

    @brake_counter.setter
    def brake_counter(self, brake_counter: int):
        self._brake_counter = brake_counter

    @property
    def driver_in_front_id(self) -> int:
        return self._driver_in_front_id

    @driver_in_front_id.setter
    def driver_in_front_id(self, driver_in_front_id: int):
        self._driver_in_front_id = driver_in_front_id


class Driver:
    def __init__(self, *args, **kwargs):
        """
        Constructor for the driver class.

        This method will be called by the simulation to create a new
        driver. It is expected to set the parameters of the driver
        based on the parameters passed to it.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters of the driver.
            --> "config": DriverConfig
        """
        if 'config' not in kwargs:
            raise ValueError("config not passed to the constructor")
        assert isinstance(kwargs['config'], DriverConfig)
        self.config = kwargs['config']

    def __eq__(self, other: 'Driver'):
        return self.config.id == other.config.id

    def __hash__(self):
        return hash(self.config.id)

    def __repr__(self):
        # return self.show()
        return self.show_verbose()

    def show(driver: 'Driver') -> str:
        return f'driver@{driver.config.id}'

    def show_verbose(driver: 'Driver') -> str:
        return f'driver@{driver.config.id} | '\
               f'{driver.config.driver_type} | {driver.config.car_type} | '\
               f'{driver.config.location} | {driver.config.speed} | '\
               f'{driver.config.lane} | {driver.config.index} | '\
               f'{driver.config.accidented}'

    @property
    def config(self) -> DriverConfig:
        return self._config

    @config.setter
    def config(self, config: DriverConfig):
        self._config = config

    def keep_going(
            self: 'Driver',
            callback_fn: callable,  # type: ignore
    ) -> 'Driver':
        """
        This method is called by the simulation to keep the driver
        moving forward. It is expected to update the driver's
        location based on the current speed, not to change the
        speed nor the lane.
        """
        __driver_updated = Driver.copy(self)
        __driver_updated.config.location +=\
            __driver_updated.config.speed / 3.6  # km/h -> m/s
        return callback_fn(self, __driver_updated)

    def action(
            self: 'Driver',
            state: Dict[int, Dict[int, 'Driver']],
            update_fn: callable,  # type: ignore
            callback_fn: callable,  # type: ignore
            **kwargs
            ) -> 'Driver':
        """
        This method will be called by the simulation to act on a
        discrete time step. The time step is defined by the
        simulation. The driver will act on the given state, i.e.
        the drivers on the road with their corresponding parameters.

        Parameters
        ----------
        state : dict
            A dictionary containing the drivers on the road with
            their corresponding parameters, indexed by their lane
            number.
        update_fn : callable
            A function that will be called to update the driver
            speed & lane based on the given state. Should be called
            with lib.driver_distributions.speed_update, although
            custom functions can be provided.
        callback_fn : callable
            params: old_driver, new_driver
            Should return the updated driver.

        Returns
        -------
        Returns a call to the callback_fn with the old driver & 
        the updated one.
        """

        __driver: Driver = update_fn(
            driver=self,
            state=state,
            **kwargs
        )

        __driver.config.running_time += 1

        # Call the callback function
        return callback_fn(
            old_driver=self,
            new_driver=__driver
        )

    @staticmethod
    def classify_by_type(drivers: list['Driver']) ->\
            Dict[DriverType, List['Driver']]:
        """
        Returns a dictionary that maps driver types to a list of drivers
        of that type.
        """
        dict = {}

        for d in drivers:
            if d.config.driver_type in dict.keys():
                dict[d.config.driver_type].append(d)
            else:
                dict[d.config.driver_type] = [d]
        # Check if all driver types are present
        for t in DriverType:
            if t not in dict.keys():
                dict[t] = []

        return dict

    @staticmethod
    def classify_by_car(
        drivers: list['Driver']
    ) -> Dict[CarType, List['Driver']]:
        dict = {}

        for d in drivers:
            if d.config.car_type in dict.keys():
                dict[d.config.car_type].append(d)
            else:
                dict[d.config.car_type] = [d]

        return dict

    @staticmethod
    def classify_by_lane(
        drivers: list['Driver']
    ) -> Dict[int, List['Driver']]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane.
        """

        dict = {}

        if len(drivers) == 0:
            return dict

        for d in drivers:
            if d.config.lane in dict.keys():
                dict[d.config.lane].append(d)
            else:
                dict[d.config.lane] = [d]

        # For all lanes with no drivers, add an empty list
        for lane in range(max(dict.keys()) + 1):
            if lane not in dict.keys():
                dict[lane] = []

        return dict

    @staticmethod
    def classify_by_id(drivers: list['Driver']) -> Dict[int, 'Driver']:
        """
        Returns a dictionary that maps driver ids to drivers.
        """
        dict = {}

        for d in drivers:
            dict[d.config.id] = d

        return dict

    @staticmethod
    def sort_by_position(
        drivers_by_lane: Dict[int, List['Driver']]
    ) -> List['Driver']:
        """
        Returns a list of drivers sorted by their position on the track.
        """
        # Collect all drivers into a list
        drivers = []
        for drivers_in_lane in drivers_by_lane.values():
            drivers += drivers_in_lane

        # Sort by location
        drivers.sort(key=lambda d: d.config.location)

        return drivers

    @staticmethod
    def sort_by_position_in_lane(
        drivers_by_lane: Dict[int, List['Driver']]
    ) -> Dict[int, List['Driver']]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane, sorted by their position in the lane.
        """
        ret = {}
        for lane, drivers in drivers_by_lane.items():
            ret[lane] = sorted(drivers, key=lambda d: d.config.location)
        return ret

    @staticmethod
    def driver_at_front(
        driver: 'Driver',
        state: Dict[int, Dict[int, 'Driver']],
        lane: Optional[int] = None
    ) -> Union['Driver', None]:
        """
        Returns the driver in front of the given driver in the same lane.
        If the given driver is at the front of the lane, returns None.
        """
        if lane:
            drivers_in_lane = state[lane]
            for d in drivers_in_lane.values():
                if d.config.location > driver.config.location:
                    return d
        else:
            # Get the list of drivers in the same lane
            drivers_in_lane = state[driver.config.lane]

            # Is there a driver in front?
            if driver.config.index == 0:  # It's the first driver
                # No, return None
                return None
            else:
                # Yes, return the driver in front
                try:
                    drivers_in_lane[driver.config.index - 1]
                except KeyError:
                    raise KeyError
                return drivers_in_lane[driver.config.index - 1]

    @staticmethod
    def driver_at_back(
        driver: 'Driver',
        state: Dict[int, Dict[int, 'Driver']],
        lane: Optional[int] = None
    ) -> Union['Driver', None]:
        """
        Returns the driver behind the given driver in the same lane.
        If the given driver is at the back of the lane, returns None.
        """
        if lane:
            drivers_in_lane = state[lane]
            for d in reversed(drivers_in_lane.values()):
                if d.config.location < driver.config.location:
                    return d
        else:
            # Get the list of drivers in the same lane
            drivers_in_lane = state[driver.config.lane]

            # Is there a driver behind?
            if driver.config.index == 0:
                # No, return None
                return None
            else:
                # Yes, return the driver behind
                return drivers_in_lane[driver.config.index - 1]

    @staticmethod
    def drivers_close(
        driver: 'Driver',
        drivers_in_lane: Dict[int, 'Driver'],
        safe_distance: float
    ) -> List['Driver']:
        """
        Returns a list of drivers in the same lane as the given driver
        that are close enough to collide with the given driver.

        Parameters:
            driver: The driver to check for collisions with.
            drivers_in_lane: A list of drivers in the same lane as the
                given driver, indexed by their position in the lane.
        """

        if len(drivers_in_lane) == 0:
            return []

        # Check drivers in front of the given driver until
        # the distance between them is greater than the length
        # of the given driver's car, so no further collisions
        # are possible
        drivers_close = []
        for i in range(0, driver.config.index):
            try:
                drivers_in_lane[i]
            except IndexError:
                print(i)
                print(driver.config.index)
                print(drivers_in_lane)
                print(len(drivers_in_lane))
                raise
            if Driver.distance_between(driver, drivers_in_lane[i]) <\
                    CarType.get_length(driver.config.car_type) + safe_distance:
                drivers_close.append(drivers_in_lane[i])
            else:
                break

        # Check drivers behind the given driver until
        # the distance between them is greater than the length
        # of the given driver's car, so no further collisions
        # are possible
        for i in range(driver.config.index + 1, 1, len(drivers_in_lane)):
            if Driver.distance_between(driver, drivers_in_lane[i]) <\
                    CarType.get_length(driver.config.car_type) + safe_distance:
                drivers_close.append(drivers_in_lane[i])
            else:
                break

        return drivers_close

    @staticmethod
    def drivers_close_to(
        drivers_in_lane: Dict[int, 'Driver'],
        position: float,
        driver_safe_distance: float
    ) -> List['Driver']:
        # Check for each lane
        drivers_close = []
        for driver in drivers_in_lane.values():
            if abs(position - driver.config.location) <\
                    driver_safe_distance:
                drivers_close.append(driver)
            else:
                break
        return drivers_close

    @staticmethod
    def distance_between(front: 'Driver', back: 'Driver') -> float:
        """
        Returns the distance between the back of the front driver and the
        front of the back driver, i.e. the real distance between the
        drivers.

        NOTE: This method assumes that the front driver is ahead
        of the back driver, if that weren't the case then the distance
        would be negative. It doesn't check the lane of the drivers.
        """
        real_front = front.config.location -\
            CarType.get_length(front.config.car_type) / 2
        real_back = back.config.location +\
            CarType.get_length(back.config.car_type) / 2
        return real_front - real_back

    @staticmethod
    def collision(d1: 'Driver', d2: 'Driver') -> bool:
        """
        Returns True if the given drivers are colliding, False otherwise.
        """
        if d1.config.lane != d2.config.lane:
            return False
        else:
            front = d1 if d1.config.location > d2.config.location else d2
            back = d1 if d1.config.location < d2.config.location else d2
            return Driver.distance_between(front, back) < 0

    @staticmethod
    def copy(driver: 'Driver') -> 'Driver':
        """
        Returns a deepcopy of the given driver.
        """
        return Driver(config=copy.deepcopy(driver.config))

    @staticmethod
    def copy_list(drivers: List['Driver']) -> List['Driver']:
        """
        Returns a copy of the given list of drivers.
        """
        return [Driver.copy(d) for d in drivers]
