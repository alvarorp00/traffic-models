"""
Abstract base class for all drivers.

This class defines the interface that all drivers must implement. It
also provides a few utility methods that are useful for all drivers.

The driver interface is designed to be as simple as possible. The
action() method will be called to act on a discrete time step,
where the time step is defined by the simulation.

The action() method would depend on the parameters of the driver
which will be set depending on the type of driver. For example, a
driver that takes risks will consider overtaking other vehicles
within less distance than a driver that is more cautious.

Almost all the methods in this class will be implemented by itself,
but the parameters of the driver will be set by the simulation
and so are required to be passed as arguments to the constructor.
"""

import enum
import random
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats as st
from scipy.spatial.distance import squareform, pdist


class LanePriority(enum.Enum):
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
    SUV = 2
    SEDAN = 3
    MOTORCYCLE = 4

    @staticmethod
    def random(size=1) -> list['CarType']:
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
            weights=[.2, .4, .3, .1],
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
    def get_max_speed(car_type: 'CarType') -> float:
        """
        Returns the maximum speed of the car.

        Parameters
        ----------
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The maximum speed of the car.
        """
        if car_type == CarType.MOTORCYCLE:
            return 130
        elif car_type == CarType.SEDAN:
            return 120
        elif car_type == CarType.SUV:
            return 100
        elif car_type == CarType.TRUCK:
            return 80
        else:
            raise ValueError(f"Invalid car type @ {car_type}")

    @staticmethod
    def get_min_speed(car_type: 'CarType') -> float:
        """
        Returns the minimum speed of the car.

        Parameters
        ----------
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The minimum speed of the car.
        """
        return 60  # All cars have a minimum speed of 60 km/h

    @staticmethod
    def max_speed() -> float:
        """
        Returns the maximum speed of any car.

        Returns
        -------
        float
            The maximum speed of any car.
        """
        return CarType.get_max_speed(CarType.MOTORCYCLE)


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
    def random(size=1) -> list['DriverType']:
        """
        Returns a random driver type.

        Probability distribution is defined by a multinomial
        distribution.

        Probabilities:
        - CAUTIOUS: 0.4
        - NORMAL: 0.3
        - RISKY: 0.15
        - AGGRESSIVE: 0.1
        - RECKLESS: 0.05

        Returns
        -------
        DriverType
            A random driver type.
        """
        choice = random.choices(
            population=list(DriverType),
            weights=[.4, .3, .15, .1, .05],
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


class DriverDistributions:
    @staticmethod
    def lane_initialize(n_lanes: int) -> int:
        """
        Returns a random lane, namely
        a random integer between 0 and n_lanes - 1.

        Parameters
        ----------
        n_lanes : int
            The number of lanes
            of the road.

        Returns
        -------
        int
            A random lane.
        """
        return random.randint(0, n_lanes - 1)

    @staticmethod
    def lane_initialize_weighted(n_lanes: int,
                                 lane_priority: LanePriority,
                                 size: int = 1) -> np.ndarray:
        """
        Returns a random lane, namely a random integer
        between 0 and n_lanes - 1 with a weighted
        probability distribution.

        Probability distribution is defined by
        a decreasing distribution of probabilities.
        """

        # Decreasing distribution of probabilities from
        # left to right, so the left has the highest
        weights = [1/(idx+1) for idx in range(n_lanes)]
        accum = sum(weights)
        weights = [w/accum for w in weights]

        if lane_priority == LanePriority.LEFT:
            # Reverse the probabilities so that the left
            # has the lowest probability --> such lane
            # will be chosen less often
            weights = weights[::-1]

        lanes = np.arange(n_lanes)

        if lane_priority == LanePriority.LEFT:
            # Reverse the lanes so that the left
            # has the highest index and the right
            # the lowest
            lanes = lanes[::-1]

        return np.random.choice(
            a=lanes,
            p=weights,
            size=size,
        )

    @staticmethod
    def speed_initialize(car_type: CarType,
                         driver_type: DriverType, size=1):
        """
        Returns a random rample gathered from a normal distribution
        that represents the initial speed of the driver.

        It considers the type of driver and the type of car.

        Parameters
        ----------
        driver_type : DriverType
        car_type : CarType
        size : int

        Formula:
            mean = (driver_type.value / driver_type.RISKY.value) *\
                     (max_speed - min_speed) + min_speed
            std = 1 / random_between(.75, 1.25)

        Notes:
            If velocity is greater than max_speed, it will be because
            there is a random factor in the formula. So we will
            admit that this is a valid velocity.
        """
        max_speed = CarType.get_max_speed(car_type)
        min_speed = CarType.get_min_speed(car_type)

        portion = (driver_type.value / driver_type.RISKY.value) *\
                  (max_speed - min_speed)
        avg_speeds = portion + min_speed

        mean = avg_speeds
        std = 1 / (np.random.uniform(low=.75, high=1.25))

        rvs = st.norm.rvs(loc=mean, scale=std, size=size)

        return float(rvs)

    @staticmethod
    def lane_location_initialize(
        start: float, end: float,
        size: int, n_lanes: int,
        lane_prio: LanePriority,
        safe_distance: float,
        probs: list[float],
        max_tries: int = 100
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Returns a dictionary with the lane as key and the
        locations as values.

        Parameters
        ----------
        start : float
            The start of the road.
        end : float
            The end of the road.
        size : int
            The number of locations to generate.
        n_lanes : int
            The number of lanes of the road.
        lane_prio : LanePriority
            The lane priority.
        safe_distance : float
            The safe distance between two drivers.
        probs : list[float]
            The probability of each lane.
        max_tries : int
            The maximum number of tries to find a suitable
            location for a driver that satisfies the safe
            distance.

        Returns
        -------
        dict[int, np.ndarray]
            A dictionary with the lane as key and the
            locations as values.

        NOTE: if no suitable location is found for a driver
        (i.e. the driver is too close to another driver
        and no position at a safe distance is found in a lane),
        then the returned value will be None
        """
        ret = {}

        # Generate for each lane

        assert len(probs) == n_lanes
        assert np.sum(probs) == 1

        lane_density = np.array(np.zeros(shape=n_lanes), dtype=int)

        for lane in range(n_lanes):
            lane_density[lane] = int(probs[lane] * size)

        # If the sum of the lane densities is less than the
        # total number of drivers, then we add the difference
        # to the first lane
        if np.sum(lane_density) < size:
            lane_density[0] += size - np.sum(lane_density)

        # Initialize ret with the lanes
        for lane in range(n_lanes):
            ret[lane] = np.zeros(shape=lane_density[lane])

        for i in range(n_lanes):
            # Generate drivers for each lane
            lane_qty = lane_density[i]

            locations = DriverDistributions._location_initialize_safe(
                start=start, end=end,
                size=lane_qty, safe_distance=safe_distance,
                max_tries=max_tries
            )

            if locations is None:
                return None

            # Flatten & sort locations
            ret[i] = np.sort(locations.flatten())

        return ret

    @DeprecationWarning
    @staticmethod
    def location_initialize(start: float, end: float, size: int,
                            safe: bool = False,
                            **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Returns a random position in the road.

        Setting up a safe mode means that the locations
        will be generated so that each of them is at least
        safe_distance away from the other locations.

        Parameters
        ----------
        start : int
            The start of the road.
        end : int
            The end of the road.
        size : int
            The number of locations to generate.
        safe : bool
            Whether to generate safe locations or not.
        max_tries : int
            The maximum number of tries to generate a safe location.
        safe_distance : int
            The minimum distance between each location.

        Returns
        -------
        (float, bool)
            The location and whether it is safe or not.
        """
        if safe is False:
            return (DriverDistributions._location_initialize_unsafe(
                start, end, size
            ), False)
        else:
            # Generate safe
            max_tries = kwargs.get('max_tries', 100)
            safe_distance = kwargs.get('safe_distance', 10)
            res = DriverDistributions._location_initialize_safe(
                    start, end, size, safe_distance, max_tries  # type: ignore
                  )
            if res is None:
                # If we can't find a safe position, we'll return
                # an unsafe one
                return (DriverDistributions._location_initialize_unsafe(
                    start, end, size
                ), False)
            else:
                return (res, True)

    @staticmethod
    def _location_initialize_unsafe(start: float,
                                    end: float,
                                    size: int) -> np.ndarray:
        """
        Returns a random position in the road, with no safe distance
        (i.e. it can be too close to other locations)
        """
        return np.random.uniform(low=start, high=end, size=size)

    @staticmethod
    def _location_initialize_safe(start: float, end: float, size: int,
                                  safe_distance: float,
                                  max_tries: int,
                                  dim: int = 1) -> Union[np.ndarray, None]:
        """
        Returns a random position in the road that is safe.

        Safe distance means that the locations
        will be generated so that each of them is at least
        safe_distance away from the other locations.

        Use the `location_initialize` method instead,
        it'll use this one if the safe_distance
        parameter is not None.

        If it can't find a safe group of positions,
        it'll return None.

        Parameters
        ----------
        start : float
            The start of the road.
        end : float
            The end of the road.
        size : int
            The number of locations to generate.
        safe_distance : float
            The safe distance.
        max_tries : int
            The maximum number of tries to generate a safe location.

        NOTE: each location is a vector of dimension `dim`,
        so be careful when accessing the values, as the returned
        value is a 2-level array, one for the whole locations
        and one for each vector for each location.
        """

        pos = np.random.uniform(
            low=start,
            high=end,
            size=(size, dim)
        )
        dist_matrix = squareform(pdist(pos))
        np.fill_diagonal(dist_matrix, np.inf)
        for i in range(max_tries):
            if np.min(dist_matrix) >= safe_distance:
                return pos
            idx = np.argmin(dist_matrix)
            i, j = np.unravel_index(idx, dist_matrix.shape)  # type: ignore
            vec = pos[i] - pos[j]
            # Ignore possible error
            # (it already puts a NaN in the vector)
            np.seterr(divide='ignore', invalid='ignore')  # type: ignore
            vec /= np.linalg.norm(vec)
            pos[i] += vec * (safe_distance - dist_matrix[i, j]) / 2
            pos[j] -= vec * (safe_distance - dist_matrix[i, j]) / 2
            dist_matrix = squareform(pdist(pos))
            np.fill_diagonal(dist_matrix, np.inf)
            # Restore the error settings
            np.seterr(divide='warn', invalid='warn')  # type: ignore
        return None

    @staticmethod
    def risk_overtake_distance(driver: 'Driver', size=1):
        """
        Returns a random rample gathered from a halfnormal distribution
        (a normal distribution with only right side values so we control
        the minimum distances that a driver will consider overtaking)
        that represents the minimum distance that a driver will
        consider overtaking another vehicle. This distance will be
        considered both for overtaking in front and the car comming
        from behind on left lane.

        It considers the type of driver and the type of car.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        car_type : CarType
            The type of car.
        size : int

        Formula:
            mean = 25 - 5 * (driver_type - 1)
            std = 1 / (CarType.get_max_speed(car_type) / CarType.max_speed())

        Returns
        -------
        ndarray: A random sample from the normal distribution.

        Notes:
            Mean --> 25 meters for CAUTIOUS driver, decreasing by 5 meters
            for each driver type.

            Standard deviation --> 1 /
                                    (CarType.get_max_speed(car_type) /
                                    CarType.max_speed()),
            For std is considered the current speed of the car and
            the maximum speed it can reach, so that the standard deviation
            is 1 meter for the fastest car and increases as the car
            gets slower, which has sense as the

            Size --> The number of samples to be generated. For example, two
                     samples: one to the car in front and one to the car in
                     the left lane.
        """
        MAXIMUM = 25  # 25 meters in the case of driver_type == CAUTIOUS
        GAP = 5  # 5 meters per driver_type
        mean = MAXIMUM - GAP * (
            driver.config.driver_type.value - 1)
        std = 1 / (driver.config.speed
                   /
                   CarType.get_max_speed(driver.config.car_type))
        rvs = st.halfnorm.rvs(loc=mean, scale=std, size=size)
        return rvs

    @staticmethod
    def speed_change(driver: 'Driver', free_space: float,
                     size=1, increase=True):
        """
        Returns a random sample gathered from a halfnormal distribution
        that represents the speed change (increase or decrease) of the
        driver.

        Parameters
        ----------
        driver: Driver
            The driver.
        free_space: float
            The free space at front or back of the driver.
        size: int
            The number of samples to be generated.
        increase: bool
            If True, the speed change will be an increase. If False,
            the speed change will be a decrease.
        """
        current_speed = driver.config.speed
        # TODO


class DriverConfig:
    """
    Configuration for a driver.

    This class is used to configure the parameters of a driver. It
    is used by the simulation to create a new driver.

    Attributes
    ----------
    driver_type : DriverType
        The type of driver.
    car_type : CarType
        The type of car.
    location : float
        The initial location of the driver.
    speed : float
        The initial speed of the driver.
    lane : int
    """
    def __init__(self, id, **kwargs):
        """
        Constructor for the DriverConfig class.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        """
        self.id = id

        if 'driver_type' not in kwargs:
            raise ValueError("driver_type not passed to the constructor")
        self.driver_type = kwargs['driver_type']

        if 'car_type' not in kwargs:
            raise ValueError("car_type not passed to the constructor")
        self.car_type = kwargs['car_type']

        if 'location' in kwargs:
            # assert isinstance(kwargs['location'], float)
            self.location = kwargs['location']
        else:
            self.location = 0

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

    @property
    def driver_type(self) -> DriverType:
        return self._driver_type

    @driver_type.setter
    def driver_type(self, driver_type: DriverType):
        self._driver_type = driver_type

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

    @property
    def config(self) -> DriverConfig:
        return self._config

    @config.setter
    def config(self, config: DriverConfig):
        self._config = config

    def action(self, state, **kwargs):
        # TODO
        return NotImplementedError("action() not implemented yet")
