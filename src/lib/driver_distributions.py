"""
This class contains the driver distributions
for the simulation.

The distributions are defined as static methods
and defines the initialazation function for speeds, positions,
lanes and so on.
"""

from lib.driver import LanePriority, DriverType,\
    Driver, CarType
import numpy as np
import random
from typing import Dict, List, Tuple, Union
from scipy.spatial.distance import squareform, pdist
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

dtype = np.float32


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


def lane_initialize_weighted(
    n_lanes: int,
    size: int = 1
) -> np.ndarray:
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

    lanes = np.arange(n_lanes)

    return np.random.choice(
        a=lanes,
        p=weights,
        size=size,
    )


def speed_initialize(
    driver_type: DriverType,
    car_type: CarType,
    size=1
) -> np.ndarray:
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

    rvs = tfp.distributions.Normal(
        loc=mean, scale=std
    ).sample(sample_shape=size).numpy()

    return rvs


def lane_location_initialize(
    start: float, end: float,
    size: int, **kwargs
) -> Tuple[Dict[int, np.ndarray], bool]:
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
    safe_distance : float
        The safe distance between two drivers.
    lane_density : np.ndarray (float)
        The probability of each lane.
    max_tries : int
        The maximum number of tries to find a suitable
        location for a driver that satisfies the safe
        distance. Default is 100.
    safe : bool
        If True, then the locations will be generated
        in a way that the safe distance is satisfied.
        Otherwise do not consider the safe distance.
        Default is False.

    Locations are generated as follows:
        1. Determine number of locations per lane
        2. Generate a random location in the lane
        3. Check if the location satisfies the safe distance
        4. If not, repeat from 2. until max_tries is reached
        5. If max_tries is reached, then return None
        6. If safe is True, then repeat from 2. until
              all locations are generated for all lanes

    Returns
    -------
    Tuple[Dict[int, np.ndarray], bool]
        The dictionary with the lane as key and the
        locations as values and a boolean that indicates
        if the safe distance was satisfied.

    NOTE: if no suitable location is found for a driver
    (i.e. the driver is too close to another driver
    and no position at a safe distance is found in a lane),
    then the returned value will be None
    """
    ret = {}

    # Get parameters

    n_lanes: int = kwargs.get('n_lanes')  # type: ignore
    safe_distance: float = kwargs.get('safe_distance')  # type: ignore
    lane_density: np.ndarray = kwargs.get('lane_density')  # type: ignore
    max_tries = kwargs.get('max_tries', 100)
    safe = kwargs.get('safe', False)

    # Generate for each lane

    assert len(lane_density) == n_lanes
    assert np.isclose(np.sum(lane_density), 1.0)

    lane_selection = np.random.choice(
        a=np.arange(n_lanes),
        p=lane_density,
        size=size
    )

    lane_bins = np.bincount(lane_selection, minlength=n_lanes)

    # Initialize ret with the lanes
    for lane in range(n_lanes):
        ret[lane] = np.zeros(shape=lane_bins[lane])

    __safe = True

    for i in range(n_lanes):
        # Generate drivers for each lane
        lane_qty = lane_bins[i]

        locations = _location_initialize_safe(
            distribution=tfp.distributions.Uniform(
                low=start, high=end
            ),
            size=lane_qty,  # type: ignore
            safe_distance=safe_distance,
            max_tries=max_tries
        )

        if locations is None and safe:
            locations = _location_initialize_unsafe(
                distribution=tfp.distributions.Uniform(
                    low=start, high=end
                ),
                size=lane_qty,  # type: ignore
            )
            __safe = False
            # Flatten & sort locations
            ret[i] = np.sort(locations.flatten())
        elif locations is None and not safe:
            return ({}, False)
        elif locations is not None:
            # Flatten & sort locations
            ret[i] = np.sort(locations.flatten())

    return (ret, __safe)


def lane_location_initialize_biased(
    start: float, end: float,
    size: int, **kwargs
) -> Tuple[Dict[int, np.ndarray], bool]:
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
    safe_distance : float
        The safe distance between two drivers.
    lane_density : np.ndarray (float)
        The probability of each lane.
    max_tries : int
        The maximum number of tries to find a suitable
        location for a driver that satisfies the safe
        distance. Default is 100.
    safe : bool
        If True, then the locations will be generated
        in a way that the safe distance is satisfied.
        Otherwise do not consider the safe distance.
        Default is False.

    It differs from `lane_location_initialize` in that
    it generates locations in a biased way. The bias
    is given by the density of the lanes, i.e. the probability
    of each lane: first it generates locations for the lane
    with the lowest priority, i.e, the first in the density array,
    then for the second lowest priority and so on.
    The following generations (i.e. all but the first)
    are generated from random sampling from a mixture (normal) distribution
    distribution with equiprobable outcomes from all the drivers
    of the previous generation (the one with lower priority to the
    one being generated).

    Procedure:

    1. From lane_density, get the number of drivers for each lane
    2. Generate locations for the lane with the lowest priority
    3. Generate locations for the lane with the second lowest priority
        from random sampling from a mixture (normal) distribution with
        equiprobable outcomes centered at the locations of the previous
        generation.
    4. Repeat until all lanes are generated.

    Returns
    -------
    Tuple[Dict[int, np.ndarray], bool]
        The dictionary with the lane as key and the
        locations as values and a boolean that indicates
        if the safe distance was satisfied.
    """
    # Retrieve parameters
    n_lanes: int = kwargs.get('n_lanes')  # type: ignore
    safe_distance: float = kwargs.get('safe_distance')  # type: ignore
    lane_density: np.ndarray = kwargs.get('lane_density')  # type: ignore
    max_tries = kwargs.get('max_tries', 100)
    safe = kwargs.get('safe', False)

    assert len(lane_density) == n_lanes
    assert np.isclose(np.sum(lane_density), 1.0)

    ret = {}

    # Generate number of drivers for each lane

    lane_selection = np.random.choice(
        a=np.arange(n_lanes),
        p=lane_density,
        size=size
    )

    lane_bins = np.bincount(lane_selection, minlength=n_lanes)

    # Initialize the first lane

    initial_distribution = tfp.distributions.Uniform(
        low=start, high=end
    )

    if safe:
        ret[0] = _location_initialize_safe(
            distribution=initial_distribution,
            size=lane_bins[0],
            safe_distance=safe_distance,
            max_tries=max_tries
        )
    else:
        ret[0] = _location_initialize_unsafe(
            distribution=initial_distribution,
            size=lane_bins[0],
        )

    if ret[0] is None:
        logging.critical('Could not generate locations for the first lane')
        return ({}, False)

    ret[0] = np.sort(ret[0].flatten())

    # Initialize the rest of the lanes

    # print(f'Bins: {lane_bins}')

    assert len(ret[0]) == lane_bins[0]

    # print(f'Probs: {np.ones(shape=lane_bins[0]) / lane_bins[0]}')
    # print(f'Locs: {ret[0]}')

    ret[0] = np.float32(ret[0])  # type: ignore

    for i in range(1, n_lanes):
        # Build the multivariate normal distribution
        # with equiprobable outcomes
        mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=np.ones(shape=lane_bins[i-1]) / lane_bins[i-1]
            ),
            components_distribution=tfp.distributions.Normal(
                loc=ret[i - 1],
                scale=np.float32(0.1*np.ones(shape=lane_bins[i-1])),
            )
        )
        if safe:
            ret[i] = _location_initialize_safe(
                distribution=mixture,
                size=lane_bins[i],  # type: ignore
                safe_distance=safe_distance,
                max_tries=max_tries
            )

            if ret[i] is None:
                logging.critical(f'Could not generate locations for lane {i}')
                return ({}, False)
        else:
            ret[i] = _location_initialize_unsafe(
                distribution=mixture,
                size=lane_bins[i],  # type: ignore
            )
        ret[i] = np.sort(ret[i].flatten())

    return (ret, True)


@DeprecationWarning
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
        return (_location_initialize_unsafe(
            start, end, size
        ), False)
    else:
        # Generate safe
        max_tries = kwargs.get('max_tries', 100)
        safe_distance = kwargs.get('safe_distance', 10)
        res = _location_initialize_safe(
                start, end, size, safe_distance, max_tries  # type: ignore
                )
        if res is None:
            # If we can't find a safe position, we'll return
            # an unsafe one
            return (_location_initialize_unsafe(
                start, end, size
            ), False)
        else:
            return (res, True)


def _location_initialize_unsafe(
    distribution: tfp.distributions.Distribution,
    size: int,
    dim: int = 1
) -> np.ndarray:
    """
    Returns random positions in the road with no safe distance
    (i.e. it can be too close to other locations)

    Parameters
    ----------
    distribution : tfp.distributions.Distribution
        The distribution to sample from.
    size : int
        The number of locations to generate.
    dim : int
        The dimension of the distribution.
    """
    return distribution.sample(
        sample_shape=(size, dim),
    ).numpy()


def _location_initialize_safe(
    distribution: tfp.distributions.Distribution,
    size: int,
    safe_distance: float,
    max_tries: int,
    dim: int = 1
) -> Union[np.ndarray, None]:
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
    pos = distribution.sample(
        sample_shape=(size, dim)
    ).numpy()

    # print(f'Size: {size}')

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


def max_close_distance(driver_type: DriverType) -> float:
    """
    Returns the maximum a car can be from the car in front of it
    depending on the driver type.
    """
    MAXIMUM = 25  # 25 meters in the case of driver_type == CAUTIOUS
    GAP = 5  # 5 meters per driver_type
    return MAXIMUM - GAP * (driver_type.value - 1)


def risk_overtake_distance(
    driver: 'Driver',
    size=1
) -> float:
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
    driver: Driver
        The driver that will overtake, from
        which we'll get driver & car type.
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
    mean = max_close_distance(driver.config.driver_type)
    std = 1 / (driver.config.speed /
               CarType.get_max_speed(driver.config.car_type))
    rvs = tfp.distributions.HalfNormal(
        scale=std
    ).sample(sample_shape=size).numpy() + mean
    return rvs


def safe_lane_change(
    approaching_driver: Driver,
    drivers_by_lane: Dict[int, List[Driver]],
    new_lane: int,
    risk_distance: float,
) -> bool:
    """
    Returns whether the given driver can safely change lane.

    Proceeds as follows:
    1. Check that the driver is not in the leftmost lane and that the
            lane change is not negative.
    2. Check that there are no cars close to the driver in the new lane.
        2.1 If there are, check that the ones at front are not moving slower
                and that the ones at back are not moving faster.
    3. If all the above is true, return True, else False.

    NOTE: One of the implications of this function is the consideration,
    while overtaking, that no crash will happen if the car in front is faster
    than the actual car (and the car in the back is slower).
    It could occur that car behind is faster than the one in front but not
    faster than the actual car, so it would be safe to overtake
    (both cars would be faster than the front car).

    NOTE: Case when driver is going back to previous lane
    and car behind is faster must be managed, so there is no
    crash and no illegal overtaking (overtaking through right lane)
    """
    # Check lane safety
    if new_lane < 0 or new_lane >= len(drivers_by_lane):
        return False

    location = approaching_driver.config.location
    drivers_in_lane = drivers_by_lane[new_lane]

    # Check lane is within bounds
    if new_lane < 0 or new_lane >= len(drivers_by_lane):
        return False

    # Check that there are no cars close to the driver in the new lane
    if np.any(
        np.array([driver.config.location - location
                  for driver in drivers_in_lane]) < risk_distance
    ):
        # If they are, see if the ones at front are moving slower
        # and if the ones at back are moving faster
        drivers_front = list(filter(
            lambda driver: driver.config.location > location,
            drivers_in_lane
        ))

        drivers_back = list(filter(
            lambda driver: driver.config.location < location,
            drivers_in_lane
        ))

        # Check if the drivers at front are moving slower
        if np.any(
            np.array([driver.config.speed
                      for driver in drivers_front]) > approaching_driver.config.speed  # noqa: E501
        ):
            return False

        # Check if the drivers at back are moving faster
        if np.any(
            np.array([driver.config.speed
                      for driver in drivers_back]) < approaching_driver.config.speed  # noqa: E501
        ):
            return False

    return True


def safe_overtake(
    driver: Driver,
    drivers_by_lane: Dict[int, List[Driver]],
    risk_distance: float
) -> bool:
    """
    Returns whether the given driver can safely overtake the driver

    It checks whether the driver can move to the next lane, i.e.
    if there are no cars close to the driver in the next lane, taking into
    account the type of driver and current speed; and whether the driver
    is moving faster than the driver in front of it.

    NOTE: when a driver is overtaking, it'll consider too the
    driver at the back, so the risk overtake distance of the driver
    at the back considered by the driver will be with the
    driver at the back being of the same type as the driver.
    (THIS MIGHT CHANGE IN THE FUTURE).

    Parameters
    ----------
    driver : Driver
        The driver that wants to overtake.
    drivers_by_lane : Dict[int, List[Driver]]
        A dictionary that maps lane numbers to a list of drivers
        in that lane.
    n_lanes : int
        The number of lanes in the track.
    """
    n_lanes = len(drivers_by_lane)

    # Check that the current lane is not the rightmost lane
    if driver.config.lane == n_lanes - 1:
        return False
    # TODO
    # Get driver in front
    front_driver = Driver.driver_at_front(driver, drivers_by_lane)
    # Check if driver is moving faster than driver in front
    if driver.config.speed <= front_driver.config.speed:
        return False
    else:
        # Check distance between driver and driver in front
        distance = front_driver.config.location - driver.config.location
        # Check such distance is greater than the risk overtake distance
        if distance > risk_distance:
            # Car is not close enough to the car in front
            return False
    # Get driver in back
    back_driver = Driver.driver_at_back(driver, drivers_by_lane)
    # Check if there's a driver in back
    if back_driver is not None:
        # Check if driver is moving faster than driver in back
        if driver.config.speed < back_driver.config.speed:
            # Driver at back is faster than driver
            # Check distance between driver and driver in back
            distance = driver.config.location - back_driver.config.location
            # Check such distance is greater than the risk overtake distance
            if distance < risk_distance:
                # Car behind is close enough to the car
                # so driver will suppose that car behind
                # is going to overtake too
                return False
    # Check if driver can move to next lane

    return safe_lane_change(
        driver,
        drivers_by_lane,
        driver.config.lane + 1,
        risk_distance
    )


class OvertakeDecission:
    """
    Easy way to keep track of the decission made by the driver
    on each step of the simulation.

    NOTE: Might change in the future.
    """
    OVERTAKE = 0,
    STAY = 1,
    BACK = 2

# def safe_go_back(
#         driver: 'Driver',
#         risk_distance: float,
#         drivers_by_lane: Dict[int, List['Driver']],
# ) -> bool:
#     """
#     Returns True if the driver will go back to the previous lane,
#     False otherwise. It will return False if the driver is in the
#     first lane.
#     """
#     return safe_lane_change(
#         driver,
#         drivers_by_lane,
#         driver.config.lane - 1,
#         risk_distance
#     )


def decide_overtake(
        driver: 'Driver',
        drivers_by_lane: Dict[int, List['Driver']],
        risk_distance: float,
) -> OvertakeDecission:
    """
    Returns True if the driver will overtake the driver in front,
    False otherwise.
    """
    # If driver not in first lane, check if driver from right
    # is comming faster and if there is enough space to go back
    # to the right lane
    if safe_overtake(driver, drivers_by_lane, risk_distance) is False:
        if safe_lane_change(
            driver,
            drivers_by_lane,
            driver.config.lane - 1,  # Go back to previous lane?
            risk_distance
        ):
            return OvertakeDecission.BACK
        else:
            return OvertakeDecission.STAY
    else:
        return OvertakeDecission.OVERTAKE


def _speed_increase(driver: Driver) -> float:
    """
    Returns a speed increase for the driver, i.e. a new speed
    """
    std = (driver.config.speed /
           CarType.get_max_speed(driver.config.car_type))
    rvs = tfp.distributions.HalfNormal(
        scale=std
    ).sample(sample_shape=1).numpy().flatten()[0]

    return rvs


def _speed_decrease(driver: Driver) -> float:
    """
    Returns a speed decrease for the driver, i.e. a new speed
    """
    std = (driver.config.speed /
           CarType.get_min_speed(driver.config.car_type))
    rvs = tfp.distributions.HalfNormal(
        scale=std
    ).sample(sample_shape=1).numpy().flatten()[0]
    return rvs


# TODO: add previous functionallity (lane change, speed change, etc.)
# and check behaviour
def speed_update(driver: Driver,
                 drivers_by_lane: Dict[int, List['Driver']],
                 ) -> Driver:
    """
    Returns the updated driver after considering the
    drivers in front and behind, the current speed,
    the distances, the drivers at the left; after
    varying the speed and, if proceeds, the lane.

    It might change the lane if the driver in front is too slow
    and so the driver will change to the left lane to overtake
    if available (no other driver in the left lane).

    If it can't overtake, it'll decrease its speed if it can.

    Procedure:
    1. Check if the driver can overtake the driver in front
       (distance & speed comparison)
        1.1 It can: check if driver at back is faster
            1.1.1. It is: stay, increase speed if enough distance
                    else mantain speed & wait to be overtaken
            1.1.2. It isn't: check if it can change lane to the left
                1.1.2.1. It can: change lane and increase speed
                1.1.2.2. It can't: mantain lane & speed if enough distance
                        else decrease speed & wait to opportunity
                        to change lane
        1.2 It can't: check if it can change lane to the right
            1.2.1 It can: change lane and mantain speed
            1.2.2 It can't: mantain lane & speed
    2. Check if driver can go back to previous lane (if not first lane)
        2.1 It can: mantain speed, switch lane
        2.2 It can't: mantain speed

    NOTE: This function must be tested and possibly changed. For example,
    some considerations should be taken, like the fact that when driver at
    the left is slower, the driver should decrease speed so no illegal overtake
    is done and there's enough space for the driver at the left to switch lane
    back to the right.

    Parameters
    ----------
    driver: Driver
        The driver.
    driver_by_lane: Dict[int, List[Driver]]
    """

    # Retrieve front and back drivers
    driver_front = Driver.driver_at_front(driver, drivers_by_lane)
    driver_back = Driver.driver_at_back(driver, drivers_by_lane)

    # Get risk overtake distance
    risk_distance = risk_overtake_distance(driver)

    # Copy current driver
    new_driver = Driver.copy(driver)

    variation = 0
    lane = driver.config.lane

    __increase = _speed_increase(driver)
    __decrease = _speed_decrease(driver)

    # print(f'Possible speed increase: {__increase}')
    # print(f'Possible speed decrease: {__decrease}')

    __dec = 0

    if driver_front is not None:
        # Can overtake driver in front?
        if safe_overtake(driver, drivers_by_lane, risk_distance):
            # Can overtake driver in back?
            if driver_back is not None and\
                    driver_back.config.speed > driver.config.speed:
                if driver.config.location -\
                        driver_back.config.location > risk_distance:
                    # Increase speed
                    variation = __increase
                    __dec = 1
                else:
                    # Mantain speed
                    __dec = 2
                    pass
            else:
                # Can change lane to the left?
                if safe_lane_change(
                    driver,
                    drivers_by_lane,
                    driver.config.lane + 1,
                    risk_distance
                ):
                    # Change lane and increase speed
                    lane += 1
                    variation = __increase
                    __dec = 3
                else:
                    # Check distance
                    if driver_front.config.location -\
                        driver.config.location >\
                            risk_distance:
                        # mantain speed
                        __dec = 4
                        pass
                    else:
                        # Decrease speed
                        variation = __decrease
                        __dec = 5
        else:  # can't overtake
            # Can change lane to the right?
            if safe_lane_change(
                driver,
                drivers_by_lane,
                driver.config.lane - 1,
                risk_distance
            ):
                # Change lane and mantain speed
                lane -= 1
                __dec = 6
            else:
                # Mantain speed
                __dec = 7
                pass
    else:
        # Can change lane to the right?
        if safe_lane_change(
            driver,
            drivers_by_lane,
            driver.config.lane - 1,
            risk_distance
        ):
            # Change lane and mantain speed
            lane -= 1
            __dec = 8
        else:
            # Mantain speed
            __dec = 9
            pass

    new_driver.config.speed += variation
    new_driver.config.lane = lane

    # if new_driver.config.speed != driver.config.speed:
    #     print(f'Driver {driver.config.id} changed speed from '
    #           f'{driver.config.speed}@{driver.config.lane} to {new_driver.config.speed}@{new_driver.config.lane}')  # noqa: E501

    # print(f'\tDec: {__dec}\n')

    return new_driver
