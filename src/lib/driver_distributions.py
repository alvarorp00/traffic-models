"""
This class contains the driver distributions
for the simulation.

The distributions are defined as static methods
and defines the initialazation function for speeds, positions,
lanes and so on.
"""

from lib.driver import DriverType, Driver, CarType
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union
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
    probs: np.ndarray,
    size: int = 1
) -> np.ndarray:
    """
    Returns a random lane, namely a random integer
    between 0 and n_lanes - 1 with a weighted
    probability distribution.

    Probability distribution is defined by
    a decreasing distribution of probabilities.
    """
    return np.random.choice(
        a=np.arange(n_lanes),
        p=probs,
        size=size,
    )


def speed_initialize_old(
    driver_type: DriverType,
    modifiers: List[float],
    car_type: CarType,
    cars_max_speeds,
    cars_min_speeds,
    size=1,
) -> float:
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

    max_speed = DriverType.get_max_speed(
        driver_type=driver_type,
        modifiers=modifiers,
        car_type=car_type,
        cars_max_speeds=cars_max_speeds
    )

    min_speed = DriverType.get_min_speed(
        driver_type=driver_type,
        modifiers=modifiers,
        car_type=car_type,
        cars_min_speeds=cars_min_speeds
    )

    mean = (driver_type.value / driver_type.RISKY.value) *\
        (max_speed - min_speed) + min_speed

    std = 1 / (np.random.uniform(low=.75, high=1.25))

    rvs = tfp.distributions.Normal(
        loc=mean, scale=std
    ).sample(sample_shape=size).numpy()[0]

    return float(rvs)


def speed_initialize(
    driver_type: DriverType,
    modifiers: List[float],
    car_type: CarType,
    cars_max_speeds: List[float],
    cars_min_speeds: List[float],
    size=1,
) -> float:
    initial_speed_range = [
        DriverType.get_min_speed(
            driver_type=driver_type,
            modifiers=modifiers,
            car_type=car_type,
            cars_min_speeds=cars_min_speeds
        ),
        DriverType.get_max_speed(
            driver_type=driver_type,
            modifiers=modifiers,
            car_type=car_type,
            cars_max_speeds=cars_max_speeds
        )
    ]

    initial_speed = random.uniform(
        *initial_speed_range
    )

    return initial_speed


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
    lanes_density : np.ndarray (float)
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
    lanes_density: np.ndarray = kwargs.get('lanes_density')  # type: ignore
    max_tries = kwargs.get('max_tries', 100)
    safe = kwargs.get('safe', False)

    # Generate for each lane

    assert len(lanes_density) == n_lanes
    assert np.isclose(np.sum(lanes_density), 1.0)

    lane_selection = np.random.choice(
        a=np.arange(n_lanes),
        p=lanes_density,
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
    lanes_density : np.ndarray (float)
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

    1. From lanes_density, get the number of drivers for each lane
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
    lanes_density: np.ndarray = kwargs.get('lanes_density')  # type: ignore
    max_tries = kwargs.get('max_tries', 100)
    safe = kwargs.get('safe', False)

    assert len(lanes_density) == n_lanes
    assert np.isclose(np.sum(lanes_density), 1.0)

    ret = {}

    # Generate number of drivers for each lane

    lane_selection = np.random.choice(
        a=np.arange(n_lanes),
        p=lanes_density,
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
    initial_distribution = tfp.distributions.Uniform(
        low=start, high=end
    )
    if safe is False:
        return (
            _location_initialize_unsafe(
                initial_distribution, size
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
            return (
                _location_initialize_unsafe(
                    initial_distribution, size
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


def risk_overtake_distance(
    driver: 'Driver',
    max_speed_fixed: float,
    max_speed_gap: float,
    driver_safe_distance: float,
    size=1
) -> float:
    """
    Returns a random sample gathered from a halfnormal distribution
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

        Standard deviation --> 1 / (driver.config.speed /
                               CarType.get_max_speed(driver.config.car_type)),
        For std is considered the current speed of the car and
        the maximum speed it can reach, so that the standard deviation
        is 1 meter for the fastest car and increases as the car
        gets slower, which has sense as the

        Size --> The number of samples to be generated. For example, two
                    samples: one to the car in front and one to the car in
                    the left lane.
    """
    mean = DriverType.max_close_distance(
        driver.config.driver_type,
        driver_safe_distance,
    )
    std =\
        1 / (
                driver.config.speed /
                CarType.get_max_speed(
                    driver.config.car_type,
                    max_speed_fixed,
                    max_speed_gap
                )
            )
    rvs = tfp.distributions.HalfNormal(
        scale=std
    ).sample(sample_shape=size).numpy() + mean
    # print(f'[MEAN] {mean} [STD] {std} [DIST] {rvs}')
    return rvs


def safe_lane_change(
    approaching_driver: Driver,
    state: Dict[int, Dict[int, Driver]],
    new_lane: int,
    driver_safe_distance: float,
    driver_view_distance: float,
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
    """

    n_lanes = len(state)
    # Check lane safety
    if new_lane < 0 or new_lane >= n_lanes:
        return False

    # drivers_in_lane does not include the approaching_driver
    drivers_in_lane = state[new_lane]

    # Check lane is within bounds
    if new_lane < 0 or new_lane >= n_lanes:
        return False

    driver_predict_location = approaching_driver.config.location +\
        approaching_driver.config.speed / 3.6

    drivers_in_lane_predict = {
        k: Driver.copy(v) for k, v in drivers_in_lane.items()
    }

    for driver in drivers_in_lane_predict.values():
        driver.config.location += driver.config.speed / 3.6

    __drivers_close = Driver.drivers_close_to(
        drivers_in_lane=drivers_in_lane_predict,
        position=approaching_driver.config.location + driver_predict_location,
        driver_safe_distance=driver_safe_distance,
    )

    # Check that there are no cars close to the driver in the new lane
    if len(__drivers_close) > 0:
        # If they are, see if the ones at front are moving slower
        # and if the ones at back are moving faster

        # Drivers at front are the ones indexed lower than the approaching
        drivers_front = [
            v for k, v in drivers_in_lane.items()
            if k < approaching_driver.config.index
        ]

        # Drivers at back are the ones indexed higher than the approaching
        drivers_back = [
            v for k, v in drivers_in_lane.items()
            if k > approaching_driver.config.index
        ]

        # Check if the front driver is moving slower
        # TODO: check this --> should only check the driver at the front?
        if np.any(
            np.array([driver.config.speed
                      for driver in drivers_front]) > approaching_driver.config.speed  # noqa: E501
        ):
            return False

        # Check if the drivers at back are moving faster
        # TODO: check this --> should only check the driver at the back?
        if np.any(
            np.array([driver.config.speed
                      for driver in drivers_back]) < approaching_driver.config.speed  # noqa: E501
        ):
            return False

    return True


def safe_overtake(
    driver: Driver,
    state: Dict[int, Dict[int, Driver]],
    driver_safe_distance: float,
    driver_view_distance: float,
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
    n_lanes = len(state)

    # Check that the current lane is not the highest priority lane
    if driver.config.lane == n_lanes - 1:
        return False
    # Get driver in front
    front_driver = Driver.driver_at_front(driver, state)
    # Check if driver is moving faster than driver in front
    if front_driver is None:
        return False  # No need to overtake
    if driver.config.speed <= front_driver.config.speed:
        return False
    else:
        # Check distance between driver and driver in front
        distance = front_driver.config.location - driver.config.location
        # Check such distance is greater than the risk overtake distance
        if distance > driver_safe_distance:
            # Car is not close enough to the car in front
            return False
    # Get driver in back
    back_driver = Driver.driver_at_back(driver, state)
    # Check if there's a driver in back
    if back_driver is not None:
        # Check if driver is moving faster than driver in back
        if driver.config.speed < back_driver.config.speed:
            # Driver at back is faster than driver
            # Check distance between driver and driver in back
            distance = driver.config.location - back_driver.config.location
            # Check such distance is greater than the risk overtake distance
            if distance < driver_safe_distance:
                # Car behind is close enough to the car
                # so driver will suppose that car behind
                # is going to overtake too
                return False
    # Check if driver can move to next lane

    return safe_lane_change(
        approaching_driver=driver,
        state=state,
        new_lane=driver.config.lane + 1,
        driver_safe_distance=driver_safe_distance,
        driver_view_distance=driver_view_distance
    )


def speed_update(
    driver: Driver,
    driver_at_front: Optional[Driver],
    driver_at_back: Optional[Driver],
    **kwargs
) -> float:
    # Check if there is a driver in front
    if driver_at_front is not None:
        # Calculate the safe distance based on driver's reaction time
        safe_distance = (driver.config.speed / 3.6) *\
            driver.config.reaction_time.value

        # Calculate the distance to the driver in front
        distance_to_front = driver_at_front.config.location -\
            driver.config.location

        # Calculate the desired speed based on the safe distance
        desired_speed = (
            driver_at_front.config.speed
            if distance_to_front > safe_distance
            else (distance_to_front / driver.config.reaction_time.value) * 3.6
        )

        if desired_speed < driver.config.speed:
            # It means that the driver is going to slow down
            driver.config.brake_counter += 1
    else:
        # If there is no driver in front, use the maximum speed
        desired_speed = DriverType.get_max_speed(
            driver_type=driver.config.driver_type,
            modifiers=kwargs.get('modifiers', None),
            car_type=driver.config.car_type,
            cars_max_speeds=kwargs.get('cars_max_speeds', None)
        )

    # # Check if there is a driver at the back
    # if driver_at_back is not None:
    #     # Adjust the desired speed
    #     desired_speed = min(desired_speed, driver_at_back.config.speed)

    # Adjust the desired speed based on the driver's type
    # You can implement different speed adjustments based on the driver type

    # Apply any other speed adjustments or constraints here

    return min(desired_speed, DriverType.get_max_speed(
            driver_type=driver.config.driver_type,
            car_type=driver.config.car_type,
            **kwargs))


def lane_update(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    driver_safe_distance: float,
    driver_view_distance: float,
    time_in_lane: float,
    time_in_lane_factor: List[float],
    **kwargs
) -> int:
    current_lane = driver.config.lane

    # Update driver's overtaking time
    if driver.config.time_in_lane > 0:
        driver.config.time_in_lane -= 1

    driver_at_front = Driver.driver_at_front(driver, state)

    # Check if the driver at the front crashed (accidented)
    if driver_at_front is not None and driver_at_front.config.accidented:
        # Check if the driver can switch to the opposite lane
        _next_lane = next_lane(current_lane, len(state))
        can_switch_lane = can_switch_to_lane(
            driver, state, _next_lane, driver_safe_distance)
        if can_switch_lane and driver.config.time_in_lane == 0:
            return _next_lane

    # Check if the driver can overtake another driver
    can_overtake = can_overtake_driver(
        driver, state, current_lane,
        driver_safe_distance, driver_view_distance, **kwargs)

    if can_overtake:
        # Can switch lane checks also if there's enough space
        _next_lane = next_lane(current_lane, len(state))
        can_switch_lane = can_switch_to_lane(
            driver, state, _next_lane, driver_safe_distance)
        if can_switch_lane:
            if driver.config.time_in_lane == 0:
                # Driver can switch to the target lane for overtaking
                driver.config.time_in_lane =\
                    DriverType.get_time_in_lane(
                        driver.config.driver_type,
                        time_in_lane,
                        time_in_lane_factor
                    )
                return _next_lane
    else:
        can_return = can_return_lane(
            driver,
            state,
            current_lane,
            driver_safe_distance,
            driver_view_distance
        )

        if can_return:
            return prev_lane(
                current_lane,
                len(state)
            )

    # Driver stays in the current lane
    return current_lane


def can_switch_to_lane(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    new_lane: int,
    driver_safe_distance: float,
) -> bool:
    # Check if there is a driver at the front
    driver_at_front = Driver.driver_at_front(driver, state)
    if driver_at_front is None:
        return False

    # Check if overtaking time has elapsed
    if driver.config.time_in_lane > 0:
        return False

    # Check if there is enough space in the opposite lane
    enough_space = has_enough_space(
        driver, state, new_lane, driver_safe_distance)

    if not enough_space:
        return False

    return True


def next_lane(current_lane: int, n_lanes: int) -> int:
    # Check if the driver is in the leftmost lane
    return current_lane + 1 if current_lane < n_lanes - 1 else current_lane


def prev_lane(current_lane: int, n_lanes: int) -> int:
    # Check if the driver is in the leftmost lane
    return current_lane - 1 if current_lane > 0 else current_lane


def can_overtake_driver(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    current_lane: int,
    driver_safe_distance: float,
    driver_view_distance: float,
    **kwargs
) -> bool:
    # Check if we're in the right lane for overtaking
    if current_lane == len(state) - 1:
        return False

    # Check if there is a driver in front
    driver_at_front = Driver.driver_at_front(driver, state)
    if driver_at_front is None:
        return False

    if driver.config.time_in_lane > 0:
        return False

    # Check if driver at front is at view distance
    if abs(
        driver_at_front.config.location - driver.config.location
    ) > driver_view_distance:
        return False

    # Check if there's a driver in the other lane close to driver
    # so that we can't switch to the other lane
    _next_lane = next_lane(current_lane, len(state))
    enough_space = has_enough_space(
        driver, state, _next_lane, driver_safe_distance)

    if not enough_space:
        return False

    # Check if current speed is greater than the driver in front
    if driver.config.speed > driver_at_front.config.speed:
        return True

    # Check the maximum speed of the driver in front
    # and calculate the maximum speed of the driver
    # to decide probability of overtaking
    driver_max_speed = DriverType.get_max_speed(
        driver_type=driver.config.driver_type,
        car_type=driver.config.car_type,
        **kwargs
    )
    front_max_speed = DriverType.get_max_speed(
        driver_type=driver_at_front.config.driver_type,
        car_type=driver_at_front.config.car_type,
        **kwargs
    )

    # Calculate the probability of overtaking
    probability = (driver_max_speed - front_max_speed) / driver_max_speed

    # Check if the driver can overtake the driver in front
    return random.random() < probability


def can_return_lane(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    current_lane: int,
    driver_safe_distance: float,
    driver_view_distance: float
) -> bool:
    # Check if we're in the right lane for overtaking
    if current_lane == 0:
        return False

    if driver.config.time_in_lane > 0:
        return False

    # Check if there are drivers in the other lane
    drivers_in_other_lane = list(state[current_lane - 1].values())

    reaction_distance = (driver.config.speed / 3.6) *\
        driver.config.reaction_time.value

    if len(drivers_in_other_lane) > 0:
        for other_driver in drivers_in_other_lane:
            if abs(other_driver.config.location - driver.config.location) +\
                    reaction_distance < driver_view_distance:
                return False

    # Check if car behind is too close
    driver_behind = Driver.driver_at_back(driver, state)

    if driver_behind is not None:
        if abs(
            driver_behind.config.location - driver.config.location
        ) < driver_safe_distance:
            if driver_behind.config.speed < driver.config.speed:
                return np.random.choice([True, False], p=[0.5, 0.5])
            else:
                return True  # Always return lane if car behind is faster

    return has_enough_space(
        driver, state, current_lane - 1, driver_safe_distance)


def has_enough_space(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    target_lane: int,
    driver_safe_distance: float
) -> bool:
    # Get the drivers in the target lane
    drivers_in_target_lane = state.get(target_lane, {}).values()

    # Check if there is enough space in the target lane
    for other_driver in drivers_in_target_lane:
        if abs(
            other_driver.config.location - driver.config.location
        ) < driver_safe_distance:
            return False

    return True


def has_overtaking_time_elapsed(
    driver: Driver,
    target_lane: int,
    time_in_lane: float,
    time_in_lane_factor: float
) -> bool:
    # Calculate the overtaking time based on the driver's speed and a factor
    calculated_overtaking_time = time_in_lane_factor[
        driver.config.driver_type.value - 1
    ] * (driver.config.speed / 3.6)

    return calculated_overtaking_time >= time_in_lane


def speed_lane_update(
    driver: Driver,
    state: Dict[int, Dict[int, 'Driver']],
    **kwargs
) -> Driver:

    driver_at_front = Driver.driver_at_front(driver, state)
    driver_at_back = Driver.driver_at_back(driver, state)

    if driver_at_front is not None:
        # Update driver_at_front id in driver if it has changed
        if driver.config.driver_in_front_id != driver_at_front.config.id:
            driver.config.driver_in_front_id = driver_at_front.config.id
            # Reset braking times if driver in front has changed
            driver.config.brake_counter = 0
    else:
        # Reset driver in front
        driver.config.driver_in_front_id = -1
        driver.config.brake_counter = 0

    # Update lane
    lane = lane_update(
        driver,
        state,
        **kwargs
    )

    if lane != driver.config.lane:
        # Update drivers at front and back
        driver_at_front = Driver.driver_at_front(driver, state)
        driver_at_back = Driver.driver_at_back(driver, state)

    # Update speed
    speed = speed_update(
        driver,
        driver_at_front,
        driver_at_back,
        **kwargs
    )

    # print(f'[SPEED VARIATION]: {driver.config.speed} -> {speed}\n')

    # Update driver
    new_driver = Driver.copy(driver)
    new_driver.config.speed = speed
    new_driver.config.lane = lane
    new_driver.config.location += speed / 3.6

    return new_driver
