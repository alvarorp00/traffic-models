# Add required imports
import logging
from typing import List, Dict, Set
from lib.driver import CarType, Driver, DriverType
from lib.engine import Engine
import enum


class StatsItems(enum.Enum):
    LANE_CHANGES = 0
    DRIVERS_FINISHED_DRV_TYPE = 1
    AVG_TIME_TAKEN = 2
    AVG_STARTING_POSITION = 3
    DRIVERS_FINISHED_DRV_CAR_TYPE = 4
    SPEED_CHANGES = 5
    DRIVERS_ACCIDENTED_BY_DRV_TYPE = 6
    CARS_BY_DRV_TYPE = 7


class Stats:
    def __init__(self, *args, **kwargs):
        if 'engine' not in kwargs:
            logging.critical("Cannot create ModelStats without a model.")
            raise Exception()
        self.engine = kwargs['engine']

    def save_to_file(self, file_path: str):
        """
        Saves the stats to a file.

        Parameters
        ----------
        file_path : str
            The path to the file.
        """
        stats = self.get_stats()
        with open(file_path, 'w') as f:
            f.write('Average time\n')
            for d, at in stats[StatsItems.AVG_TIME_TAKEN].items():
                f.write(f"\tDriver {d} avg time: {at}\n")
                # Print how many drivers of each type finished
                f.write(f"\t\t{stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE][d]} {d} drivers finished\n")
                # Print the number of cars of each type that finished for this driver type
                for car_type, num in stats[StatsItems.DRIVERS_FINISHED_DRV_CAR_TYPE][d].items():
                    f.write(f"\t\t\t{num} {car_type} cars finished\n")
            f.write('\n')

            f.write('Average starting position\n')
            for d, asp in stats[StatsItems.AVG_STARTING_POSITION].items():
                f.write(f"\tDriver {d} avg starting position: {asp}\n")
            f.write('\n')

            f.write('Total cars by driver type\n')
            for d, c in stats[StatsItems.CARS_BY_DRV_TYPE].items():
                f.write(f"\tDriver {d} total cars: {c}\n")
            f.write('\n')

            f.write('Accidents by driver type\n')
            for d, a in stats[StatsItems.DRIVERS_ACCIDENTED_BY_DRV_TYPE].items():
                f.write(f"\tDriver {d} accidents: {a}\n")
            f.write('\n')

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
        # Get all drivers
        __all_drivers = self.engine.model.all_active_drivers()
        __all_drivers.extend(self.engine.model.inactive_drivers.values())

        # Initialize as dict of lists
        lane_changes = {driver.config.id: [] for driver in __all_drivers}

        # Iterate over all time steps
        # and check the lane for each step, if it is different
        # from the previous one, then add it to the list

        for t in range(self.engine.run_config.time_steps):
            # Get the trace for the current time step
            __trace_data = self.engine.trace.data[t]
            __all_trace_drivers = __trace_data.all_active_drivers()

            # Iterate over all drivers
            for driver in __all_trace_drivers:
                # Check if id is in the dictionary
                if driver.config.id not in lane_changes:
                    # Add it
                    lane_changes[driver.config.id] = []

                # Check if the lane list is empty
                if len(lane_changes[driver.config.id]) == 0:
                    # Append the current lane
                    lane_changes[driver.config.id].append(
                        driver.config.lane
                    )
                else:
                    # Check if the lane is different from the previous one
                    if lane_changes[driver.config.id][-1] !=\
                            driver.config.lane:
                        # Append the current lane
                        lane_changes[driver.config.id].append(
                            driver.config.lane
                        )

        # Sanitize is not necessary since
        # the condition above (len(...) == 0) already
        # takes care of it

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
        avg_time_taken = {
            driver_type: 0 for driver_type in DriverType
        }
        for driver in self.engine.model.inactive_drivers.values():
            if driver.config.driver_type not in avg_time_taken:
                avg_time_taken[driver.config.driver_type] = 0
            avg_time_taken[driver.config.driver_type] +=\
                self.engine.model.time_taken[driver.config.id]
        for driver_type in avg_time_taken:
            avg_time_taken[driver_type] /=\
                len(Driver.classify_by_type(
                    list(self.engine.model.inactive_drivers.values())
                )[
                    driver_type
                ]) | 1
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
        for driver in self.engine.model.inactive_drivers.values():
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
            Driver.classify_by_type(
                list(self.engine.model.inactive_drivers.values())
            )
        for driver_type in avg_starting_position:
            if len(classified_by_type[driver_type]) == 0:
                avg_starting_position[driver_type] = 0
            else:
                avg_starting_position[driver_type] /=\
                    len(classified_by_type[
                        driver_type
                    ])
        return avg_starting_position

    def _get_drivers_finished_by_drv_type(self) -> Dict[DriverType, int]:
        """
        Returns the number of drivers that finished the simulation
        for each driver type.
        """
        drivers_finished = {}
        for driver in self.engine.model.inactive_drivers.values():
            if driver.config.driver_type not in drivers_finished:
                drivers_finished[driver.config.driver_type] = 0
            drivers_finished[driver.config.driver_type] += 1
        # Sanitize the dictionary
        for driver_type in DriverType:
            if driver_type not in drivers_finished:
                drivers_finished[driver_type] = 0
        return drivers_finished

    def _get_drivers_finished_by_drv_car_type(self) -> Dict[Driver, Dict[CarType, int]]:
        """
        Returns the number of drivers classified by car for each driver type.
        that finished the simulation.
        """
        drivers_by_car_finished = {
            driver_type: {
                car_type: 0 for car_type in CarType
            } for driver_type in DriverType
        }

        for driver in self.engine.model.inactive_drivers.values():
            drivers_by_car_finished[driver.config.driver_type][driver.config.car_type] += 1

        return drivers_by_car_finished

    def _get_speed_changes(self) -> Dict[Driver, List[float]]:
        """
        Returns a dictionary with the speed changes of each driver

        Similar to _get_lane_changes but with speeds
        during the simulation.
        """

        # Get all drivers
        __all_drivers = self.engine.model.all_active_drivers()
        __all_drivers.extend(self.engine.model.inactive_drivers.values())

        # Initialize as dict of lists
        speed_changes = {driver.config.id: [] for driver in __all_drivers}

        # Iterate over all time steps
        # and check the speed for each step, if it is different
        # from the previous one, then add it to the list

        for t in range(self.engine.run_config.time_steps):
            # Get the trace for the current time step
            __trace_data = self.engine.trace.data[t]
            __all_trace_drivers = __trace_data.all_active_drivers()

            # Iterate over all drivers
            for driver in __all_trace_drivers:
                # Check if id is in the dictionary
                if driver.config.id not in speed_changes:
                    # Add it
                    speed_changes[driver.config.id] = []

                # Check if the speed list is empty
                if len(speed_changes[driver.config.id]) == 0:
                    # Append the current speed
                    speed_changes[driver.config.id].append(
                        driver.config.speed
                    )
                else:
                    # Check if the speed is different from the previous one
                    if speed_changes[driver.config.id][-1] !=\
                            driver.config.speed:
                        # Append the current speed
                        speed_changes[driver.config.id].append(
                            driver.config.speed
                        )

        # Sanitize is not necessary since
        # the condition above (len(...) == 0) already
        # takes care of it

        return speed_changes

    def _get_number_of_drvs_accidented(self) -> Dict[DriverType, int]:
        """
        Returns the number of cars accidented for each driver type.
        """
        number_of_drvs_accidented = {
            driver_type: 0 for driver_type in DriverType
        }
        trace_data = self.engine.trace.data
        driver_set: Set[int] = set()
        for t in range(self.engine.run_config.time_steps):
            for driver in trace_data[t].all_active_drivers():
                if driver.config.id in driver_set:
                    continue
                if driver.config.driver_type not in number_of_drvs_accidented:
                    number_of_drvs_accidented[driver.config.driver_type] = 0
                if driver.config.accidented:  # Driver accidented
                    number_of_drvs_accidented[driver.config.driver_type] += 1
                    # We don't want to count the same driver twice
                    driver_set.add(driver.config.id)
        return number_of_drvs_accidented

    def _get_number_of_cars_by_drv_type(self) -> Dict[DriverType, int]:
        """
        Returns the number of cars by driver type even if they have
        finished or not.
        """
        number_of_cars_by_drv_type = {
            driver_type: 0 for driver_type in DriverType
        }
        driver_id_set: Set[int] = set()

        trace_data = self.engine.trace.data
        for t in range(self.engine.run_config.time_steps):
            data_t = trace_data[t].all_active_drivers()
            for driver in data_t:
                if driver.config.id in driver_id_set:
                    continue
                if driver.config.driver_type not in number_of_cars_by_drv_type:
                    number_of_cars_by_drv_type[driver.config.driver_type] = 0
                number_of_cars_by_drv_type[driver.config.driver_type] += 1
                driver_id_set.add(driver.config.id)

        return number_of_cars_by_drv_type

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

            'lane_changes' : Dict[DriverType, int]
                The number of lane changes for each driver type.

            'drivers_finished_drv_type' : Dict[DriverType, int]
                The number of drivers that finished the simulation
                for each driver type.

            'drivers_finished_drv_car_type' : Dict[DriverType, Dict[CarType, int]]
                The number of drivers that finished the simulation
                for each driver type and car type.

            'speed_changes' : Dict[DriverType, List[float]]
                The speed changes for each driver type.

            'drivers_accidented_by_drv_type' : Dict[DriverType, int]
                The number of cars accidented for each driver type.

            'cars_by_drv_type' : Dict[DriverType, int]
                The number of cars by driver type even if they have
                finished or not.
        """
        stats = {}

        stats[StatsItems.AVG_STARTING_POSITION] = self._avg_starting_position()
        stats[StatsItems.AVG_TIME_TAKEN] = self._get_avg_time_taken()
        stats[StatsItems.LANE_CHANGES] = self._get_lane_changes()
        stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE] =\
            self._get_drivers_finished_by_drv_type()
        stats[StatsItems.DRIVERS_FINISHED_DRV_CAR_TYPE] =\
            self._get_drivers_finished_by_drv_car_type()
        stats[StatsItems.SPEED_CHANGES] = self._get_speed_changes()
        stats[StatsItems.DRIVERS_ACCIDENTED_BY_DRV_TYPE] =\
            self._get_number_of_drvs_accidented()
        stats[StatsItems.CARS_BY_DRV_TYPE] =\
            self._get_number_of_cars_by_drv_type()

        return stats

    @staticmethod
    def append_simulation_results_to_file(
            stats: Dict,
            file_path: str,
            simulation_number: int
    ):
        with open(file_path, 'a+') as f:
            driver_types: List[DriverType] = list(DriverType)
            if simulation_number == 0:
                # Clear file
                f.seek(0)
                for driver_type in driver_types:
                    f.write(f' | {driver_type}(avg.time)')
                for driver_type in driver_types:
                    f.write(f' | {driver_type}(n.accidents)')
                for driver_type in driver_types:
                    f.write(f' | {driver_type}(n.drivers)')
                f.write('\n')

            f.write(f'{simulation_number}')
            for driver_type in driver_types:
                f.write(f' | {stats[StatsItems.AVG_TIME_TAKEN][driver_type]}')
            for driver_type in driver_types:
                f.write(f' | {stats[StatsItems.DRIVERS_ACCIDENTED_BY_DRV_TYPE][driver_type]}')
            for driver_type in driver_types:
                f.write(f' | {stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE][driver_type]}')
            f.write('\n')
