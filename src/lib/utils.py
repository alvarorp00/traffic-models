from lib.driver import Driver, DriverType, CarType
from typing import List, Dict


class Utils:
    @staticmethod
    def classify_by_driver(drivers: list[Driver]) ->\
            Dict[DriverType, List[Driver]]:
        dict = {}

        for d in drivers:
            if d.config.driver_type in dict.keys():
                dict[d.config.driver_type].append(d)
            else:
                dict[d.config.driver_type] = [d]

        return dict

    @staticmethod
    def classify_by_car(drivers: list[Driver]) -> Dict[CarType, List[Driver]]:
        dict = {}

        for d in drivers:
            if d.config.car_type in dict.keys():
                dict[d.config.car_type].append(d)
            else:
                dict[d.config.car_type] = [d]

        return dict

    @staticmethod
    def classify_by_lane(drivers: list[Driver]) -> Dict[int, List[Driver]]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane.
        """

        dict = {}

        for d in drivers:
            if d.config.lane in dict.keys():
                dict[d.config.lane].append(d)
            else:
                dict[d.config.lane] = [d]

        return dict

    @staticmethod
    def sort_by_position(
        drivers_by_lane: Dict[int, List[Driver]]
    ) -> List[Driver]:
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
        drivers_by_lane: Dict[int, List[Driver]]
    ) -> Dict[int, List[Driver]]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane, sorted by their position in the lane.
        """
        ret = {}
        for lane, drivers in drivers_by_lane.items():
            ret[lane] = sorted(drivers, key=lambda d: d.config.location)
        return ret
