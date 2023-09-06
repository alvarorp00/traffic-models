"""
Run 1

- 500m section length
- 5000m road length
- 1000 time steps
- 2 lanes
- Population size: 10 -> 100 (step 10)
- Drivers distribution: (distributed around normal type)
    · 100% risky
"""

import sys
from typing import Dict, List
# setting path
sys.path.append('../src')

from lib.engine import Engine, RunConfig
from lib.stats import Stats, StatsItems
from lib.driver import DriverType 
import lib.graphics as graphics
from config import Config
import numpy as np
import time


REPEATS = 200  # Number of times to repeat the simulation for each configuration


def run():
    results: Dict[int, List[Dict]] = {}

    config = Config()
    run_config = RunConfig.get_run_config(config)

    run_config.section_length = 500
    run_config.road_length = 5000
    run_config.population_size = 10
    run_config.minimum_load_factor = 1.0
    run_config.n_lanes = 2
    run_config.time_steps = 1000
    run_config.driver_type_density = [0.0, 0.0, 1.0, 0.0, 0.0]
    run_config.verbose = False
    run_config.start_with_population = False
    run_config.accident_max_threshold = 0.10

    population_sizes = [
        a for a in range(10, 110, 10)
    ]

    # Get the start time
    start_time = time.time()

    for i in range(0, 10):
        run_config.population_size = population_sizes[i]

        sim_results = []

        for j in range(0, REPEATS):
            engine = Engine(run_config)
            try:
                engine.run()
            except Exception as e:
                print(f'\tSimulation {i+1} [{j+1} / {REPEATS}] failed. Discarding...')
                print(e)
                continue

            # Check if the simulation was valid
            if not engine.validate_simulation():
                print(f'\tSimulation {i+1} [{j+1} / {REPEATS}] was not valid. Discarding...')
                continue
            else:
                print(f'\tSimulation {i+1} [{j+1} / {REPEATS}] was valid')

            stats = Stats(engine=engine)
            # stats_dict = stats.get_stats()
            stats_dict = stats.get_stats()
            sim_results.append(stats_dict)
        results[i] = sim_results

    # Get the end time
    end_time = time.time()

    # Print the time taken
    time_taken = end_time - start_time
    # Save the time taken to a file
    with open('runs/run_3/txt/time_taken.txt', 'w+') as f:
        f.write(f'Time taken: {time_taken / 3600} hours')
        f.write('\n')
        f.write(f'Time taken: {(time_taken - (int(time_taken / 3600) * 3600)) / 60} minutes')
        f.write('\n')
        f.write(f'Time taken: {(time_taken - (int(time_taken / 60) * 60))} seconds')

    # Process results
    medians = []
    for i in range(0, len(results)):
        sim_results = results[i]
        if len(sim_results) == 0:
            medians.append([])
            continue
        median_time_by_driver_type = {
            driver_type: [] for driver_type in DriverType
        }
        median_accidents_by_driver_type = {
            driver_type: [] for driver_type in DriverType
        }
        median_completed_by_driver_type = {
            driver_type: [] for driver_type in DriverType
        }
        for j in range(0, len(sim_results)):
            stats = sim_results[j]
            for driver_type in DriverType:
                median_time_by_driver_type[driver_type].append(
                    stats[StatsItems.AVG_TIME_TAKEN][driver_type])
                median_accidents_by_driver_type[driver_type].append(
                    stats[StatsItems.DRIVERS_ACCIDENTED_BY_DRV_TYPE][driver_type])
                median_completed_by_driver_type[driver_type].append(
                    stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE][driver_type])
        median_time_by_driver_type = {
            driver_type: np.median(median_time_by_driver_type[driver_type]) for driver_type in DriverType
        }
        median_accidents_by_driver_type = {
            driver_type: np.median(median_accidents_by_driver_type[driver_type]) for driver_type in DriverType
        }
        median_completed_by_driver_type = {
            driver_type: np.median(median_completed_by_driver_type[driver_type]) for driver_type in DriverType
        }
        medians.append([
            median_time_by_driver_type,
            median_accidents_by_driver_type,
            median_completed_by_driver_type
        ])

    for i in range(0, len(results)):
        for j in range(0, len(results[i])):
            sim_results = results[i][j]
            Stats.append_simulation_results_to_file(
                sim_results, f'runs/run_3/txt/results_{i}.txt',
                j
            )

    for i in range(0, len(medians)):
        median = medians[i]
        if len(median) == 0:
            continue
        graphics.plot_avg_time_and_accidents(
            median, f'runs/run_3/img/avg_time_and_accidents_{i}.png',
            pop_size=population_sizes[i],
            section_length=run_config.section_length,
            road_length=run_config.road_length,
            )


if __name__ == '__main__':
    print('Running run_3.py')
    run()
    print('Finished run_3.py')
