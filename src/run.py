"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from lib.engine import Engine, ModelStats, RunConfig
from config import Config
import tests


def run():
    """
    Runs the simulation
    """

    # Map every configuration parameter in lib.config to a RunConfig attribute
    config = Config()
    # run_config = RunConfig(**config.__dict__)

    # Get all the members of the Config class
    members = [attr for attr in dir(config) if not callable(getattr(config, attr)) and not attr.startswith("__")]

    # Lowcase all the members
    members_lw = [member.lower() for member in members]

    # associate each member with its value
    members_values = [getattr(config, member) for member in members]

    # Create a dictionary with the members and their values
    run_config_dict = dict(zip(members_lw, members_values))

    # Create a RunConfig object with the dictionary
    run_config = RunConfig(**run_config_dict)

    engine = Engine(run_config)
    engine.run()

    model_stats = ModelStats(model=engine.model)
    stats = model_stats.get_stats()

    print(stats)

    # Just test for now

    # test(run_config=run_config)


def test(run_config: RunConfig):
    # Call the test functions here
    # tests.test_distances(run_config=run_config, plot=True)
    # tests.test_velocities(run_config=run_config, plot=True)
    # tests.test_locations(run_config=run_config, plot=True)
    # tests.test_overtake(run_config=run_config, plot=True)
    pass


if __name__ == "__main__":
    run()
