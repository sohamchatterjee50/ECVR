"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots_v2 import gecko_v2, snake_v2, spider_v2, ant_v2
from revolve2.ci_group.modular_robots_v1 import queen_v1


DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 20
INITIAL_STD = 0.5
NUM_GENERATIONS = 50
BODIES = [spider_v2(), gecko_v2(), snake_v2(), ant_v2(), queen_v1()]
