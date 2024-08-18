import logging
from typing import Any, List
from random import Random

from pyrr import Vector3
import config
import multineat
import numpy as np
import numpy.typing as npt

from database_components import (
    Generation,
    Genotype,
    Individual,
    Population,
    Parents,
)

from evaluator import Evaluator
from learner import CMAESLearner

from revolve2.ci_group import terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.simulation.scene._pose import Pose
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator

def main():
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    if dbengine is None:
        raise RuntimeError("Failed to create database engine.")

    with Session(dbengine) as session:
        latest_generation = session.query(Generation).order_by(Generation.id.desc()).first()
        if not latest_generation:
            raise Exception("The generation table is empty")
        population = latest_generation.population
        parents = Parents(
            parent1=population.individuals[0],
            parent2=population.individuals[1],
            parent_gen_id=latest_generation.id,
        )
        session.add(parents)
        session.commit()

if __name__ == "__main__":
    main()
