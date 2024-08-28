from copy import deepcopy
import logging
import json
from random import Random
from typing import Any
from pyrr import Vector3

import multineat
import numpy as np
import numpy.typing as npt
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from evaluator import Evaluator
from learner import CMAESLearner
from revolve2.modular_robot.body.base._active_hinge import ActiveHinge
from revolve2.modular_robot.brain.cpg._make_cpg_network_structure_neighbor import active_hinges_to_cpg_network_structure_neighbor
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed, seed_from_time
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.ci_group import terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.simulation.scene import Pose

def load_config():
    with open('Assets/revolve2/vr/db/config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()

def assert_random_genotypes(innov_db_body, initial_genotypes):
    final_init_genotypes = []
    for genotype in initial_genotypes:
        temp_genotype = deepcopy(genotype)
        body = temp_genotype.develop_body()
        active_hinges = body.find_modules_of_type(ActiveHinge)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        new_body_genotype = None
        empty_bodies = 0
        while len(active_hinges) == 0 and cpg_network_structure.num_connections == 0:
            empty_bodies += 1
            new_body_genotype = Genotype.random(
                grid_size=config['GRID_SIZE'],
                innov_db_body=innov_db_body,
                rng=Random(),
            )
            temp_temp_genotype = deepcopy(new_body_genotype)
            body = temp_temp_genotype.develop_body()
            active_hinges = body.find_modules_of_type(ActiveHinge)
            (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        if new_body_genotype is not None:
            final_init_genotypes.append(new_body_genotype)
        else:
            final_init_genotypes.append(genotype)
    assert len(final_init_genotypes) == len(initial_genotypes)
    return final_init_genotypes

def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng = make_rng_time_seed()
    rng_exp = seed_from_time()

    # CPPN innovation databases.
    # If you don't understand CPPN, just know that a single database is shared in the whole evolutionary process.
    # One for body, and one for brain.
    innov_db_body = multineat.InnovationDatabase()

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_exp)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # evaluator: Allows us to evaluate a population of modular robots.
    evaluator = Evaluator(headless=True, num_simulators=config['NUM_SIMULATORS'])

    # Create an initial population, as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            grid_size=config['GRID_SIZE'],
            innov_db_body=innov_db_body,
            rng=Random(),
        )
        for _ in range(config['POPULATION_SIZE'])
    ]

    evaluator = Evaluator(headless=True, num_simulators=config['NUM_SIMULATORS'])
    learner = CMAESLearner(
        evaluator=evaluator,
        generations=config['CMAES_NUM_GENERATIONS'],
        initial_std=config['CMAES_INITIAL_STD'],
        pop_size=config['CMAES_POP_SIZE'],
        bounds=config['CMAES_BOUNDS'],
        seed=seed_from_time() % 2**32,
    )

    initial_genotypes = assert_random_genotypes(innov_db_body, initial_genotypes)

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_genotypes, initial_fitnesses = learner.learn(initial_genotypes)

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
            )
        ]
    )
    logging.info(f"Population: {population}")

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()
    
    scene = ModularRobotScene(terrain=terrains.flat())
    for i, individual in enumerate(population.individuals):
        scene.add_robot(individual.genotype.develop(), pose = Pose(Vector3([float(i), 0.0, 0.0])))
    simulator = LocalSimulator(viewer_type="custom", headless=True)
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = 120
    batch_parameters.simulation_timestep = 0.01
    batch_parameters.sampling_frequency = 10
    simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
        vr=True,
    )
    

def main() -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, overwrite if already exists.
    dbengine = open_database_sqlite(
        config['DATABASE_FILE'], open_method=OpenMethod.OVERWITE_IF_EXISTS
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)    
    run_experiment(dbengine)

if __name__ == "__main__":
    main()