import logging
import json
from typing import Any, List
from random import Random

from pyrr import Vector3
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

def load_config():
    with open('Assets/revolve2/vr/db/config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()

def save_to_db(dbengine: Engine, generation: Generation) -> None:
    """
    Save the current generation to the database.

    :param dbengine: The database engine.
    :param generation: The current generation.
    """
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()



class ParentSelector(Selector):
    """Selector class for parent selection."""

    rng: np.random.Generator
    offspring_size: int

    def __init__(self, offspring_size: int, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param offspring_size: The offspring size.
        :param rng: The rng generator.
        """
        self.offspring_size = offspring_size
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int_], dict[str, Population]]:
        """
        Select the parents.

        :param population: The population of robots.
        :param kwargs: Other parameters.
        :return: The parent pairs.
        """
        return np.array(
            [
                selection.multiple_unique(
                    selection_size=2,
                    population=[
                        individual.genotype for individual in population.individuals
                    ],
                    fitnesses=[
                        individual.fitness for individual in population.individuals
                    ],
                    selection_function=lambda _, fitnesses: selection.tournament(
                        rng=self.rng, fitnesses=fitnesses, k=2
                    ),
                )
                for _ in range(self.offspring_size)
            ],
        ), {"parent_population": population}


class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param rng: The rng generator.
        """
        self.rng = rng

    def select(
        self, population: Population, **kwargs: Any
    ) -> tuple[Population, dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        if offspring is None or offspring_fitness is None:
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance'."
            )

        original_survivors, offspring_survivors = population_management.generational(
            old_genotypes=[i.genotype for i in population.individuals],
            old_fitnesses=[i.fitness for i in population.individuals],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n, genotypes, fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=fitnesses, k=2
                ),
            ),
        )

        return (
            Population(
                individuals=[
                    Individual(
                        genotype=population.individuals[i].genotype,
                        fitness=population.individuals[i].fitness,
                    )
                    for i in original_survivors
                ]
                + [
                    Individual(
                        genotype=offspring[i],
                        fitness=offspring_fitness[i],
                    )
                    for i in offspring_survivors
                ]
            ),
            {},
        )


class CrossoverReproducer(Reproducer):
    """A simple crossover reproducer using multineat."""

    rng: np.random.Generator
    innov_db_body: multineat.InnovationDatabase

    def __init__(
        self,
        rng: np.random.Generator,
        innov_db_body: multineat.InnovationDatabase,
    ):
        """
        Initialize the reproducer.

        :param rng: The ranfom generator.
        :param innov_db_body: The innovation database for the body.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body

    def reproduce(
        self, population: npt.NDArray[np.int_], **kwargs: Any
    ) -> list[Genotype]:
        """
        Reproduce the population by crossover.

        :param population: The parent pairs.
        :param kwargs: Additional keyword arguments.
        :return: The genotypes of the children.
        :raises ValueError: If the parent population is not passed as a kwarg `parent_population`.
        """
        parent_population: Population | None = kwargs.get("parent_population")
        if parent_population is None:
            raise ValueError("No parent population given.")

        offspring_genotypes = [
            Genotype.crossover(
                parent_population.individuals[parent1_i].genotype,
                parent_population.individuals[parent2_i].genotype,
                self.rng,
                parent_population.individuals[parent1_i].fitness >= parent_population.individuals[parent2_i].fitness,
            ).mutate(self.innov_db_body, self.rng)
            for parent1_i, parent2_i in population
        ]
        return offspring_genotypes
    
class InteractiveReproducer(Reproducer):
    """A simple crossover reproducer using multineat."""

    rng: np.random.Generator
    innov_db_body: multineat.InnovationDatabase
    dbengine: Engine

    def __init__(
        self,
        rng: np.random.Generator,
        innov_db_body: multineat.InnovationDatabase,
        dbengine: Engine,
    ):
        """
        Initialize the reproducer.

        :param rng: The ranfom generator.
        :param innov_db_body: The innovation database for the body.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body
        self.dbengine = dbengine

    def reproduce(self):
        experiment = None
        gen_index = None
        offspring_genotypes: List[Genotype] = []
        # Get the selected parents
        with Session(self.dbengine) as session:
            # Query the latest generation based on its unique id
            latest_generation = session.query(Generation).order_by(Generation.id.desc()).first()
            if not latest_generation:
                raise Exception("The generation table is empty")
            experiment = latest_generation.experiment
            gen_index = latest_generation.generation_index
            parents = session.query(Parents).filter(Parents.parent_gen_id == latest_generation.id).all()
            if not parents:
                raise Exception("The parents table is empty")
            for parent_data in parents:
                parent1_genotype = parent_data.parent1.genotype
                parent2_genotype = parent_data.parent2.genotype if parent_data.parent2 else None
                parent1_fitness = parent_data.parent1.fitness
                parent2_fitness = parent_data.parent2.fitness if parent_data.parent2 else None
                if parent2_genotype:
                    offspring_genotypes.append(Genotype.crossover(
                        parent1_genotype,
                        parent2_genotype,
                        self.rng,
                        parent1_fitness >= parent2_fitness,
                        ).mutate(
                            self.innov_db_body,
                            self.rng
                        )
                    )
                elif parent_data.mutate:
                    offspring_genotypes.append(parent1_genotype.mutate(self.innov_db_body, self.rng))
                else: offspring_genotypes.append(parent1_genotype)
        return offspring_genotypes, experiment, gen_index
        

def run_experiment(dbengine: Engine) -> None:
    """Run the program."""
    # Set up logging.
    logging.info("----------------")
    logging.info("Continue experiment")

    # CPPN innovation databases.
    # If you don't understand CPPN, just know that a single database is shared in the whole evolutionary process.
    # One for body, and one for brain.
    innov_db_body = multineat.InnovationDatabase()
    """
    Here we initialize the components used for the evolutionary process.

    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    evaluator = Evaluator(headless=True, num_simulators=config['NUM_SIMULATORS'])
    learner = CMAESLearner(
        evaluator=evaluator,
        generations=config['CMAES_NUM_GENERATIONS'],
        initial_std=config['CMAES_INITIAL_STD'],
        pop_size=config['CMAES_POP_SIZE'],
        seed=seed_from_time() % 2**32,
    )
    rng=seed_from_time()
    parent_selector = ParentSelector(
        offspring_size=config['OFFSPRING_SIZE'], rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=Random(), innov_db_body=innov_db_body
    )
    interactive_reproducer = InteractiveReproducer(
        rng=Random(), innov_db_body=innov_db_body, dbengine=dbengine
    )
    logging.info("Breeding selected parents")
    children, experiment, generation_index = interactive_reproducer.reproduce()
    child_task_performance = [0.0]
    children, child_task_performance = learner.learn(children)
    assert child_task_performance !=[0.0]

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                children, child_task_performance, strict=True
            )
        ]
    )

    generation = Generation(
            experiment=experiment,
            generation_index=generation_index + 1,
            population=population,
        )
    save_to_db(dbengine, generation)


    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=None,
        reproducer=crossover_reproducer,
        learner=learner,
    )

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    for _ in range(config['STEP_SIZE']-1): # STEP_SIZE indicates how many steps should be taken before the user does parent selection again
        logging.info(f"Generation {generation.generation_index + 1} / {config['STEP_SIZE']}.")
        population = modular_robot_evolution.step(
            population
        )  # Step the evolution forward.

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        save_to_db(dbengine, generation)       
    
    scene = ModularRobotScene(terrain=terrains.flat())
    i = 0
    for individual in population.individuals:
        scene.add_robot(individual.genotype.develop(), pose = Pose(Vector3([i, 0.0, 0.0])))
        i += 2
    simulator = LocalSimulator(viewer_type="custom", headless=True)
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = 90
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

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config['DATABASE_FILE'], open_method=OpenMethod.OPEN_IF_EXISTS
    )

    if dbengine is None:
        raise RuntimeError("Failed to create database engine.")

    run_experiment(dbengine)


if __name__ == "__main__":
    main()
