"""Main script for the example."""

import logging
import pickle
from typing import Any
from random import Random
from copy import deepcopy
import config
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

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng_time_seed, seed_from_time, make_rng
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot.body.base import ActiveHinge


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


def find_best_robot(
    current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population if current_best is None else [current_best] + population,
        key=lambda x: x.fitness,
    )


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
                grid_size=config.GRID_SIZE,
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
    """Run the program."""
    # Set up logging.
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

    """
    Here we initialize the components used for the evolutionary process.

    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)
    learner = CMAESLearner(
        evaluator=evaluator,
        generations=config.CMAES_NUM_GENERATIONS,
        initial_std=config.CMAES_INITIAL_STD,
        pop_size=config.CMAES_POP_SIZE,
        bounds=config.CMAES_BOUNDS,
        seed=seed_from_time() % 2**32,
    )
    parent_selector = ParentSelector(
        offspring_size=config.OFFSPRING_SIZE, rng=rng)
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=Random(), innov_db_body=innov_db_body
    )

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        #evaluator=evaluator,
        reproducer=crossover_reproducer,
        learner=learner,
    )

    # Create an initial population as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            grid_size=config.GRID_SIZE,
            innov_db_body=innov_db_body,
            rng=Random(),
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    initial_genotypes = assert_random_genotypes(innov_db_body, initial_genotypes)

    # print(initial_genotypes[0])
    # breakpoint()

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_genotypes, initial_fitnesses = learner.learn(initial_genotypes)
    # initial_robots = [genotype.develop() for genotype in initial_genotypes]
    # initial_fitnesses = evaluator.evaluate(initial_robots)

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        individuals=[
            Individual(genotype=genotype, fitness=fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
            )
        ]
    )

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    save_to_db(dbengine, generation)

    # print(population[0])
    # breakpoint()

    # # Save the best robot
    # best_robot = find_best_robot(None, population)

    # Set the current generation to 0.
    # generation_index = 0

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}.")

        """
        In contrast to the previous example we do not explicitly stat the order of operations here, but let the ModularRobotEvolution object do the scheduling.
        This does not give a performance boost, but is more readable and less prone to errors due to mixing up the order.

        Not that you are not restricted to the classical ModularRobotEvolution object, since you can adjust the step function as you want.
        """
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

        # # Find the new best robot
        # best_robot = find_best_robot(best_robot, population)

        # logging.info(f"Best robot until now: {best_robot.fitness}")
        # # logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")
        # # Save best robot pickle in file
        # logging.info(f"Genotype pickle for generation: {generation_index + 1}")
        # with open(f'best_robot_gen_{generation_index + 1}.pickle', 'wb') as handle:
        #     pickle.dump(best_robot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Increase the generation index counter.
        # generation_index += 1
    
    # # Save best robot pickle in file
    # logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")
    # with open('best_robot_final.pickle', 'wb') as handle:
    #     pickle.dump(best_robot, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OVERWITE_IF_EXISTS
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
