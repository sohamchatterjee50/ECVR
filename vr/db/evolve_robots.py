import logging
from typing import Any
from pyrr import Vector3

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
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.evolution import ModularRobotEvolution
from revolve2.experimentation.evolution.abstract_elements import Reproducer, Selector
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.rng import make_rng, seed_from_time
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.ci_group import terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters

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

        original_survivors, offspring_survivors = population_management.steady_state(
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

        :param rng: The random generator.
        :param innov_db_body: The innovation database for the body.
        """
        self.rng = rng
        self.innov_db_body = innov_db_body

    def reproduce(
        self, **kwargs: Any
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
        offspring_genotypes: List[Genotype] = []
        for parent_data in parent_population:
            parent1_genotype, parent2_genotype, fitness1, fitness2, mutate_flag = parent_data
            if parent2_genotype:
                offspring_genotype.append(Genotype.crossover(
                parent1_genotype,
                parent2_genotype,
                fitness1 >= fitness2
                self.rng
                ).mutate(self.innov_db_body, self.rng))
            else if mutate_flag:
                offspring_genotype.append(parent1_genotype).mutate((self.innov_db_body, self.rng))
            else:
                offspring_genotype.append(parent1_genotype)
        return offspring_genotypes

def main() -> None:
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    parent_pairs = []
    # Get the selected parents
    with Session(dbengine) as session:
        # Query the latest generation based on its unique id
        latest_generation = session.query(Generation).order_by(Generation.id.desc()).first()
        if not latest_generation:
            raise Exception("The generation table is empty")
        parents = session.query(Parents).filter(parent_gen_id == latest_generation.id)
        if not parents:
            raise Exception("The parents table is empty")
        for parent_pair in parents:
            parent_pairs.append([
                parent_pair.parent1.genotype,
                parent_pair.parent2.genotype if parent_pair.parent2 else None,
                parent_pair.parent1.fitness,
                parent_pair.parent2.fitness if parent_pair.parent2 else None,
                parent_pair.mutate # This only matters if there is only one parent
            ])
    
        

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

if __name__ == "__main__":
    main()