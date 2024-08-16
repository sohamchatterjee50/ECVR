""" CMA-ES learner """

from database_components import Genotype

from revolve2.experimentation.evolution.abstract_elements import Learner, Evaluator
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic
from revolve2.modular_robot import ModularRobot

from copy import deepcopy

import cma
import logging
import numpy as np
import math


class CMAESLearner(Learner):

    _reward_function: Evaluator
    _options: cma.CMAOptions
    _initial_std: float
    _generations: int
    _grid_size = 22

    def __init__(self,
                 evaluator: Evaluator,
                 generations: int,
                 initial_std: float, pop_size: int, bounds: list[float], seed):
        self._reward_function = evaluator
        self._options = {
            "popsize": pop_size,
            "seed": seed,
            "verbose": -1
        }
        self._generations = generations
        self._initial_std = initial_std

    def _relative_pos(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        mapping = {(1, 0): 1, (1, 1): 2, (0, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6,
                   (-1, 1): 7, (1, -1): 8, (2, 0): 9, (0, 2): 10, (-2, 0): 11, (0, -2): 12, (0, 0): 13}

        return mapping[(dx, dy)]

    def learn(self, population: list[Genotype]) -> tuple[list[Genotype], list[float]]:
        """
        Optimize the brain using CMA-ES algorithm

        :param population: The population of robots.
        :return: The population of robots after optimization.
        """
        logging.info(
            "Start brain learner optimization process for the population")

        child_task_performance = []
        new_genotypes = []

        total_no_0_active_hinges_robot = 0
        total_no_1_active_hinges_robot = 0

        for genotype in population:
            temporary_genotype = deepcopy(genotype)
            body = temporary_genotype.develop_body()
            active_hinges = body.find_modules_of_type(ActiveHinge)
            (
                cpg_network_structure,
                output_mapping,
            ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
            

            if len(active_hinges) == 0 and cpg_network_structure.num_connections == 0:
                new_genotypes.append(deepcopy(genotype))
                child_task_performance.append(-150.0)
                logging.info("No Active Hinges or CPG Connections in the body, fitness marked to -100.0")
                total_no_0_active_hinges_robot += 1
                continue


            brain_params = []
            for hinge in active_hinges:
                pos = body.grid_position(hinge)
                cpg_idx = int(pos[0] + pos[1] *
                              self._grid_size + self._grid_size**2 / 2)
                brain_params.append(temporary_genotype.brain[
                    cpg_idx*14
                ])

            for connection in cpg_network_structure.connections:
                hinge1 = connection.cpg_index_highest.index
                pos1 = body.grid_position(active_hinges[hinge1])
                cpg_idx1 = int(pos1[0] + pos1[1] *
                               self._grid_size + self._grid_size**2 / 2)
                hinge2 = connection.cpg_index_lowest.index
                pos2 = body.grid_position(active_hinges[hinge2])
                cpg_idx2 = int(pos2[0] + pos2[1] *
                               self._grid_size + self._grid_size**2 / 2)
                rel_pos = self._relative_pos(pos1[:2], pos2[:2])
                idx = max(cpg_idx1, cpg_idx2)
                brain_params.append(temporary_genotype.brain[
                    idx*14 + rel_pos
                ])
                
            if len(brain_params) == 1:
                new_genotypes.append(deepcopy(genotype))
                child_task_performance.append(-150.0)
                logging.info("brain params has only 1 value, hence 1-D, which cannot be used to learn")
                total_no_1_active_hinges_robot += 1
                continue

            optimizer = cma.CMAEvolutionStrategy(
                brain_params,
                self._initial_std,
                self._options
            )

            generation_index = 0
            while generation_index < self._generations:
                logging.info(
                    f"CMAES-Learner Gen: {generation_index + 1} / {self._generations}.")

                # Get the sampled solutions(parameters) from cma.
                candidate_solutions = optimizer.ask()

                robots = [
                    ModularRobot(
                        body=body,
                        brain=BrainCpgNetworkStatic.uniform_from_params(
                            params=params,
                            cpg_network_structure=cpg_network_structure,
                            initial_state_uniform=0.5 * math.pi / 2.0,
                            output_mapping=output_mapping,
                        ),
                    )
                    for params in candidate_solutions
                ]

                # Evalue the individual with the reward function
                # The fitness is the negative of the reward function as CMA-ES minimizes the objective function.
                fitness = self._reward_function.evaluate(robots)
                np_fitness = -np.array(fitness)

                # Update the optimizer
                optimizer.tell(candidate_solutions, np_fitness)

                # Increase the generation index counter.
                generation_index += 1

            # Get the best solution from the optimizer
            best_solution = optimizer.result.xbest
            best_fitness = optimizer.result.fbest

            logging.info(f"Best Fitness: {best_fitness}")

            new_params = genotype.brain.copy()
            for hinge, learned_weight in zip(active_hinges, best_solution[:len(active_hinges)]):
                pos = body.grid_position(hinge)
                cpg_idx = int(pos[0] + pos[1] *
                              self._grid_size + self._grid_size**2 / 2)
                new_params[
                    cpg_idx*14
                ] = learned_weight

            for connection, connection_weight in zip(cpg_network_structure.connections,
                                                     best_solution[len(active_hinges):]):
                hinge1 = connection.cpg_index_highest.index
                pos1 = body.grid_position(active_hinges[hinge1])
                cpg_idx1 = int(pos1[0] + pos1[1] *
                               self._grid_size + self._grid_size**2 / 2)
                hinge2 = connection.cpg_index_lowest.index
                pos2 = body.grid_position(active_hinges[hinge2])
                cpg_idx2 = int(pos2[0] + pos2[1] *
                               self._grid_size + self._grid_size**2 / 2)
                rel_pos = self._relative_pos(pos1[:2], pos2[:2])
                idx = max(cpg_idx1, cpg_idx2)
                new_params[
                    idx*14 + rel_pos
                ] = connection_weight

            # Create the new genotype using the same body but the learned brain
            new_genotype = Genotype(
                body=genotype.body,
                brain=new_params
            )
            new_genotypes.append(new_genotype)
            child_task_performance.append(-best_fitness)

        logging.info(f"Total Robots With No Active Hinges: {total_no_0_active_hinges_robot}")
        logging.info(f"Total Robots With 1-D Brain Params: {total_no_1_active_hinges_robot}")
        logging.info("End brain learner optimization process.")

        return new_genotypes, child_task_performance
