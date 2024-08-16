"""Genotype class."""

from __future__ import annotations

import multineat
import numpy as np
import math

from random import Random
from array_genotypes.array_genotype_orm import ArrayGenotypeOrm
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeOrmV1
from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic


from ._base import Base


class Genotype(Base, HasId, BodyGenotypeOrmV1, ArrayGenotypeOrm):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    _grid_size = 22
    __tablename__ = "genotype"

    @classmethod
    def random(
        cls,
        grid_size: int,
        innov_db_body: multineat.InnovationDatabase,
        rng: Random,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)
        brain = cls.random_brain(grid_size, rng)

        return Genotype(body=body.body, brain=brain.brain)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
        rng: Random,
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        body = self.mutate_body(innov_db_body, rng)
        brain = self.mutate_brain(0, 0.5, 0.8)

        return Genotype(body=body.body, brain=brain.brain)

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: Random,
        first_best: bool,
    ) -> Genotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)
        brain = cls.crossover_brain(parent1, parent2, first_best)

        return Genotype(body=body.body, brain=brain.brain)

    def _relative_pos(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        mapping = {(1, 0): 1, (1, 1): 2, (0, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6,
                   (-1, 1): 7, (1, -1): 8, (2, 0): 9, (0, 2): 10, (-2, 0): 11, (0, -2): 12, (0, 0): 13}

        return mapping[(dx, dy)]

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.brain
        active_hinges = body.find_modules_of_type(ActiveHinge)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

        brain_params = []
        for hinge in active_hinges:
            pos = body.grid_position(hinge)
            cpg_idx = int(pos[0] + pos[1] *
                          self._grid_size + self._grid_size**2 / 2)
            brain_params.append(brain[
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
            brain_params.append(brain[
                idx*14 + rel_pos
            ])

        modular_robot = ModularRobot(
            body=body,
            brain=BrainCpgNetworkStatic.uniform_from_params(
                params=brain_params,
                cpg_network_structure=cpg_network_structure,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=output_mapping,
            ),
        )
        return modular_robot
