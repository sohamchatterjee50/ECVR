from __future__ import annotations


from dataclasses import dataclass
from random import Random
from itertools import repeat
from typing_extensions import Self

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy.typing as npt
import numpy as np
import random


@dataclass
class ArrayGenotype:
    """An array genotype for the brain"""

    brain: npt.NDArray[np.float64]

    @classmethod
    def random_brain(
        cls,
        grid_size: int,
        rng: Random,
    ) -> ArrayGenotype:
        """
        Create a random genotype.

        :param genotype_len: Length of the genotype.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        nprng = np.random.Generator(
            np.random.PCG64(rng.randint(0, 2 ** 63))
        )
        num_potential_joints = ((grid_size**2)-1)
        genotype_len = num_potential_joints*14
        params_array = nprng.standard_normal(genotype_len)
        return ArrayGenotype(params_array)

    def mutate_brain(self, mu, sigma, mutation_prob):
        """This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input genotype (brain, array of weights). This mutation expects a
        :term:`sequence` individual composed of vectors.
        The *mutation_prob* argument is the probability of each vector to be mutated.
        :param genotype: Individual to be mutated.
        :param mu: Mean or :term:`python:sequence` of means for the
                gaussian addition mutation.
        :param sigma: Standard deviation or :term:`python:sequence` of
                    standard deviations for the gaussian addition mutation.
        :param mutation_prob: Independent probability for each attribute to be mutated.
        :returns: new genotype.
        This function uses the :func:`~random.random` and :func:`~random.gauss`
        functions from the python base :mod:`random` module.
        https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py
        """
        mutated_brain = self.brain.copy()
        size = len(mutated_brain)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError(
                "mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError(
                "sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < mutation_prob:
                mutated_brain[i] += random.gauss(m, s)

        return ArrayGenotype(brain=mutated_brain)  # new_genotype

    @classmethod
    def crossover_brain(cls,
                        parent_a: Self,
                        parent_b: Self,
                        first_best: bool) -> ArrayGenotype:
        """
        The brain of the best parent is returned
        """

        if first_best:
            return ArrayGenotype(parent_a.brain.copy())
        else:
            return ArrayGenotype(parent_b.brain.copy())

    def develop_brain(self):
        return self.brain
