#
from .array_genotype_config import ArrayCrossoverConfig
from .array_genotype import ArrayGenotype


def crossover(parent_a: ArrayGenotype,
              parent_b: ArrayGenotype,
              crossover_prob: ArrayCrossoverConfig,
              first_best: bool):
    """
    The brain of the best parent is returned
    """

    if first_best:
        return ArrayGenotype(parent_a.params_array.copy())
    else:
        return ArrayGenotype(parent_b.params_array.copy())
