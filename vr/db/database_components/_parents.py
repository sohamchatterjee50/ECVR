"""Parents class."""

from dataclasses import dataclass

from revolve2.experimentation.optimization.ea import Parents as GenericParents

from ._base import Base
from ._individual import Individual


@dataclass
class Parents(
    Base, GenericParents[Individual], kw_only=True
):
    """Parents of individuals in a population."""

    __tablename__ = "parents"
