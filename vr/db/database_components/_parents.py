"""Parents class."""

import sqlalchemy
import sqlalchemy.orm as orm

from revolve2.experimentation.database import HasId

from ._base import Base
from ._individual import Individual
from ._generation import Generation  # Import Generation model for the foreign key relationship

class Parents(Base, HasId):
    """A table representing parent relationships between individuals and their generation."""

    __tablename__ = "parents"

    # Foreign key columns
    parent1_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("individual.id"), nullable=False, init=False
    )
    parent2_id: orm.Mapped[int | None] = orm.mapped_column(
        sqlalchemy.ForeignKey("individual.id"), nullable=True, init=False
    )
    parent_gen_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("generation.id"), nullable=False, init=False
    )

    # Relationships
    parent1: orm.Mapped[Individual] = orm.relationship("Individual", foreign_keys=[parent1_id])
    parent2: orm.Mapped[Individual] = orm.relationship("Individual", foreign_keys=[parent2_id])
    parent_gen: orm.Mapped[Generation] = orm.relationship("Generation", foreign_keys=[parent_gen_id])
