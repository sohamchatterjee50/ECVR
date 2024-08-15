from typing import TYPE_CHECKING, Any, ClassVar, Type
from sqlalchemy import ForeignKey, Integer
import sqlalchemy.orm as orm
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import Self

from ...database import HasId


class Parents(HasId, orm.MappedAsDataclass):
    """
    SQLAlchemy model for the 'parents' table.

    This table stores relationships between parent individuals and their generation.
    """

    __tablename__ = "parents"

    # -------------------------------------
    # Class members interesting to the user
    # -------------------------------------
    if TYPE_CHECKING:
        parent1_id: Mapped[int] = mapped_column(nullable=False, init=False)
        parent2_id: Mapped[int | None] = mapped_column(nullable=True, init=False)
        parent_gen_id: Mapped[int] = mapped_column(nullable=False, init=False)

    # ----------------------
    # Implementation details
    # ----------------------
    else:

        @orm.declared_attr
        def parent1_id(cls) -> Mapped[int]:  # noqa
            return cls.__parent1_id_impl()

        @orm.declared_attr
        def parent2_id(cls) -> Mapped[int | None]:  # noqa
            return cls.__parent2_id_impl()

        @orm.declared_attr
        def parent_gen_id(cls) -> Mapped[int]:  # noqa
            return cls.__parent_gen_id_impl()

    # ClassVar to store related table names for foreign key relations
    __individual_table: ClassVar[str] = "individual"
    __generation_table: ClassVar[str] = "generation"

    def __init_subclass__(
        cls: Type[Self], **kwargs: dict[str, Any]
    ) -> None:
        """
        Initialize a version of this class when it is subclassed.

        Sets up table name variables for later use.
        """
        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]

    # Foreign key implementations
    @classmethod
    def __parent1_id_impl(cls) -> Mapped[int]:
        return mapped_column(
            Integer,
            ForeignKey(f"{cls.__individual_table}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __parent2_id_impl(cls) -> Mapped[int | None]:
        return mapped_column(
            Integer,
            ForeignKey(f"{cls.__individual_table}.id"),
            nullable=True,
            init=False,
        )

    @classmethod
    def __parent_gen_id_impl(cls) -> Mapped[int]:
        return mapped_column(
            Integer,
            ForeignKey(f"{cls.__generation_table}.id"),
            nullable=False,
            init=False,
        )
