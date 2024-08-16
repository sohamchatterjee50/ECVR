from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, Type, TypeVar
from sqlalchemy import ForeignKey, Integer, Boolean
import sqlalchemy.orm as orm
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import Self

from ...database import HasId
from ..._util.init_subclass_get_generic_args import init_subclass_get_generic_args

# Define a generic type variable for the Individual class
TIndividual = TypeVar("TIndividual")


class Parents(HasId, orm.MappedAsDataclass, Generic[TIndividual]):
    """
    Generic SQLAlchemy model for the 'parents' table.

    This table stores relationships between parent individuals and their generation.
    The generic parameter `TIndividual` refers to the user-defined individual type,
    which should have an `id` field that will be used as a foreign key reference.
    """

    __tablename__ = "parents"

    # -------------------------------------
    # Class members interesting to the user
    # -------------------------------------
    if TYPE_CHECKING:
        parent1_id: Mapped[int] = mapped_column(nullable=False, init=False)
        parent2_id: Mapped[Optional[int]] = mapped_column(nullable=True, init=False)
        parent_gen_id: Mapped[int] = mapped_column(nullable=False, init=False)
        mutate: Mapped[bool] = mapped_column(nullable=False, default=True, init=False)
        
        # New: Relationships to the 'TIndividual' objects for the parents
        parent1: Mapped[TIndividual] = orm.relationship(foreign_keys="Parents.parent1_id", lazy="joined")
        parent2: Mapped[Optional[TIndividual]] = orm.relationship(foreign_keys="Parents.parent2_id", lazy="joined")

    # ----------------------
    # Implementation details
    # ----------------------
    else:

        @orm.declared_attr
        def parent1_id(cls) -> Mapped[int]:  # noqa
            return cls.__parent1_id_impl()

        @orm.declared_attr
        def parent2_id(cls) -> Mapped[Optional[int]]:  # noqa
            return cls.__parent2_id_impl()

        @orm.declared_attr
        def parent_gen_id(cls) -> Mapped[int]:  # noqa
            return cls.__parent_gen_id_impl()

        @orm.declared_attr
        def mutate(cls) -> Mapped[bool]:  # noqa
            return cls.__mutate_impl()

        # New: Relationships to the 'TIndividual' objects for the parents
        @orm.declared_attr
        def parent1(cls) -> Mapped[TIndividual]:  # noqa
            return orm.relationship(cls.__type_tindividual, foreign_keys=cls.parent1_id, lazy="joined")

        @orm.declared_attr
        def parent2(cls) -> Mapped[Optional[TIndividual]]:  # noqa
            return orm.relationship(cls.__type_tindividual, foreign_keys=cls.parent2_id, lazy="joined")

    # ClassVars to store related table names for foreign key relations
    __type_tindividual: ClassVar[Type[TIndividual]]  # type: ignore[misc]
    __generation_table: ClassVar[str] = "generation"

    def __init_subclass__(
        cls: Type[Self], **kwargs: dict[str, Any]
    ) -> None:
        """
        Initialize a version of this class when it is subclassed.

        Gets the actual type of `TIndividual` and stores it for later use.
        """
        generic_types = init_subclass_get_generic_args(cls, Parents)
        assert len(generic_types) == 1
        cls.__type_tindividual = generic_types[0]

        assert not isinstance(
            cls.__type_tindividual, TypeVar
        ), "TIndividual generic argument cannot be a forward reference."

        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]

    # Foreign key implementations
    @classmethod
    def __parent1_id_impl(cls) -> Mapped[int]:
        return mapped_column(
            Integer,
            ForeignKey(f"{cls.__type_tindividual.__tablename__}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __parent2_id_impl(cls) -> Mapped[Optional[int]]:
        return mapped_column(
            Integer,
            ForeignKey(f"{cls.__type_tindividual.__tablename__}.id"),
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

    @classmethod
    def __mutate_impl(cls) -> Mapped[bool]:
        return mapped_column(
            Boolean,
            nullable=False,
            default=True,
            init=False,
        )
