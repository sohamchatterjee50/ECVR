import logging
import json
from typing import Any, List

from database_components import (
    Generation,
    Parents,
)

from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging

def load_config():
    with open('Assets/revolve2/vr/db/config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()

def main():
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config["DATABASE_FILE"], open_method=OpenMethod.OPEN_IF_EXISTS
    )

    if dbengine is None:
        raise RuntimeError("Failed to create database engine.")

    with Session(dbengine) as session:
        latest_generation = session.query(Generation).order_by(Generation.id.desc()).first()
        if not latest_generation:
            raise Exception("The generation table is empty")
        population = latest_generation.population
        parents = Parents(
            parent1=population.individuals[0],
            parent2=population.individuals[1],
            parent_gen_id=latest_generation.id,
        )
        session.add(parents)
        session.commit()
        parents = Parents(
            parent1=population.individuals[2],
            parent2=population.individuals[1],
            parent_gen_id=latest_generation.id,
        )
        session.add(parents)
        session.commit()
if __name__ == "__main__":
    main()
