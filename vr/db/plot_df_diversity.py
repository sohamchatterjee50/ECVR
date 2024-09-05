"""Plot fitness over generations for all experiments, averaged."""

import json
import matplotlib.pyplot as plt
import pandas as pd
import multineat
import apted

from database_components import Experiment, Generation, Individual, Population, Genotype
from sqlalchemy import select
from sqlalchemy.orm import Session
from apted import APTED
from itertools import combinations


from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot.body.v1 import ActiveHingeV1, CoreV1, BrickV1

def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

config = load_config()

def body_to_tree(body) -> str:
    tree = body_to_tree_recur(body.core)

    return tree


def body_to_tree_recur(node) -> str:
    tree = None
    if isinstance(node, CoreV1):
        tree = '{c'
    if isinstance(node, BrickV1):
        tree = '{b'
    if isinstance(node, ActiveHingeV1):
        tree = '{a'

    if node is None:
        tree = '{e}'
        return tree
    else:
        for key, child in node.children.items():
            tree = tree + body_to_tree_recur(child)
        tree = tree + '}'
        return tree


def fetch_data(dbengine) -> pd.DataFrame:
    """
    Retrieves experiment data from the database and prepares a DataFrame with robot bodies.
    """
    with Session(dbengine) as ses:
        result = ses.execute(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
        ).all()

    data = []
    for row in result:
        experiment_id = row.experiment_id
        generation_index = row.generation_index
        genotype = row.Genotype
        body = genotype.develop_body()  # Develop the robot's body from its genotype
        data.append({
            'experiment_id': experiment_id,
            'generation_index': generation_index,
            'body': body
        })

    return pd.DataFrame(data)


def compute_tree_edit_distance(body1, body2) -> int:
    """
    Computes the tree-edit distance between two robot bodies.
    """
    # Convert both bodies to their tree string representations
    tree_str1 = body_to_tree(body1)
    tree_str2 = body_to_tree(body2)

    # Parse the tree strings into Tree objects for APTED
    tree1 = apted.helpers.Tree.from_text(tree_str1)
    tree2 = apted.helpers.Tree.from_text(tree_str2)

    # Compute the tree-edit distance using APTED
    apted_instance = APTED(tree1, tree2)
    return apted_instance.compute_edit_distance()


def compute_morphological_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates mean and maximum morphological diversity for each experiment and generation.
    """
    results = []

    # Group the data by experiment and generation
    grouped = df.groupby(['experiment_id', 'generation_index'])

    for (experiment_id, generation_index), group in grouped:
        bodies = group['body'].tolist()

        if len(bodies) > 1:
            # Calculate pairwise tree-edit distances
            distances = [
                compute_tree_edit_distance(body1, body2)
                for body1, body2 in combinations(bodies, 2)
            ]
            mean_diversity = sum(distances) / len(distances)
            max_diversity = max(distances)
        else:
            # If only one robot is present, diversity is zero
            mean_diversity = 0
            max_diversity = 0

        results.append({
            'experiment_id': experiment_id,
            'generation_index': generation_index,
            'mean_diversity': mean_diversity,
            'max_diversity': max_diversity
        })

    return pd.DataFrame(results)


def plot_diversity_over_generations():
    """
    Aggregates morphological diversity data and plots mean and max diversity across generations.
    """
    # Initialize logging
    setup_logging()

    # Open the database connection
    dbengine = open_database_sqlite(
        config['DATABASE_FILE'], open_method=OpenMethod.OPEN_IF_EXISTS
    )

    # Fetch and prepare data
    df = fetch_data(dbengine)

    # Calculate morphological diversity
    diversity_df = compute_morphological_diversity(df)

    # Aggregate diversity metrics across all experiments for each generation
    agg_per_generation = (
        diversity_df.groupby("generation_index")
        .agg({
            "max_diversity": ["mean", "std"],
            "mean_diversity": ["mean", "std"]
        })
        .reset_index()
    )

    # Rename columns for clarity
    agg_per_generation.columns = [
        "generation_index",
        "max_diversity_mean",
        "max_diversity_std",
        "mean_diversity_mean",
        "mean_diversity_std",
    ]

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    ## Plot mean of maximum diversity with shading for standard deviation
    #plt.plot(
    #    agg_per_generation["generation_index"],
    #    agg_per_generation["max_diversity_mean"],
    #    label="Max Diversity Mean",
    #    color="b",
    #)
    #plt.fill_between(
    #    agg_per_generation["generation_index"],
    #    agg_per_generation["max_diversity_mean"] -
    #    agg_per_generation["max_diversity_std"],
    #    agg_per_generation["max_diversity_mean"] +
    #    agg_per_generation["max_diversity_std"],
    #    color="b",
    #    alpha=0.2,
    #)

    # Plot mean of mean diversity with shading for standard deviation
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_diversity_mean"],
        label="Mean Diversity Mean",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_diversity_mean"] -
        agg_per_generation["mean_diversity_std"],
        agg_per_generation["mean_diversity_mean"] +
        agg_per_generation["mean_diversity_std"],
        color="r",
        alpha=0.2,
    )

    # Configure plot aesthetics
    plt.xlabel("Generation Index")
    plt.ylabel("Morphological Diversity")
    plt.title("Mean and Max Morphological Diversity Across Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.savefig("figs/ea12_diversity_plot.png")


if __name__ == "__main__":
    plot_diversity_over_generations()
