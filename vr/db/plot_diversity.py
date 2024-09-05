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
    tree_str1 = body_to_tree(body1)
    tree_str2 = body_to_tree(body2)

    tree1 = apted.helpers.Tree.from_text(tree_str1)
    tree2 = apted.helpers.Tree.from_text(tree_str2)

    apted_instance = APTED(tree1, tree2)
    return apted_instance.compute_edit_distance()

def compute_morphological_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates mean and maximum morphological diversity for each experiment and generation.
    """
    results = []

    grouped = df.groupby(['experiment_id', 'generation_index'])

    for (experiment_id, generation_index), group in grouped:
        bodies = group['body'].tolist()

        if len(bodies) > 1:
            distances = [
                compute_tree_edit_distance(body1, body2)
                for body1, body2 in combinations(bodies, 2)
            ]
            mean_diversity = sum(distances) / len(distances)
            max_diversity = max(distances)
        else:
            mean_diversity = 0
            max_diversity = 0

        results.append({
            'experiment_id': experiment_id,
            'generation_index': generation_index,
            'mean_diversity': mean_diversity,
            'max_diversity': max_diversity
        })

    return pd.DataFrame(results)

def process_database(database_file: str) -> pd.DataFrame:
    """
    Opens the database, fetches data, computes morphological diversity, and returns the aggregated results.
    """
    # Open the database connection
    dbengine = open_database_sqlite(
        database_file, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    # Fetch and prepare data
    df = fetch_data(dbengine)

    # Calculate morphological diversity
    diversity_df = compute_morphological_diversity(df)

    # Aggregate diversity metrics across all experiments for each generation
    agg_per_generation = (
        diversity_df.groupby("generation_index")
        .agg({
            "mean_diversity": ["mean", "std"]
        })
        .reset_index()
    )

    # Rename columns for clarity
    agg_per_generation.columns = [
        "generation_index",
        "mean_diversity_mean",
        "mean_diversity_std",
    ]

    return agg_per_generation

def plot_diversity_over_generations():
    """
    Plots mean morphological diversity for both databases (iea and ea), with standard deviation shading.
    """
    # Initialize logging
    setup_logging()

    # Process both databases
    iea_data = process_database('iea_database.sqlite')
    ea_data = process_database('ea12_database.sqlite')

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot IEA database diversity
    plt.plot(
        iea_data["generation_index"],
        iea_data["mean_diversity_mean"],
        label="IEA Mean Diversity",
        color="b",
    )
    plt.fill_between(
        iea_data["generation_index"],
        iea_data["mean_diversity_mean"] - iea_data["mean_diversity_std"],
        iea_data["mean_diversity_mean"] + iea_data["mean_diversity_std"],
        color="b",
        alpha=0.2,
    )

    # Plot EA database diversity
    plt.plot(
        ea_data["generation_index"],
        ea_data["mean_diversity_mean"],
        label="EA Mean Diversity",
        color="r",
    )
    plt.fill_between(
        ea_data["generation_index"],
        ea_data["mean_diversity_mean"] - ea_data["mean_diversity_std"],
        ea_data["mean_diversity_mean"] + ea_data["mean_diversity_std"],
        color="r",
        alpha=0.2,
    )

    # Configure plot aesthetics
    plt.xlabel("Generation Index")
    plt.ylabel("Mean Morphological Diversity")
    plt.title("Mean Morphological Diversity Across Generations (IEA vs EA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("figs/iea_vs_ea_diversity_plot.png")

if __name__ == "__main__":
    plot_diversity_over_generations()