from revolve2.ci_group.morphological_measures import MorphologicalMeasures

from database_components import (
    Generation,
)
from sqlalchemy.orm import Session
import pandas as pd
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from matplotlib import pyplot as plt
import logging
from revolve2.experimentation.logging import setup_logging

def process_database(db_path, label, measure_name):
    """
    Process a single database and return the aggregated generation data 
    for the specified morphological measure with the specified label.
    
    Parameters:
    - db_path: Path to the SQLite database.
    - label: A label to distinguish datasets in the plot.
    - measure_name: The name of the morphological measure to analyze 
                   (e.g., 'num_active_hinges', 'num_bricks').
    
    Returns:
    - A DataFrame with generation_index, mean of the selected measure, 
      standard deviation, and the label.
    """
    dbengine = open_database_sqlite(
        db_path, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    if dbengine is None:
        raise RuntimeError(f"Failed to create database engine for {db_path}.")
    
    generation_data = []
    
    with Session(dbengine) as session:
        generations = session.query(Generation).order_by(Generation.id.asc()).all()
        print(f"Generation id: {generations.id}")
        for generation in generations:
            for individual in generation.population.individuals:
                body = individual.genotype.develop_body()
                measures = MorphologicalMeasures(body)
                generation_data.append((generation.generation_index, getattr(measures, measure_name)))
    
    # Convert the list of tuples into a DataFrame
    generation_df = pd.DataFrame(generation_data, columns=['generation_index', 'measure_count'])
    # Group by generation_index and calculate mean and standard deviation
    agg_per_generation = generation_df.groupby('generation_index').agg(
        mean_measure=('measure_count', 'mean'),
        std_measure=('measure_count', 'std')
    ).reset_index()
    
    # Add a label for distinguishing datasets in the plot
    agg_per_generation['label'] = label
    
    return agg_per_generation

def plot_measure_across_databases(db_paths, labels, measure_names):
    """
    Plot the specified morphological measure across generations for multiple databases.
    
    Parameters:
    - db_paths: List of database file paths.
    - labels: List of labels corresponding to the databases.
    - measure_name: The name of the morphological measure to plot 
                   (e.g., 'active_hinges', 'bricks').
    """
    for measure_name in measure_names:
        logging.info(f"Making plot for {measure_name}...")
        combined_data = pd.DataFrame()

        for db_path, label in zip(db_paths, labels):
            data = process_database(db_path, label, measure_name)
            combined_data = pd.concat([combined_data, data])

        # Plotting
        plt.figure(figsize=(10, 6))

        for label, group_data in combined_data.groupby('label'):
            # Plot mean measure with shading for standard deviation for each database
            plt.plot(
                group_data['generation_index'],
                group_data['mean_measure'],
                label=f'Mean {measure_name.capitalize()} ({label})'
            )
            plt.fill_between(
                group_data['generation_index'],
                group_data['mean_measure'] - group_data['std_measure'],
                group_data['mean_measure'] + group_data['std_measure'],
                alpha=0.2,
            )

        # Configure plot aesthetics
        plt.xlabel("Generation Index")
        plt.ylabel(f"Mean {measure_name.capitalize()}")
        plt.title(f"Mean {measure_name.capitalize()} Across Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_name = f"figs/combined_{measure_name}_plot.png"
        # Save the plot
        plt.savefig(plot_name)
        logging.info(f"Plot for {measure_name} saved in {plot_name}.")


def main():
    setup_logging(file_name="log.txt")
    db_paths = ["iea_database.sqlite", "ea_database.sqlite"]
    labels = ["IEA", "EA"]
    measure_names = ["num_active_hinges", "num_bricks", "num_modules", "branching", "symmetry", "mix_active_hinge_brick", "core_only"]
    process_database(db_paths, labels, measure_names)

if __name__ == "__main__":
    main()