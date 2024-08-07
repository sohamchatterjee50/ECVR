"""Rerun the best robot between all experiments."""

import logging

import config
from evaluator import Evaluator

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    robot_1_parameters = [0.82509686,  0.2789461,  -0.99433904,  0.66171539, -0.98227151, -0.2798158,
  0.95666296, -0.55872119,0.99991511, -0.99086972, -0.90127563,  0.99180543]
    robot_2_parameters = [ 0.17569931,  0.41384742, -0.95124468, -0.02327898,  0.99999679, -0.9958191,
 -0.74471502, -0.9890799, 0.99867015]
    robot_3_parameters = [ 0.42746379, -0.03956272,  0.50200536, -0.72051552,  0.94038187,  0.30419658,
 -0.1369869,  -0.82498644,  0.27521657,  0.75943557,  0.97826398, -0.19601156,
  0.73742149,  0.89581896, -0.13096161]
    robot_4_parameters = [ 0.50261211, -0.84255927,  0.93524634,  0.81083183,  0.98603595, -0.0929038,
  0.99753502, -0.952183,    0.73930466, -0.76130959,  0.99315007, -0.0218019,
 -0.99718827]
    robot_5_parameters = [ 0.67947524,  0.06687912,  0.90861018,  0.88084893,  0.17075663,  0.81530468,
  0.70847257,  0.05183455,  0.97684404, -0.51178614,  0.99161513,  0.97825265,
  0.69067145,  0.86171327]


    # Prepare the body and brain structure
    cpg_list = []
    output_mappings = []
    for body in config.BODIES:
        active_hinges = body.find_modules_of_type(ActiveHinge)
        (
            cpg_network_structure,
            output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        cpg_list.append(cpg_network_structure)
        output_mappings.append(output_mapping)


    # Create the evaluator.
    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        cpg_network_structures=cpg_list,
        bodies=config.BODIES,
        output_mappings=output_mappings,
    )

    # Show the robot.
    fitness = evaluator.evaluate([robot_1_parameters, robot_2_parameters, 
                                  robot_3_parameters, robot_4_parameters, 
                                  robot_5_parameters])
    logging.info(f"XY Displacement covered: {fitness}")


if __name__ == "__main__":
    main()
