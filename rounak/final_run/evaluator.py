"""Evaluator class."""
import math

import numpy as np
import numpy.typing as npt

from pyrr import Vector3
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.ci_group.interactive_objects import Ball
from revolve2.simulation.scene import Pose


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structures: any
    _bodies: any
    _output_mappings: any
    _target_point = [(-2.0, 0.0)]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structures: any,
        bodies,
        output_mappings,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators, viewer_type="custom"
        )
        self._terrain = terrains.flat()
        self._cpg_network_structures = cpg_network_structures
        self._bodies = bodies
        self._output_mappings = output_mappings

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate a generation.

        Fitness is the point navigation task, where the robot has
        to move from the starting position to the end position.

        :param solutions: Solutions to evaluate.
        :returns: Fitnesses of the solutions.
        """
        # Create robots from the brain parameters.
        idx = 0
        robots = [
            ModularRobot(
                body=body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=output_mapping,
                ),
            )
            for body, cpg_network_structure, output_mapping, params in zip(self._bodies, 
                                                                           self._cpg_network_structures,
                                                                           self._output_mappings, 
                                                                           solutions)
        ]   

        # Create the scenes.
        scenes = []
        idx = 0.0
        # for robot in robots:
        #     scene = ModularRobotScene(terrain=self._terrain)
        #     # start the robot with post
        #     scene.add_robot(robot, pose=Pose(Vector3([idx + 1, 0.0, 0.0])))
        #     scenes.append(scene)
        #     idx += 1
        
        scene = ModularRobotScene(terrain=self._terrain)
        for robot in robots:
            scene.add_robot(robot, pose=Pose(Vector3([idx + 1, 0.0, 0.0])))
            idx += 1

        scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=60,
                sampling_frequency=100
            ),
            scenes=scenes[0],
            vr=True,
        )

        #robot_info = {}
        ## print(len(robots))
        ## print(len(scene_states))
        #for robot in robots:
        #    # print(robot.uuid)
        #    # print(states)
        #    fitness = fitness_functions.point_navigation(robot, scene_states[0], self._target_point)
        #    uuid = robot.uuid
        #    robot_info[uuid] = fitness

        #
        #print(robot_info)
      
        ## return np.array(xy_displacements)
        ## Calculate point navigation fitness
        #point_navigation_fitness = []
        #for robot, states in zip(robots, scene_states):
        #    point_navigation = fitness_functions.point_navigation(
        #        robot, states, self._target_point
        #    )
        #    point_navigation_fitness.append(point_navigation)

        return None
