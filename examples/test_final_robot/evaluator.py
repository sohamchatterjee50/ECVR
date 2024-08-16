"""Evaluator class."""

from pyrr import Vector3

from revolve2.simulation.scene import Pose
from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _target_point = [(0.0, -3.0)]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        robots,
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param population: The robots to simulate.
        :returns: Fitnesses of the robots.
        """
        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot, Pose(Vector3([0.0, 0.0, 0.0])))
            scene.add_interactive_object(
                Ball(radius=0.1, mass=0.1, pose=Pose(
                    Vector3([self._target_point[0][0], self._target_point[0][1], 0.005])))
            )
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate point navigation fitness
        point_navigation_fitness = []
        for robot, states in zip(robots, scene_states):
            point_navigation = fitness_functions.point_navigation(
                robot, states, self._target_point
            )
            point_navigation_fitness.append(point_navigation)

        return point_navigation_fitness
