"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot_simulation import ModularRobotSimulationState


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )

def point_navigation(robot, states, targets) -> float:
    """
    Calculate the point navigation fitness for a single modular robot.

    :param robot: The robot.
    :param states: The states of the robot.
    :param targets: The target points.
    :returns: The calculated fitness.
    """
    trajectory = [(0.0, 0.0)] + targets
    distances = [_compute_distance(trajectory[i], trajectory[i-1])
                 for i in range(1, len(trajectory))]
    target_range = 0.1
    reached_target_counter = 0

    coordinates = []
    for state in states:
        coordinates.append(
            state.get_modular_robot_simulation_state(
                robot).get_pose().position[:2]
        )
    lengths = []
    for i in range(1, len(coordinates)):
        lengths.append(_compute_distance(coordinates[i-1], coordinates[i]))

    starting_idx = 0
    for idx, state in enumerate(coordinates):
        if (reached_target_counter < len(targets) and
                _check_target(state, targets[reached_target_counter],
                              target_range)):
            reached_target_counter += 1
            starting_idx = idx

    fitness = 0
    if reached_target_counter > 0:
        path_len = sum(lengths[:starting_idx])
        fitness = sum(distances[:reached_target_counter]) - 0.1*path_len
    if reached_target_counter == len(targets):
        return fitness
    else:
        if reached_target_counter == 0:
            last_target = (0.0, 0.0)
        else:
            last_target = trajectory[reached_target_counter]
        last_coord = coordinates[-1]
        distance = _compute_distance(
            targets[reached_target_counter], last_target)
        distance -= _compute_distance(
            targets[reached_target_counter], last_coord)
        new_path_len = sum(lengths[:]) - sum(lengths[:starting_idx])
        return fitness + (distance - 0.1*new_path_len)


def _compute_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )


def _check_target(coord, target, target_range):
    return abs(coord[0] - target[0]) < target_range and abs(coord[1] - target[1]) < target_range