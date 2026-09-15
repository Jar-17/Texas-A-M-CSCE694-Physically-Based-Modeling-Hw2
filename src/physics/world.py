from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# change of epsilon
CONTACT_TIME_EPSILON = 1.0e-9
CONTACT_POSITION_EPSILON = 1.0e-2
CONTACT_VELOCITY_EPSILON = 1.0e-9
REST_VELOCITY_EPSILON = 1.0e-4


@dataclass
class BallState:
    position: np.ndarray
    velocity: np.ndarray
    mass: float = 1.0
    radius: float = 0.06


@dataclass
class SimulationConfig:
    gravity: np.ndarray
    wind: np.ndarray
    drag_coefficient: float
    restitution: float
    friction: float
    time_step: float


class PhysicsWorld:
    def __init__(self, ball: BallState, config: SimulationConfig, bounds_min, bounds_max):
        self.ball = ball
        self.config = config
        self.bounds_min = np.array(bounds_min, dtype=float)
        self.bounds_max = np.array(bounds_max, dtype=float)

    def compute_net_force(self) -> np.ndarray:
        gravity_force = self.ball.mass * self.config.gravity
        wind_force = np.array(self.config.wind, dtype=float)
        drag_force = -self.config.drag_coefficient * self.ball.velocity
        return gravity_force + wind_force + drag_force

    def step(self, dt: float) -> None:
        """Advance one Euler step, splitting it at the first box contact."""
        acceleration = self.compute_net_force() / self.ball.mass
        remaining = max(0.0, float(dt))

        while remaining > CONTACT_TIME_EPSILON:
            """check if the ball is resting on a surface and if so, zero out velocity and acceleration"""
            resting_contact = self._resting_contact(acceleration)
            if resting_contact is not None:
                axis, normal = resting_contact
                limit = self.bounds_min[axis] + self.ball.radius if normal > 0 else self.bounds_max[axis] - self.ball.radius
                self.ball.position[axis] = limit
                self.ball.velocity[:] = 0.0
                break

            hit_time, hit_axis, hit_normal = self._first_contact(remaining)
            """if contact, advance to surface"""
            advance = remaining if hit_time is None else hit_time
            self.ball.position = self.ball.position + self.ball.velocity * advance
            self.ball.velocity = self.ball.velocity + acceleration * advance
            remaining -= advance

            if hit_axis is None:
                break

            limit = self.bounds_min[hit_axis] + self.ball.radius if hit_normal > 0 else self.bounds_max[hit_axis] - self.ball.radius
            self.ball.position[hit_axis] = limit
            self._resolve_contact(hit_axis, hit_normal)

    def _resting_contact(self, acceleration: np.ndarray):
        lower = self.bounds_min + self.ball.radius
        upper = self.bounds_max - self.ball.radius

        for axis in range(3):
            for limit, normal in ((lower[axis], 1), (upper[axis], -1)):
                if abs(self.ball.position[axis] - limit) > CONTACT_POSITION_EPSILON:
                    continue

                normal_vector = np.zeros(3, dtype=float)
                normal_vector[axis] = normal
                normal_speed = float(np.dot(self.ball.velocity, normal_vector))
                tangent_velocity = self.ball.velocity - normal_speed * normal_vector
                normal_acceleration = float(np.dot(acceleration, normal_vector))
                tangent_acceleration = acceleration - normal_acceleration * normal_vector
                friction_limit = self.config.friction * max(0.0, -normal_acceleration)

                if (
                    abs(normal_speed) <= REST_VELOCITY_EPSILON
                    and np.linalg.norm(tangent_velocity) <= REST_VELOCITY_EPSILON
                    and normal_acceleration < 0.0
                    and np.linalg.norm(tangent_acceleration) <= friction_limit + REST_VELOCITY_EPSILON
                ):
                    return axis, normal

        return None

    def _first_contact(self, max_time: float):
        lower = self.bounds_min + self.ball.radius
        upper = self.bounds_max - self.ball.radius
        best_time = None
        best_axis = None
        best_normal = 0

        for axis in range(3):
            velocity = self.ball.velocity[axis]
            # already in contact with lower surface
            if velocity < -CONTACT_VELOCITY_EPSILON and self.ball.position[axis] <= lower[axis] + CONTACT_POSITION_EPSILON:
                time = 0.0
                normal = 1
            # already in contact with upper surface
            elif velocity > CONTACT_VELOCITY_EPSILON and self.ball.position[axis] >= upper[axis] - CONTACT_POSITION_EPSILON:
                time = 0.0
                normal = -1
            # moving to the lower surface now    
            elif velocity < -CONTACT_VELOCITY_EPSILON and self.ball.position[axis] > lower[axis]:
                time = (lower[axis] - self.ball.position[axis]) / velocity
                normal = 1
            # moving to the upper surface now
            elif velocity > CONTACT_VELOCITY_EPSILON and self.ball.position[axis] < upper[axis]:
                time = (upper[axis] - self.ball.position[axis]) / velocity
                normal = -1
            # not moving towards any surface in current axis
            else:
                continue
            # go with the earliest contact
            if -CONTACT_TIME_EPSILON <= time <= max_time + CONTACT_TIME_EPSILON and (best_time is None or time < best_time):
                best_time = max(0.0, time)
                best_axis = axis
                best_normal = normal

        return best_time, best_axis, best_normal

    def _resolve_contact(self, axis: int, normal: int) -> None:
        normal_vector = np.zeros(3, dtype=float)
        normal_vector[axis] = normal
        normal_speed = float(np.dot(self.ball.velocity, normal_vector))

        if abs(normal_speed) <= REST_VELOCITY_EPSILON:
            self.ball.velocity -= normal_speed * normal_vector
            return

        if normal_speed >= 0.0:
            return
        self.ball.velocity -= (1.0 + self.config.restitution) * normal_speed * normal_vector
        tangent = self.ball.velocity - np.dot(self.ball.velocity, normal_vector) * normal_vector
        self.ball.velocity -= self.config.friction * tangent
