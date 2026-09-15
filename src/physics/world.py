from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Time tolerance is relative to the current timestep, with a small absolute floor.
TIME_EPSILON_RELATIVE = 1.0e-4
TIME_EPSILON_MIN = 1.0e-12
MAX_PHYSICS_STEP = 1.0 / 240.0

# Position and velocity tolerances use their own units and remain independent of dt.
CONTACT_POSITION_EPSILON = 1.0e-3
CONTACT_VELOCITY_EPSILON = 1.0e-9

# small enough to be considered at rest
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
    def __init__(self, ball: BallState, config: SimulationConfig, bounds_min, bounds_max, balls=None):
        self.balls = list(balls) if balls is not None else [ball]
        if not self.balls:
            raise ValueError("PhysicsWorld requires at least one ball")
        self.ball = self.balls[0]
        self.config = config
        self.bounds_min = np.array(bounds_min, dtype=float)
        self.bounds_max = np.array(bounds_max, dtype=float)

    def compute_net_force(self, ball: BallState | None = None) -> np.ndarray:
        ball = self.ball if ball is None else ball
        gravity_force = ball.mass * self.config.gravity
        wind_force = np.array(self.config.wind, dtype=float)
        drag_force = -self.config.drag_coefficient * ball.velocity
        return gravity_force + wind_force + drag_force

    def step(self, dt: float) -> None:
        """Advance the simulation using bounded substeps to prevent large restitution effects."""
        remaining = max(0.0, float(dt))

        while remaining > 0.0:
            substep = min(remaining, MAX_PHYSICS_STEP)
            self._step_once(substep)
            self._resolve_ball_collisions()
            remaining -= substep

    def _step_once(self, dt: float) -> None:
        """Advance one substep, splitting it at the first box contact."""
        for ball in self.balls:
            self._step_ball(ball, dt)

    def _step_ball(self, ball: BallState, dt: float) -> None:
        acceleration = self.compute_net_force(ball) / ball.mass
        remaining = max(0.0, float(dt))
        time_epsilon = max(TIME_EPSILON_MIN, remaining * TIME_EPSILON_RELATIVE)

        while remaining > time_epsilon:
            resting_contact = self._resting_contact(ball, acceleration)
            if resting_contact is not None:
                axis, normal = resting_contact
                limit = self.bounds_min[axis] + ball.radius if normal > 0 else self.bounds_max[axis] - ball.radius
                ball.position[axis] = limit
                ball.velocity[:] = 0.0
                break

            hit_time, hit_axis, hit_normal = self._first_contact(ball, remaining, time_epsilon)
            advance = remaining if hit_time is None else hit_time
            ball.position = ball.position + ball.velocity * advance
            ball.velocity = ball.velocity + acceleration * advance
            remaining -= advance

            if hit_axis is None:
                break

            limit = self.bounds_min[hit_axis] + ball.radius if hit_normal > 0 else self.bounds_max[hit_axis] - ball.radius
            ball.position[hit_axis] = limit
            self._resolve_contact(ball, hit_axis, hit_normal)

    def _resting_contact(self, ball: BallState, acceleration: np.ndarray):
        lower = self.bounds_min + ball.radius
        upper = self.bounds_max - ball.radius

        for axis in range(3):
            for limit, normal in ((lower[axis], 1), (upper[axis], -1)):
                if abs(ball.position[axis] - limit) > CONTACT_POSITION_EPSILON:
                    continue

                normal_vector = np.zeros(3, dtype=float)
                normal_vector[axis] = normal
                normal_speed = float(np.dot(ball.velocity, normal_vector))
                tangent_velocity = ball.velocity - normal_speed * normal_vector
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

    def _first_contact(self, ball: BallState, max_time: float, time_epsilon: float):
        lower = self.bounds_min + ball.radius
        upper = self.bounds_max - ball.radius
        best_time = None
        best_axis = None
        best_normal = 0

        """ check in 3 axes if the ball is moving towards a surface and if so, when will it hit the surface"""
        for axis in range(3):
            velocity = ball.velocity[axis]
            # very close to lower surface and moving towards it
            if velocity < -CONTACT_VELOCITY_EPSILON and ball.position[axis] <= lower[axis] + CONTACT_POSITION_EPSILON:
                time = 0.0
                normal = 1
            # very close to upper surface and moving towards it
            elif velocity > CONTACT_VELOCITY_EPSILON and ball.position[axis] >= upper[axis] - CONTACT_POSITION_EPSILON:
                time = 0.0
                normal = -1
            # moving to the lower surface now    
            elif velocity < -CONTACT_VELOCITY_EPSILON and ball.position[axis] > lower[axis]:
                time = (lower[axis] - ball.position[axis]) / velocity
                normal = 1
            # moving to the upper surface now
            elif velocity > CONTACT_VELOCITY_EPSILON and ball.position[axis] < upper[axis]:
                time = (upper[axis] - ball.position[axis]) / velocity
                normal = -1
            # not moving towards any surface in current axis
            else:
                continue
            # make sure contact happend in best time(either x,y,z)
            if -time_epsilon <= time <= max_time + time_epsilon and (best_time is None or time < best_time):
                best_time = max(0.0, time)
                best_axis = axis
                best_normal = normal

        return best_time, best_axis, best_normal

    def _resolve_contact(self, ball: BallState, axis: int, normal: int) -> None:
        normal_vector = np.zeros(3, dtype=float)
        normal_vector[axis] = normal
        normal_speed = float(np.dot(ball.velocity, normal_vector))

        if abs(normal_speed) <= REST_VELOCITY_EPSILON:
            ball.velocity -= normal_speed * normal_vector
            return

        if normal_speed >= 0.0:
            return
        ball.velocity -= (1.0 + self.config.restitution) * normal_speed * normal_vector
        tangent = ball.velocity - np.dot(ball.velocity, normal_vector) * normal_vector
        ball.velocity -= self.config.friction * tangent

    def _resolve_ball_collisions(self) -> None:
        for first_index, first in enumerate(self.balls):
            for second in self.balls[first_index + 1:]:
                offset = second.position - first.position
                distance = float(np.linalg.norm(offset))
                minimum_distance = first.radius + second.radius
                if distance >= minimum_distance:
                    continue

                if distance <= CONTACT_POSITION_EPSILON:
                    normal = np.array([1.0, 0.0, 0.0])
                    distance = 0.0
                else:
                    normal = offset / distance

                penetration = minimum_distance - distance
                inverse_mass_sum = 1.0 / first.mass + 1.0 / second.mass
                first.position -= normal * penetration * (1.0 / first.mass) / inverse_mass_sum
                second.position += normal * penetration * (1.0 / second.mass) / inverse_mass_sum

                relative_velocity = second.velocity - first.velocity
                normal_speed = float(np.dot(relative_velocity, normal))
                if normal_speed >= 0.0:
                    continue

                impulse_magnitude = -(1.0 + self.config.restitution) * normal_speed / inverse_mass_sum
                impulse = impulse_magnitude * normal
                first.velocity -= impulse / first.mass
                second.velocity += impulse / second.mass
