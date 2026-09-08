from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

"""include the physics and settings modules"""
try:
    from physics.world import BallState, PhysicsWorld, SimulationConfig
    from ui.settings import AppSettings
except ModuleNotFoundError:
    from src.physics.world import BallState, PhysicsWorld, SimulationConfig
    from src.ui.settings import AppSettings

WIDTH = 900
HEIGHT = 700
BG = (13, 18, 24)
BALL_COLOR = (255, 195, 80)
BOX_COLOR = (105, 150, 180)
BOX_FILL = (24, 44, 56)
ORIGIN = np.array([0.0, 0.0, 0.0])
DEFAULT_CAMERA_POSITION = np.array([4.5, 3.2, 6.0], dtype=float)


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    field_of_view: float = 55.0
    near_clip: float = 0.05

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.position - self.target))

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.target - self.position
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def project(self, pos: np.ndarray) -> tuple[tuple[int, int], float] | None:
        """Project a world-space point and return its pixel position and depth."""
        right, up, forward = self._basis()
        offset = np.asarray(pos, dtype=float) - self.position
        camera_x = float(np.dot(offset, right))
        camera_y = float(np.dot(offset, up))
        depth = float(np.dot(offset, forward))
        if depth <= self.near_clip:
            return None

        focal_length = (HEIGHT * 0.5) / np.tan(np.deg2rad(self.field_of_view) * 0.5)
        screen_x = WIDTH * 0.5 + focal_length * camera_x / depth
        screen_y = HEIGHT * 0.5 - focal_length * camera_y / depth
        return (int(screen_x), int(screen_y)), depth

    def orbit(self, yaw: float, pitch: float) -> None:
        offset = self.position - self.target
        radius = np.linalg.norm(offset)
        current_yaw = np.arctan2(offset[2], offset[0])
        current_pitch = np.arcsin(offset[1] / radius)
        current_yaw += yaw
        current_pitch = float(np.clip(current_pitch + pitch, -1.45, 1.45))
        self.position = self.target + radius * np.array([
            np.cos(current_pitch) * np.cos(current_yaw),
            np.sin(current_pitch),
            np.cos(current_pitch) * np.sin(current_yaw),
        ])

    def zoom(self, amount: float) -> None:
        direction = self.position - self.target
        distance = self.distance
        self.position = self.target + direction / distance * max(1.0, distance + amount)

    def reset(self) -> None:
        self.position = DEFAULT_CAMERA_POSITION.copy()
        self.target = ORIGIN.copy()


def draw_box(screen: pygame.Surface, camera: Camera, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    corners = [
        np.array([x, y, z], dtype=float)
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
        for x in (bounds_min[0], bounds_max[0])
    ]
    projected = [camera.project(corner) for corner in corners]
    points = [item[0] if item is not None else None for item in projected]
    # Corner index is (y, z, x): draw each of the twelve box edges once.
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7))
    if all(points[i] is not None for i in (0, 1, 3, 2)):
        pygame.draw.polygon(screen, BOX_FILL, [points[i] for i in (0, 1, 3, 2)])
    for first, second in edges:
        if points[first] is not None and points[second] is not None:
            pygame.draw.line(screen, BOX_COLOR, points[first], points[second], 2)


def draw_ball(screen: pygame.Surface, camera: Camera, ball: BallState) -> None:
    projected = camera.project(ball.position)
    if projected is None:
        return
    (x, y), depth = projected
    focal_length = (HEIGHT * 0.5) / np.tan(np.deg2rad(camera.field_of_view) * 0.5)
    radius = max(2, int(ball.radius * focal_length / depth))
    pygame.draw.circle(screen, BALL_COLOR, (x, y), radius)


def reset_ball(ball: BallState, position, velocity) -> None:
    ball.position = np.array(position, dtype=float)
    ball.velocity = np.array(velocity, dtype=float)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    settings = AppSettings()
    camera = Camera(
        position=np.array([4.5, 3.2, 6.0], dtype=float),
        target=ORIGIN.copy(),
    )
    presets = {
        pygame.K_1: ([-0.8, 0.55, -0.5], [2.8, 1.0, 2.0]),
        pygame.K_2: ([-0.9, -0.2, 0.0], [3.5, 0.4, 0.0]),
        pygame.K_3: ([0.0, -0.7, 0.7], [0.0, 5.0, -3.0]),
        pygame.K_4: ([0.0, 0.0, 0.0], [3.0, 4.5, 2.5]),
    }
    initial_position, initial_velocity = presets[pygame.K_1]
    ball = BallState(position=np.array(initial_position, dtype=float), velocity=np.array(initial_velocity, dtype=float))
    world = PhysicsWorld(
        ball=ball,
        config=SimulationConfig(
            gravity=settings.gravity_vector(),
            wind=settings.wind_vector(),
            drag_coefficient=settings.drag_coefficient,
            restitution=settings.restitution,
            friction=settings.friction,
            time_step=settings.fixed_dt,
        ),
        bounds_min=[-1.5, -1.0, -1.2],
        bounds_max=[1.5, 1.0, 1.2],
    )

    accumulator = 0.0
    running = True
    paused = False
    while running:
        frame_dt = clock.tick(60) / 1000.0
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_ball(ball, initial_position, initial_velocity)
                elif event.key in presets:
                    initial_position, initial_velocity = presets[event.key]
                    reset_ball(ball, initial_position, initial_velocity)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    settings.wind_x += -0.5 if event.key == pygame.K_LEFT else 0.5
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    settings.fixed_dt = float(np.clip(settings.fixed_dt + (-1 if event.key == pygame.K_UP else 1) / 240.0, 1.0 / 240.0, 1.0 / 20.0))
                elif event.key == pygame.K_f:
                    settings.friction = float(np.clip(settings.friction + 0.05, 0.0, 1.0))
                elif event.key == pygame.K_g:
                    settings.friction = float(np.clip(settings.friction - 0.05, 0.0, 1.0))
                elif event.key == pygame.K_h:
                    settings.drag_coefficient = float(np.clip(settings.drag_coefficient + 0.05, 0.0, 1.0))
                elif event.key == pygame.K_j:
                    settings.drag_coefficient = float(np.clip(settings.drag_coefficient - 0.05, 0.0, 1.0))
                elif event.key == pygame.K_c:
                    camera.reset()

        camera_keys = pygame.key.get_pressed()
        camera.orbit(
            (camera_keys[pygame.K_d] - camera_keys[pygame.K_a]) * 1.8 * frame_dt,
            (camera_keys[pygame.K_w] - camera_keys[pygame.K_s]) * 1.3 * frame_dt,
        )
        camera.zoom((camera_keys[pygame.K_e] - camera_keys[pygame.K_q]) * 3.0 * frame_dt)

        world.config.gravity = settings.gravity_vector()
        world.config.wind = settings.wind_vector()
        world.config.drag_coefficient = settings.drag_coefficient
        world.config.restitution = settings.restitution
        world.config.friction = settings.friction
        world.config.time_step = settings.fixed_dt

        if not paused:
            while accumulator >= settings.fixed_dt:
                world.step(settings.fixed_dt)
                accumulator -= settings.fixed_dt

        screen.fill(BG)
        draw_box(screen, camera, world.bounds_min, world.bounds_max)
        draw_ball(screen, camera, ball)

        info = [
            f"g {settings.gravity:.2f}   wind x {settings.wind_x:+.2f}   drag {settings.drag_coefficient:.3f}",
            f"restitution {settings.restitution:.2f}   friction {settings.friction:.2f}   dt {settings.fixed_dt:.4f}s",
            f"{'PAUSED' if paused else 'RUNNING'}   |   1-4 presets   R reset   SPACE pause   UP/DOWN dt",
            "LEFT/RIGHT wind   F/G friction   H/J drag   ESC quit",
            f"A/D orbit   W/S tilt   Q/E zoom   C reset view   dist {camera.distance:.2f}",
        ]
        for i, line in enumerate(info):
            surf = font.render(line, True, (240, 240, 240))
            screen.blit(surf, (16, 12 + i * 22))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
