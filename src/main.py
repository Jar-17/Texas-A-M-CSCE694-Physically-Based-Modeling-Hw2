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

WIDTH = 1200
HEIGHT = 760
PANEL_WIDTH = 300
VIEWPORT = (PANEL_WIDTH, 0, WIDTH - PANEL_WIDTH, HEIGHT)
BG = (13, 18, 24)
BALL_COLOR = (255, 195, 80)
BOX_COLOR = (105, 150, 180)
BOX_FILL = (24, 44, 56)
PANEL_BG = (22, 28, 36)
PANEL_BORDER = (53, 68, 82)
TEXT = (235, 241, 245)
MUTED_TEXT = (148, 164, 176)
ACCENT = (78, 186, 170)
ACCENT_HOVER = (102, 210, 191)
ORIGIN = np.array([0.0, 0.0, 0.0])
DEFAULT_CAMERA_POSITION = np.array([4.5, 3.2, 6.0], dtype=float)
MAX_FRAME_DT = 0.25
MAX_PHYSICS_STEPS_PER_FRAME = 8
BALL_RADIUS = 0.06
MIN_BALL_COUNT = 1
MAX_BALL_COUNT = 12


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

        viewport_x, viewport_y, viewport_width, viewport_height = VIEWPORT
        focal_length = (viewport_height * 0.5) / np.tan(np.deg2rad(self.field_of_view) * 0.5)
        screen_x = viewport_x + viewport_width * 0.5 + focal_length * camera_x / depth
        screen_y = viewport_y + viewport_height * 0.5 - focal_length * camera_y / depth
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
    _, _, _, viewport_height = VIEWPORT
    focal_length = (viewport_height * 0.5) / np.tan(np.deg2rad(camera.field_of_view) * 0.5)
    radius = max(2, int(ball.radius * focal_length / depth))
    pygame.draw.circle(screen, BALL_COLOR, (x, y), radius)


def reset_ball(ball: BallState, position, velocity) -> None:
    ball.position = np.array(position, dtype=float)
    ball.velocity = np.array(velocity, dtype=float)
    ball.asleep = False
    ball.sleep_time = 0.0


def random_positions(count: int, bounds_min: np.ndarray, bounds_max: np.ndarray, radius: float, rng) -> list[np.ndarray]:
    positions = []
    minimum = bounds_min + radius
    maximum = bounds_max - radius
    for _ in range(count):
        for _ in range(1000):
            candidate = rng.uniform(minimum, maximum)
            if all(np.linalg.norm(candidate - existing) >= 2.0 * radius for existing in positions):
                positions.append(candidate)
                break
        else:
            raise RuntimeError("Unable to place all balls without overlap")
    return positions


def create_balls(count: int, velocity, bounds_min: np.ndarray, bounds_max: np.ndarray, rng) -> list[BallState]:
    return [
        BallState(position=position, velocity=np.array(velocity, dtype=float))
        for position in random_positions(count, bounds_min, bounds_max, BALL_RADIUS, rng)
    ]


def reset_balls(balls: list[BallState], velocity, bounds_min: np.ndarray, bounds_max: np.ndarray, rng) -> None:
    for ball, position in zip(balls, random_positions(len(balls), bounds_min, bounds_max, BALL_RADIUS, rng)):
        reset_ball(ball, position, velocity)


@dataclass
class Slider:
    label: str
    minimum: float
    maximum: float
    value: float
    rect: pygame.Rect
    decimals: int = 2
    dragging: bool = False

    def _set_from_mouse(self, mouse_x: int) -> None:
        ratio = np.clip((mouse_x - self.rect.left) / self.rect.width, 0.0, 1.0)
        self.value = self.minimum + float(ratio) * (self.maximum - self.minimum)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.dragging = True
            self._set_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_mouse(event.pos[0])

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        screen.blit(font.render(self.label, True, TEXT), (self.rect.left, self.rect.top - 25))
        value_text = f"{self.value:.{self.decimals}f}"
        value_surface = small_font.render(value_text, True, ACCENT_HOVER)
        screen.blit(value_surface, (self.rect.right - value_surface.get_width(), self.rect.top - 24))
        pygame.draw.rect(screen, (48, 61, 72), self.rect, border_radius=3)
        ratio = (self.value - self.minimum) / (self.maximum - self.minimum)
        fill_width = max(8, int(self.rect.width * ratio))
        pygame.draw.rect(screen, ACCENT, (self.rect.left, self.rect.top, fill_width, self.rect.height), border_radius=3)
        knob_x = self.rect.left + int(self.rect.width * ratio)
        pygame.draw.circle(screen, TEXT, (knob_x, self.rect.centery), 7)


def draw_button(screen: pygame.Surface, font: pygame.font.Font, rect: pygame.Rect, label: str, hovered: bool) -> None:
    color = ACCENT_HOVER if hovered else (39, 93, 94)
    pygame.draw.rect(screen, color, rect, border_radius=5)
    text_surface = font.render(label, True, (8, 24, 28))
    screen.blit(text_surface, text_surface.get_rect(center=rect.center))


def draw_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    sliders: list[Slider],
    paused: bool,
    mouse_position: tuple[int, int],
) -> None:
    pygame.draw.rect(screen, PANEL_BG, (0, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(screen, PANEL_BORDER, (PANEL_WIDTH - 1, 0), (PANEL_WIDTH - 1, HEIGHT), 2)
    screen.blit(font.render("SIMULATION CONTROLS", True, TEXT), (22, 20))
    screen.blit(small_font.render("Adjust parameters while it runs", True, MUTED_TEXT), (22, 49))
    for slider in sliders:
        slider.draw(screen, font, small_font)

    button_y = 520
    buttons = (
        (pygame.Rect(22, button_y, 122, 38), "RESET BALL"),
        (pygame.Rect(154, button_y, 122, 38), "RESET VIEW"),
        (pygame.Rect(22, button_y + 48, 254, 40), "RESUME" if paused else "PAUSE"),
    )
    for rect, label in buttons:
        draw_button(screen, small_font, rect, label, rect.collidepoint(mouse_position))

    launch_label = small_font.render("Launch presets", True, MUTED_TEXT)
    screen.blit(launch_label, (WIDTH - launch_label.get_width() - 22, 625))
    for index, label in enumerate(("SIDE", "FLOOR", "TOP", "CORNER")):
        rect = pygame.Rect(22 + index * 64, 648, 56, 32)
        draw_button(screen, small_font, rect, label, rect.collidepoint(mouse_position))
    keyboard_label = small_font.render("Keyboard: A/D orbit  W/S tilt  Q/E zoom  ESC quit", True, MUTED_TEXT)
    screen.blit(keyboard_label, (WIDTH - keyboard_label.get_width() - 22, 710))


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 20)
    small_font = pygame.font.SysFont("segoeui", 16)

    settings = AppSettings()
    sliders = [
        Slider("Gravity", 0.0, 20.0, settings.gravity, pygame.Rect(22, 100, 254, 10)),
        Slider("Wind X", -10.0, 10.0, settings.wind_x, pygame.Rect(22, 150, 254, 10)),
        Slider("Wind Y", -10.0, 10.0, settings.wind_y, pygame.Rect(22, 200, 254, 10)),
        Slider("Air drag", 0.0, 1.0, settings.drag_coefficient, pygame.Rect(22, 250, 254, 10)),
        Slider("Restitution", 0.0, 1.0, settings.restitution, pygame.Rect(22, 300, 254, 10)),
        Slider("Friction", 0.0, 1.0, settings.friction, pygame.Rect(22, 350, 254, 10)),
        Slider("Ball count", MIN_BALL_COUNT, MAX_BALL_COUNT, 3, pygame.Rect(22, 400, 254, 10), 0),
        Slider("Physics dt", 1.0 / 240.0, 1.0 / 20.0, settings.fixed_dt, pygame.Rect(22, 450, 254, 10), 4),
    ]
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
    bounds_min = np.array([-1.5, -1.0, -1.2], dtype=float)
    bounds_max = np.array([1.5, 1.0, 1.2], dtype=float)
    rng = np.random.default_rng()
    balls = create_balls(3, initial_velocity, bounds_min, bounds_max, rng)
    ball = balls[0]
    world = PhysicsWorld(
        ball=ball,
        balls=balls,
        config=SimulationConfig(
            gravity=settings.gravity_vector(),
            wind=settings.wind_vector(),
            drag_coefficient=settings.drag_coefficient,
            restitution=settings.restitution,
            friction=settings.friction,
            time_step=settings.fixed_dt,
        ),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )

    accumulator = 0.0
    running = True
    paused = False
    while running:
        """rendering at 60 fps and physics at fixed dt"""
        frame_dt = min(clock.tick(60) / 1000.0, MAX_FRAME_DT)
        if not paused:
            accumulator += frame_dt

        for event in pygame.event.get():
            for slider in sliders:
                slider.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_position = event.pos
                reset_button = pygame.Rect(22, 520, 122, 38)
                view_button = pygame.Rect(154, 520, 122, 38)
                pause_button = pygame.Rect(22, 568, 254, 40)
                if reset_button.collidepoint(mouse_position):
                    reset_balls(balls, initial_velocity, bounds_min, bounds_max, rng)
                elif view_button.collidepoint(mouse_position):
                    camera.reset()
                elif pause_button.collidepoint(mouse_position):
                    paused = not paused
                else:
                    for index in range(4):
                        preset_button = pygame.Rect(22 + index * 64, 648, 56, 32)
                        if preset_button.collidepoint(mouse_position):
                            preset_key = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4)[index]
                            initial_position, initial_velocity = presets[preset_key]
                            reset_balls(balls, initial_velocity, bounds_min, bounds_max, rng)
                            break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_balls(balls, initial_velocity, bounds_min, bounds_max, rng)
                elif event.key in presets:
                    initial_position, initial_velocity = presets[event.key]
                    reset_balls(balls, initial_velocity, bounds_min, bounds_max, rng)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    settings.wind_x += -0.5 if event.key == pygame.K_LEFT else 0.5
                    sliders[1].value = settings.wind_x
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    settings.fixed_dt = float(np.clip(settings.fixed_dt + (-1 if event.key == pygame.K_UP else 1) / 240.0, 1.0 / 240.0, 1.0 / 20.0))
                    sliders[7].value = settings.fixed_dt
                    # Discard time accumulated using the previous timestep.
                    accumulator = 0.0
                elif event.key == pygame.K_f:
                    settings.friction = float(np.clip(settings.friction + 0.05, 0.0, 1.0))
                    sliders[5].value = settings.friction
                elif event.key == pygame.K_g:
                    settings.friction = float(np.clip(settings.friction - 0.05, 0.0, 1.0))
                    sliders[5].value = settings.friction
                elif event.key == pygame.K_h:
                    settings.drag_coefficient = float(np.clip(settings.drag_coefficient + 0.05, 0.0, 1.0))
                    sliders[3].value = settings.drag_coefficient
                elif event.key == pygame.K_j:
                    settings.drag_coefficient = float(np.clip(settings.drag_coefficient - 0.05, 0.0, 1.0))
                    sliders[3].value = settings.drag_coefficient
                elif event.key == pygame.K_c:
                    camera.reset()

        settings.gravity = sliders[0].value
        settings.wind_x = sliders[1].value
        settings.wind_y = sliders[2].value
        settings.drag_coefficient = sliders[3].value
        settings.restitution = sliders[4].value
        settings.friction = sliders[5].value
        desired_ball_count = int(round(sliders[6].value))
        if desired_ball_count != len(balls):
            balls = create_balls(desired_ball_count, initial_velocity, bounds_min, bounds_max, rng)
            world.balls = balls
            world.ball = balls[0]
        if settings.fixed_dt != sliders[7].value:
            settings.fixed_dt = sliders[7].value
            # Leftover time was measured using a different timestep.
            accumulator = 0.0

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
            physics_steps = 0
            while accumulator >= settings.fixed_dt and physics_steps < MAX_PHYSICS_STEPS_PER_FRAME:
                world.step(settings.fixed_dt)
                accumulator -= settings.fixed_dt
                physics_steps += 1
            if physics_steps == MAX_PHYSICS_STEPS_PER_FRAME:
                # Drop excessive backlog instead of allowing a spiral of death.
                accumulator = 0.0

        screen.fill(BG)
        draw_box(screen, camera, world.bounds_min, world.bounds_max)
        for ball in balls:
            draw_ball(screen, camera, ball)
        mouse_position = pygame.mouse.get_pos()
        draw_panel(screen, font, small_font, sliders, paused, mouse_position)
        status = f"{'PAUSED' if paused else 'RUNNING'}   |   camera distance {camera.distance:.2f}   |   SPACE pause"
        screen.blit(font.render(status, True, TEXT), (PANEL_WIDTH + 22, 20))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
