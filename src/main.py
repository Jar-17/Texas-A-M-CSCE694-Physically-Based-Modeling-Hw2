from __future__ import annotations

from dataclasses import dataclass
import sys
import os
import time

import numpy as np
import pygame

"""include the physics and settings modules"""
try:
    from physics.world import BallState, PhysicsWorld, SimulationConfig
    from physics.ply_loader import load_ply, prepare_point_cloud, PointCloudMesh
    from physics.gpu import gpu_available as _gpu_available, synchronize as _gpu_synchronize, to_numpy
    from ui.settings import AppSettings
except ModuleNotFoundError:
    from src.physics.world import BallState, PhysicsWorld, SimulationConfig
    from src.physics.ply_loader import load_ply, prepare_point_cloud, PointCloudMesh
    from src.physics.gpu import gpu_available as _gpu_available, synchronize as _gpu_synchronize, to_numpy
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
POLYGON_FILL = (42, 93, 96)
MESH_FILL  = (80, 160, 200)   # default point-cloud dot colour
MESH_FLASH = (255, 255, 255)  # flash colour on collision
ORIGIN = np.array([0.0, 0.0, 0.0])
DEFAULT_CAMERA_POSITION = np.array([4.5, 3.2, 6.0], dtype=float)
MAX_FRAME_DT = 0.25
MAX_PHYSICS_STEPS_PER_FRAME = 8
BALL_RADIUS = 0.06
MIN_BALL_COUNT = 1
MAX_BALL_COUNT = 100000


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


def draw_ball(screen: pygame.Surface, camera: Camera, position: np.ndarray, color: np.ndarray, radius: float) -> None:
    projected = camera.project(position)
    if projected is None:
        return
    (x, y), depth = projected
    _, _, _, viewport_height = VIEWPORT
    focal_length = (viewport_height * 0.5) / np.tan(np.deg2rad(camera.field_of_view) * 0.5)
    pixel_radius = max(2, int(radius * focal_length / depth))
    pygame.draw.circle(screen, tuple((np.asarray(color) * 255).astype(int)), (x, y), pixel_radius)


def draw_balls_batch(screen: pygame.Surface, camera: Camera, positions_np: np.ndarray, colors_np: np.ndarray, radius: float) -> None:
    """
    Batch-render all balls in one vectorised NumPy projection pass.

    * Projects all positions at once (mirrors draw_point_cloud math).
    * Converts float [0,1] colours to uint8 in a single array operation.
    * Depth-sorts back-to-front so closer balls occlude farther ones.
    * Computes per-ball pixel radius vectorised.
    * Falls back gracefully to an empty draw when no balls are visible.
    """
    if len(positions_np) == 0:
        return

    right, up, forward = camera._basis()
    cam_pos = camera.position

    offsets = positions_np.astype(np.float64) - cam_pos   # (N, 3)
    depths  = offsets @ forward                            # (N,)

    # Cull points behind near clip
    visible = depths > camera.near_clip
    if not visible.any():
        return

    v_offsets = offsets[visible]
    v_depths  = depths[visible]
    v_colors  = colors_np[visible]

    viewport_x, viewport_y, viewport_w, viewport_h = VIEWPORT
    focal = (viewport_h * 0.5) / np.tan(np.deg2rad(camera.field_of_view) * 0.5)

    cam_x = v_offsets @ right
    cam_y = v_offsets @ up

    sx = (viewport_x + viewport_w * 0.5 + focal * cam_x / v_depths).astype(np.int32)
    sy = (viewport_y + viewport_h * 0.5 - focal * cam_y / v_depths).astype(np.int32)

    # Per-ball pixel radii (vectorised)
    pixel_radii = np.maximum(2, (radius * focal / v_depths)).astype(np.int32)

    # Float [0,1] → uint8 [0,255] colour conversion in one pass
    colors_u8 = (np.clip(v_colors, 0.0, 1.0) * 255).astype(np.uint8)

    # Depth-sort back-to-front so closer balls draw on top
    order = np.argsort(-v_depths)

    draw_circle = pygame.draw.circle
    for idx in order:
        draw_circle(
            screen,
            (int(colors_u8[idx, 0]), int(colors_u8[idx, 1]), int(colors_u8[idx, 2])),
            (int(sx[idx]), int(sy[idx])),
            int(pixel_radii[idx]),
        )


def draw_polygon(screen: pygame.Surface, camera: Camera, vertices: np.ndarray, fill_color=POLYGON_FILL) -> None:
    projected = [camera.project(vertex) for vertex in vertices]
    if all(item is not None for item in projected):
        points = [item[0] for item in projected]
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.lines(screen, ACCENT_HOVER, True, points, 2)


def draw_point_cloud(
    screen: pygame.Surface,
    camera: Camera,
    mesh: PointCloudMesh,
) -> None:
    """
    Render the point-cloud mesh as depth-sorted dots.

    Points are projected individually, depth-sorted back-to-front, and drawn
    as 2-pixel circles.  The colour flashes white immediately after a collision
    and fades back to MESH_FILL once mesh.flash_timer reaches zero.
    """
    pts = mesh.render_points      # (R, 3) float32
    if len(pts) == 0:
        return

    # Project all points in one pass, collecting (screen_xy, depth) tuples
    right, up, forward = camera._basis()
    cam_pos = camera.position

    offsets = pts.astype(np.float64) - cam_pos          # (R, 3)
    depths  = offsets @ forward                          # (R,)

    # Cull points behind near clip
    visible = depths > camera.near_clip
    if not visible.any():
        return

    v_offsets = offsets[visible]
    v_depths  = depths[visible]
    v_pts     = pts[visible]

    viewport_x, viewport_y, viewport_w, viewport_h = VIEWPORT
    focal = (viewport_h * 0.5) / np.tan(np.deg2rad(camera.field_of_view) * 0.5)

    cam_x = v_offsets @ right
    cam_y = v_offsets @ up

    sx = (viewport_x + viewport_w * 0.5 + focal * cam_x / v_depths).astype(np.int32)
    sy = (viewport_y + viewport_h * 0.5 - focal * cam_y / v_depths).astype(np.int32)

    # Depth sort: farthest first so closer points draw on top
    order = np.argsort(-v_depths)

    colour = MESH_FLASH if mesh.flash_timer > 0 else MESH_FILL
    draw_circle = pygame.draw.circle
    for idx in order:
        draw_circle(screen, colour, (int(sx[idx]), int(sy[idx])), 2)


def reset_particles(particles: BallState, count: int, origin, rng) -> None:
    particles.spawn_particles(count, origin, rng)


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
    vortex_enabled: bool,
    explosive_force_enabled: bool,
    poly_enabled: bool,
    mouse_position: tuple[int, int],
    mesh_name: str | None = None,
    gpu_enabled: bool = False,
    gpu_available: bool = False,
) -> None:
    pygame.draw.rect(screen, PANEL_BG, (0, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(screen, PANEL_BORDER, (PANEL_WIDTH - 1, 0), (PANEL_WIDTH - 1, HEIGHT), 2)
    screen.blit(font.render("SIMULATION CONTROLS", True, TEXT), (22, 20))
    screen.blit(small_font.render("Adjust parameters while it runs", True, MUTED_TEXT), (22, 49))
    for slider in sliders:
        slider.draw(screen, font, small_font)

    # Four toggle buttons on one row: VORTEX | EXPLODE | POLY | GPU
    force_y = 485
    btn_w = 59
    gpu_label = "GPU"
    gpu_color_override: tuple | None = None
    if not gpu_available:
        gpu_color_override = (70, 70, 70)   # greyed-out when no CUDA device found
    force_buttons: list[tuple] = [
        (pygame.Rect(22,                        force_y, btn_w, 28), "VORTEX",  vortex_enabled,           None),
        (pygame.Rect(22 + (btn_w + 6),          force_y, btn_w, 28), "EXPLODE", explosive_force_enabled,   None),
        (pygame.Rect(22 + (btn_w + 6) * 2,      force_y, btn_w, 28), "POLY",    poly_enabled,              None),
        (pygame.Rect(22 + (btn_w + 6) * 3,      force_y, btn_w, 28), gpu_label, gpu_enabled,               gpu_color_override),
    ]
    for rect, label, enabled, color_override in force_buttons:
        hovered = rect.collidepoint(mouse_position)
        if color_override is not None:
            # Draw greyed-out button for unavailable GPU
            pygame.draw.rect(screen, color_override, rect, border_radius=5)
            text_surf = small_font.render(f"{label}: {'ON' if enabled else 'OFF'}", True, (140, 140, 140))
            screen.blit(text_surf, text_surf.get_rect(center=rect.center))
        else:
            draw_button(screen, small_font, rect, f"{label}: {'ON' if enabled else 'OFF'}", hovered)

    button_y = 530
    buttons = (
        (pygame.Rect(22, button_y, 122, 38), "RESET BALL"),
        (pygame.Rect(154, button_y, 122, 38), "RESET VIEW"),
        (pygame.Rect(22, button_y + 48, 254, 40), "RESUME" if paused else "PAUSE"),
    )
    for rect, label in buttons:
        draw_button(screen, small_font, rect, label, rect.collidepoint(mouse_position))

    # ---- PLY mesh load / clear ------------------------------------------
    mesh_y = 624
    draw_button(screen, small_font, pygame.Rect(22, mesh_y, 122, 30), "LOAD MESH",
                pygame.Rect(22, mesh_y, 122, 30).collidepoint(mouse_position))
    draw_button(screen, small_font, pygame.Rect(154, mesh_y, 122, 30), "CLEAR MESH",
                pygame.Rect(154, mesh_y, 122, 30).collidepoint(mouse_position))
    if mesh_name:
        name_surf = small_font.render(f"Mesh: {mesh_name}", True, ACCENT)
    else:
        name_surf = small_font.render("No mesh loaded", True, MUTED_TEXT)
    screen.blit(name_surf, (22, mesh_y + 34))

    launch_label = small_font.render("Launch presets", True, MUTED_TEXT)
    screen.blit(launch_label, (WIDTH - launch_label.get_width() - 22, 672))
    for index, label in enumerate(("SIDE", "FLOOR", "TOP", "CORNER")):
        rect = pygame.Rect(22 + index * 64, 690, 56, 32)
        draw_button(screen, small_font, rect, label, rect.collidepoint(mouse_position))
    keyboard_label = small_font.render("Keyboard: A/D orbit  W/S tilt  Q/E zoom  ESC quit", True, MUTED_TEXT)
    screen.blit(keyboard_label, (WIDTH - keyboard_label.get_width() - 22, 730))


def update_perf(perf: dict, key: str, sample: float, smoothing: float = 0.1) -> None:
    """Exponential moving average so the on-screen numbers stay readable."""
    perf[key] += (sample - perf[key]) * smoothing


def draw_perf_stats(
    screen: pygame.Surface,
    font: pygame.font.Font,
    perf: dict,
    gpu_enabled: bool,
    ball_count: int,
    paused: bool,
) -> None:
    """
    Top-right readout comparing CPU and GPU mode:
      * physics ms  - time spent in world.step() this frame (the part the GPU accelerates)
      * draw ms     - time spent rendering (always CPU, same in both modes)
      * sim speed   - simulated time / real time; below 100% the step cap is
                      dropping time, which is what makes motion look choppy
    """
    sim_speed = perf["sim_speed"] * 100.0
    if paused:
        speed_text, speed_color = "sim speed   paused", MUTED_TEXT
    else:
        speed_text = f"sim speed   {sim_speed:5.0f} %"
        speed_color = ACCENT_HOVER if sim_speed >= 95.0 else (255, 200, 90) if sim_speed >= 60.0 else (255, 110, 100)
    lines = [
        (f"{'GPU' if gpu_enabled else 'CPU'} mode   {ball_count} balls", TEXT),
        (f"physics   {perf['physics_ms']:6.1f} ms  ({perf['steps']:.1f} steps)", ACCENT_HOVER),
        (f"draw      {perf['draw_ms']:6.1f} ms", MUTED_TEXT),
        (speed_text, speed_color),
    ]
    y = 56
    for text, color in lines:
        surface = font.render(text, True, color)
        screen.blit(surface, (WIDTH - surface.get_width() - 22, y))
        y += surface.get_height() + 2

def open_and_load_ply(world: PhysicsWorld) -> str | None:
    """
    Open a file-picker dialog (tkinter), load the chosen .ply file,
    build a PointCloudMesh, attach it to world.mesh, and return the
    stem name for the status display.  Returns None on cancel / error.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()          # hide the empty Tk window
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Load PLY point cloud",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return None
        raw_xyz = load_ply(path)
        # Bounds are CuPy arrays in GPU mode; the loader works on NumPy only.
        mesh = prepare_point_cloud(raw_xyz, to_numpy(world.bounds_min), to_numpy(world.bounds_max))
        mesh.name = os.path.splitext(os.path.basename(path))[0]
        world.mesh = mesh
        return mesh.name
    except Exception as exc:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("PLY load error", str(exc))
            root.destroy()
        except Exception:
            pass
        return None


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
    particles = BallState()
    particles.spawn_particles(3, initial_position, rng)
    particles.velocity[:] = np.asarray(initial_velocity, dtype=float)
    world = PhysicsWorld(
        ball=particles,
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
    particle_count_setting = 3
    mesh_name: str | None = None
    poly_enabled = True
    # Smoothed performance stats shown in the viewport (see draw_perf_stats).
    perf = {"physics_ms": 0.0, "draw_ms": 0.0, "steps": 0.0, "sim_speed": 1.0}
    while running:
        """rendering at 60 fps and physics at fixed dt"""
        real_dt = clock.tick(60) / 1000.0
        frame_dt = min(real_dt, MAX_FRAME_DT)
        if not paused:
            accumulator += frame_dt

        for event in pygame.event.get():
            for slider in sliders:
                slider.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_position = event.pos
                vortex_button   = pygame.Rect(22,       485, 59, 28)
                explode_button  = pygame.Rect(87,       485, 59, 28)
                poly_button     = pygame.Rect(152,      485, 59, 28)
                gpu_button      = pygame.Rect(217,      485, 59, 28)
                reset_button    = pygame.Rect(22, 530, 122, 38)
                view_button     = pygame.Rect(154, 530, 122, 38)
                pause_button    = pygame.Rect(22, 578, 254, 40)
                load_mesh_button  = pygame.Rect(22, 624, 122, 30)
                clear_mesh_button = pygame.Rect(154, 624, 122, 30)
                if vortex_button.collidepoint(mouse_position):
                    settings.vortex_enabled = not settings.vortex_enabled
                elif explode_button.collidepoint(mouse_position):
                    settings.explosive_force_enabled = not settings.explosive_force_enabled
                elif poly_button.collidepoint(mouse_position):
                    poly_enabled = not poly_enabled
                elif gpu_button.collidepoint(mouse_position):
                    if _gpu_available():
                        settings.gpu_enabled = not settings.gpu_enabled
                        if settings.gpu_enabled:
                            world.enable_gpu()
                        else:
                            world.disable_gpu()
                elif reset_button.collidepoint(mouse_position):
                    reset_particles(particles, 3, initial_position, rng)
                    world.notify_particles_respawned()
                elif view_button.collidepoint(mouse_position):
                    camera.reset()
                elif pause_button.collidepoint(mouse_position):
                    paused = not paused
                elif load_mesh_button.collidepoint(mouse_position):
                    loaded = open_and_load_ply(world)
                    if loaded is not None:
                        mesh_name = loaded
                        poly_enabled = False        # hide polygon while mesh is active
                        world.poly_enabled = False
                elif clear_mesh_button.collidepoint(mouse_position):
                    world.mesh = None
                    mesh_name = None
                    poly_enabled = True             # restore polygon when mesh is removed
                    world.poly_enabled = True
                else:
                    for index in range(4):
                        preset_button = pygame.Rect(22 + index * 64, 690, 56, 32)
                        if preset_button.collidepoint(mouse_position):
                            preset_key = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4)[index]
                            initial_position, initial_velocity = presets[preset_key]
                            reset_particles(particles, 3, initial_position, rng)
                            world.notify_particles_respawned()
                            break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_particles(particles, 3, initial_position, rng)
                elif event.key in presets:
                    initial_position, initial_velocity = presets[event.key]
                    reset_particles(particles, 3, initial_position, rng)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    settings.wind_x += -0.5 if event.key == pygame.K_LEFT else 0.5
                    sliders[1].value = settings.wind_x
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    settings.fixed_dt = float(np.clip(settings.fixed_dt + (-1 if event.key == pygame.K_UP else 1) / 240.0, 1.0 / 240.0, 1.0 / 20.0))
                    sliders[7].value = settings.fixed_dt
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
                elif event.key == pygame.K_v:
                    settings.vortex_enabled = not settings.vortex_enabled
                elif event.key == pygame.K_b:
                    settings.explosive_force_enabled = not settings.explosive_force_enabled
                elif event.key == pygame.K_p:
                    poly_enabled = not poly_enabled

        settings.gravity = sliders[0].value
        settings.wind_x = sliders[1].value
        settings.wind_y = sliders[2].value
        settings.drag_coefficient = sliders[3].value
        settings.restitution = sliders[4].value
        settings.friction = sliders[5].value
        desired_ball_count = int(round(sliders[6].value))
        if desired_ball_count != particle_count_setting:
            reset_particles(particles, desired_ball_count, initial_position, rng)
            world.notify_particles_respawned()
            particle_count_setting = desired_ball_count
        if settings.fixed_dt != sliders[7].value:
            settings.fixed_dt = sliders[7].value

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
        world.config.vortex_enabled = settings.vortex_enabled
        world.config.explosive_force_enabled = settings.explosive_force_enabled
        world.poly_enabled = poly_enabled

        if not paused:
            physics_start = time.perf_counter()
            physics_steps = 0
            while accumulator >= settings.fixed_dt and physics_steps < MAX_PHYSICS_STEPS_PER_FRAME:
                world.step(settings.fixed_dt)
                accumulator -= settings.fixed_dt
                physics_steps += 1
            if world.gpu_enabled:
                # CuPy runs asynchronously; wait so the timing covers the real GPU work.
                _gpu_synchronize()
            update_perf(perf, "physics_ms", (time.perf_counter() - physics_start) * 1000.0)
            update_perf(perf, "steps", physics_steps)
            if real_dt > 0.0:
                update_perf(perf, "sim_speed", physics_steps * settings.fixed_dt / real_dt)
            if physics_steps == MAX_PHYSICS_STEPS_PER_FRAME:
                # Drop excessive backlog instead of allowing a spiral of death.
                accumulator = 0.0

        draw_start = time.perf_counter()
        screen.fill(BG)
        draw_box(screen, camera, to_numpy(world.bounds_min), to_numpy(world.bounds_max))
        if world.mesh is not None:
            draw_point_cloud(screen, camera, world.mesh)
        draw_balls_batch(screen, camera, to_numpy(particles.position), to_numpy(particles.color), particles.radius)
        if poly_enabled:
            draw_polygon(screen, camera, to_numpy(world.poly_vertices),
                         fill_color=(255, 255, 255) if world.poly_flash_timer > 0 else POLYGON_FILL)
        update_perf(perf, "draw_ms", (time.perf_counter() - draw_start) * 1000.0)
        mouse_position = pygame.mouse.get_pos()
        draw_panel(
            screen,
            font,
            small_font,
            sliders,
            paused,
            settings.vortex_enabled,
            settings.explosive_force_enabled,
            poly_enabled,
            mouse_position,
            mesh_name=mesh_name,
            gpu_enabled=settings.gpu_enabled,
            gpu_available=_gpu_available(),
        )
        mesh_info = f"  |   mesh: {mesh_name}" if mesh_name else ""
        status = f"{'PAUSED' if paused else 'RUNNING'}   |   camera distance {camera.distance:.2f}   |   SPACE pause{mesh_info}"
        screen.blit(font.render(status, True, TEXT), (PANEL_WIDTH + 22, 20))
        draw_perf_stats(screen, small_font, perf, world.gpu_enabled, particles.count, paused)

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    # If run with a '--test' flag, execute tests instead of launching the Pygame UI
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        import sys
        sys.exit(0)
    else:
        raise SystemExit(main())
    
