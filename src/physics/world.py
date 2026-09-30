from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from physics.ply_loader import PointCloudMesh, GRID_CELL_SIZE
    from physics.gpu import gpu_available, get_array_module, to_device, to_numpy
except ModuleNotFoundError:
    from src.physics.ply_loader import PointCloudMesh, GRID_CELL_SIZE
    from src.physics.gpu import get_array_module, to_device, to_numpy

# Time tolerance is relative to the current timestep, with a small absolute floor.
TIME_EPSILON_RELATIVE = 1.0e-4
TIME_EPSILON_MIN = 1.0e-12
MAX_PHYSICS_STEP = 1.0 / 240.0

# Position and velocity tolerances use their own units and remain independent of dt.
CONTACT_POSITION_EPSILON = 1.0e-3
CONTACT_VELOCITY_EPSILON = 1.0e-9

# small enough to be considered at rest
REST_VELOCITY_EPSILON = 1.0e-4


class BallState:
    def __init__(self, position=None, velocity=None, mass=1.0, radius=0.06):
        self.position = np.asarray(position, dtype=float).reshape((-1, 3)) if position is not None else np.empty((0, 3), dtype=float)
        self.velocity = np.asarray(velocity, dtype=float).reshape((-1, 3)) if velocity is not None else np.empty((0, 3), dtype=float)
        if self.position.shape != self.velocity.shape:
            raise ValueError("position and velocity must have matching (N, 3) shapes")
        self.mass = float(mass)
        self.radius = float(radius)
        self.lifespan = np.empty(self.count, dtype=float)
        self.initial_lifespan = np.empty(self.count, dtype=float)
        self.color = np.empty((self.count, 3), dtype=float)

    @property
    def count(self):
        return self.position.shape[0]

    def spawn_particles(self, count, origin=(0.0, 0.0, 0.0), rng=None):
        """Particle generation function. Using uniform distribution for position and lifespan, and normal distribution for velocity."""
        rng = np.random.default_rng() if rng is None else rng
        count = max(0, int(count))
        origin = np.asarray(origin, dtype=float)
        self.position = origin + rng.uniform(-0.12, 0.12, size=(count, 3))
        self.velocity = rng.normal(0.0, 1.0, size=(count, 3))
        self.velocity[:, 1] = np.abs(self.velocity[:, 1]) * 2.5 + 2.0
        self.velocity[:, (0, 2)] *= 1.5
        self.initial_lifespan = rng.uniform(1.5, 10.0, size=count)
        self.lifespan = self.initial_lifespan.copy()
        self.color = np.column_stack((
            rng.uniform(0.85, 1.0, size=count),
            rng.uniform(0.35, 0.85, size=count),
            rng.uniform(0.05, 0.35, size=count),
        ))

    def filter_active(self, active):
        """check life span and filter out dead particles"""
        self.position = self.position[active]
        self.velocity = self.velocity[active]
        self.lifespan = self.lifespan[active]
        self.initial_lifespan = self.initial_lifespan[active]
        self.color = self.color[active]


@dataclass
class SimulationConfig:
    gravity: np.ndarray
    wind: np.ndarray
    drag_coefficient: float
    restitution: float
    friction: float
    time_step: float
    """for tornado and explosion effect"""
    vortex_strength: float = 1.5
    vortex_enabled: bool = False
    explosive_force_enabled: bool = False
    explosive_force_strength: float = 4.0


class PhysicsWorld:
    def __init__(self, ball: BallState, config: SimulationConfig, bounds_min, bounds_max, balls=None):
        self.ball = ball
        self.balls = balls
        self.config = config
        self.bounds_min = np.array(bounds_min, dtype=float)
        self.bounds_max = np.array(bounds_max, dtype=float)

        """polygon plane: 4 vertices, normal, and plane offset"""
        self._poly_vertices_np = np.array([
            [-1.0, 0.0, 0.25], [0.15, 0.55, 0.9], [1.0, 0.0, -0.25], [-0.15, -0.55, -0.9]
        ], dtype=float)
        edge_a = self._poly_vertices_np[1] - self._poly_vertices_np[0]
        edge_b = self._poly_vertices_np[2] - self._poly_vertices_np[0]
        self._poly_normal_np = np.cross(edge_a, edge_b)
        self._poly_normal_np /= np.linalg.norm(self._poly_normal_np)
        self._poly_plane_offset = float(np.dot(self._poly_normal_np, self._poly_vertices_np[0]))

        """control polygon rendering and collision from the UI; flash timer is set to 0.15 seconds on collision"""
        self.poly_flash_timer: float = 0.0
        """remove polygon in simulation"""
        self.poly_enabled: bool = True
        """optional point cloud mesh"""
        self.mesh: Optional[PointCloudMesh] = None

        """when GPU mode turn on, create GPU copies of polygon data; set by _sync_constants_to_device()"""
        self.gpu_enabled: bool = False
        self.xp = np                  # active array module (numpy or cupy)
        # Device copies of world constants (set by _sync_constants_to_device)
        self._bounds_min_d = self.bounds_min
        self._bounds_max_d = self.bounds_max
        self.poly_vertices = self._poly_vertices_np
        self.poly_normal   = self._poly_normal_np
        self.poly_plane_offset = self._poly_plane_offset

    """functions for GPU acceleration; enable_gpu() and disable_gpu() are called from the UI"""
    def enable_gpu(self) -> bool:
        """
        Switch physics computation to the GPU.
        Returns True on success, False if no GPU is available.
        All mutable particle arrays and world constants are transferred to device.
        """
        if not gpu_available():
            return False
        import cupy as cp
        self.xp = cp
        self.gpu_enabled = True
        self._sync_constants_to_device()
        self._sync_ball_to_device()
        return True

    def disable_gpu(self) -> None:
        """Switch back to CPU (numpy).  Particle arrays are pulled back from device."""
        if not self.gpu_enabled:
            return
        self._sync_ball_to_host()
        self.xp = np
        self.gpu_enabled = False
        # Restore CPU-side world constants — use to_numpy() because bounds are
        # currently CuPy arrays; np.asarray() cannot implicitly convert them.
        self.bounds_min   = self._bounds_min_d = to_numpy(self.bounds_min)
        self.bounds_max   = self._bounds_max_d = to_numpy(self.bounds_max)
        self.poly_vertices = self._poly_vertices_np
        self.poly_normal   = self._poly_normal_np

    def _sync_constants_to_device(self) -> None:
        """Copy immutable world constants to device memory."""
        xp = self.xp
        self._bounds_min_d = xp.asarray(self.bounds_min)
        self._bounds_max_d = xp.asarray(self.bounds_max)
        self.poly_vertices = xp.asarray(self._poly_vertices_np)
        self.poly_normal   = xp.asarray(self._poly_normal_np)
        self.bounds_min    = self._bounds_min_d
        self.bounds_max    = self._bounds_max_d

    def _sync_ball_to_device(self) -> None:
        """Transfer all BallState arrays to the active device."""
        xp = self.xp
        b = self.ball
        b.position         = xp.asarray(to_numpy(b.position))
        b.velocity         = xp.asarray(to_numpy(b.velocity))
        b.lifespan         = xp.asarray(to_numpy(b.lifespan))
        b.initial_lifespan = xp.asarray(to_numpy(b.initial_lifespan))
        b.color            = xp.asarray(to_numpy(b.color))

    def _sync_ball_to_host(self) -> None:
        """Pull all BallState arrays back to CPU numpy arrays."""
        b = self.ball
        b.position         = to_numpy(b.position)
        b.velocity         = to_numpy(b.velocity)
        b.lifespan         = to_numpy(b.lifespan)
        b.initial_lifespan = to_numpy(b.initial_lifespan)
        b.color            = to_numpy(b.color)

    def notify_particles_respawned(self) -> None:
        """
        Call this after reset_particles / spawn_particles when GPU is active
        so the freshly-created numpy arrays are transferred to the device.
        """
        if self.gpu_enabled:
            self._sync_ball_to_device()

    """functions for computing forces"""
    def compute_net_force(self):
        xp = self.xp
        offset  = self.ball.position
        gravity = self.ball.mass * xp.asarray(self.config.gravity)
        wind    = xp.asarray(self.config.wind)
        force   = gravity + wind - self.config.drag_coefficient * self.ball.velocity
        if self.config.vortex_enabled:
            vortex = xp.column_stack((-offset[:, 2], xp.zeros(self.ball.count), offset[:, 0]))
            force += self.config.vortex_strength * vortex
        if self.config.explosive_force_enabled:
            distance  = xp.linalg.norm(offset, axis=1, keepdims=True)
            safe      = distance > 1.0e-12
            direction = xp.where(safe, offset / xp.where(safe, distance, 1.0), 0.0)
            force += self.config.explosive_force_strength * direction
        return force

    def spawn_particles(self, count, origin=(0.0, 0.0, 0.0)):
        self.ball.spawn_particles(count, origin)

    """functions for the main simulation loop"""

    def step(self, dt: float) -> None:
        if self.ball.count == 0:
            return
        xp = self.xp
        dt = max(0.0, float(dt))
        old_position = self.ball.position.copy()
        self.ball.velocity += self.compute_net_force() / self.ball.mass * dt
        self.ball.position += self.ball.velocity * dt
        self._resolve_box_collisions()
        if self.poly_enabled:
            self._resolve_polygon_collisions(old_position)
        self._resolve_mesh_collisions()
        self.poly_flash_timer = max(0.0, self.poly_flash_timer - dt)
        if self.mesh is not None:
            self.mesh.flash_timer = max(0.0, self.mesh.flash_timer - dt)
        self.ball.lifespan -= dt
        self.ball.filter_active(self.ball.lifespan > 0.0)

    """functions for resolving collisions"""

    def _resolve_box_collisions(self):
        xp = self.xp
        lower = self._bounds_min_d + self.ball.radius
        upper = self._bounds_max_d - self.ball.radius
        low  = self.ball.position < lower   # (N,3) bool
        high = self.ball.position > upper   # (N,3) bool
        self.ball.position = xp.clip(self.ball.position, lower, upper)
        """check what axis the ball is colliding with and reverse the velocity on that axis, scaled by restitution"""
        self.ball.velocity = xp.where(low | high,
                                      -self.config.restitution * self.ball.velocity,
                                      self.ball.velocity)

    """functions for polygon collisions"""

    def _point_inside_polygon(self, points):
        """polygon info"""
        xp    = self.xp
        n     = self.poly_normal
        verts = self.poly_vertices                    # (nv, 3)
        nv    = len(self._poly_vertices_np)
        v0s   = verts                                 # (nv, 3)
        v1s   = xp.roll(verts, -1, axis=0)           # (nv, 3) — next vertex wrapping around
        edges = v1s - v0s                             # (nv, 3)
        """diff for each edge to each point: (nv,N,3)"""
        diff  = points[None, :, :] - v0s[:, None, :]
        """cross product of edges and diff: (nv,N,3)"""
        cross = xp.cross(edges[:, None, :], diff)    # broadcast edges over N
        """dot product of cross and polygon normal: (nv,N)"""
        dots  = xp.einsum("ijk,k->ij", cross, n)
        """if > tolerance for all edges, the point is inside the polygon"""
        return xp.all(dots >= -2.0e-3, axis=0)

    def _resolve_polygon_collisions(self, old_position):
        xp  = self.xp
        n   = self.poly_normal
        off = self.poly_plane_offset          # plain Python float is fine here

        """use the plane equation to compute the signed distance from the polygon plane for each particle"""
        position = self.ball.position
        old_distance = old_position @ n - off
        distance     = position     @ n - off

        denom             = old_distance - distance
        safe              = xp.abs(denom) > 1.0e-12
        """if the denominator is not zero, compute the fraction of the way from old to new (ex. halfway of the motion)"""
        crossing_fraction = xp.where(safe, old_distance / xp.where(safe, denom, 1.0), 0.0)
        """compute the crossing point"""
        crossing_point      = old_position + crossing_fraction[:, None] * (position - old_position)
        """and project it onto the polygon plane"""
        crossing_projection = crossing_point - (crossing_point @ n - off)[:, None] * n

        """1) the particle crossed the plane"""
        hit  = (old_distance * distance <= 0.0) & (xp.abs(distance) <= self.ball.radius)
        """2) the projected point is inside the polygon"""
        hit &= self._point_inside_polygon(crossing_projection)
        """3) the particle is moving toward the polygon"""
        hit &= (self.ball.velocity @ n) < 0.0

        """no friction is applied to the tangential velocity; only the normal component is reversed and scaled by restitution"""
        if xp.any(hit):
            position[hit] += (self.ball.radius - distance[hit])[:, None] * n
            normal_speed    = self.ball.velocity[hit] @ n
            self.ball.velocity[hit] -= (1.0 + self.config.restitution) * normal_speed[:, None] * n
            self.poly_flash_timer = 0.15

    # ── Point-cloud mesh collision (always runs on CPU) ────────────────────────

    def _resolve_mesh_collisions(self) -> None:
        """
        Sphere-vs-point(from point-cloud) collision using the spatial hash built at load time.
        This method always operates on plain NumPy arrays; when GPU mode is active
        the ball arrays are pulled to host, resolved, then pushed back to device.
        """
        mesh = self.mesh
        if mesh is None or self.ball.count == 0:
            return

        # Pull arrays to CPU for the grid-based loop
        if self.gpu_enabled:
            pos_np = to_numpy(self.ball.position)
            vel_np = to_numpy(self.ball.velocity)
        else:
            pos_np = self.ball.position
            vel_np = self.ball.velocity

        radius      = self.ball.radius
        contact_eps = 1.0e-3
        threshold   = radius + contact_eps
        inv_cell    = 1.0 / GRID_CELL_SIZE
        grid        = mesh.grid

        """check if the particle is within the bounding box of the mesh(possible collision)"""
        lo     = mesh.aabb_min - threshold
        hi     = mesh.aabb_max + threshold
        in_box = np.all((pos_np >= lo) & (pos_np <= hi), axis=1)
        candidates = np.where(in_box)[0]

        any_hit = False
        for i in candidates:
            pos = pos_np[i]
            vel = vel_np[i]

            cx = int(np.floor(pos[0] * inv_cell))
            cy = int(np.floor(pos[1] * inv_cell))
            cz = int(np.floor(pos[2] * inv_cell))

            neighbour_chunks = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        chunk = grid.get((cx + dx, cy + dy, cz + dz))
                        if chunk is not None:
                            neighbour_chunks.append(chunk)
            if not neighbour_chunks:
                continue

            neighbours = np.concatenate(neighbour_chunks, axis=0)
            diff       = pos - neighbours
            dist_sq    = (diff * diff).sum(axis=1)
            hit_mask   = dist_sq < threshold * threshold
            if not hit_mask.any():
                continue

            hit_dist_sq = dist_sq[hit_mask]
            hit_diff    = diff[hit_mask]
            closest     = np.argmin(hit_dist_sq)
            d           = float(np.sqrt(hit_dist_sq[closest]))

            normal = np.array([0.0, 1.0, 0.0]) if d < 1.0e-9 else hit_diff[closest] / d

            v_normal = float(np.dot(vel, normal))
            if v_normal >= 0.0:
                continue

            penetration    = threshold - d
            pos_np[i]     += normal * penetration
            vel_np[i]     -= (1.0 + self.config.restitution) * v_normal * normal
            v_tan          = vel_np[i] - np.dot(vel_np[i], normal) * normal
            vel_np[i]     -= self.config.friction * v_tan
            any_hit        = True

        # Push results back to device if needed
        if any_hit:
            if self.gpu_enabled:
                xp = self.xp
                self.ball.position = xp.asarray(pos_np)
                self.ball.velocity = xp.asarray(vel_np)
            mesh.flash_timer = 0.12
