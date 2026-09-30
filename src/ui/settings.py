from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AppSettings:
    gravity: float = 9.81
    wind_x: float = 0.0
    wind_y: float = 0.0
    drag_coefficient: float = 0.05
    restitution: float = 0.8
    friction: float = 0.08
    fixed_dt: float = 1.0 / 120.0
    vortex_enabled: bool = False
    explosive_force_enabled: bool = False
    gpu_enabled: bool = False

    def gravity_vector(self) -> np.ndarray:
        return np.array([0.0, -self.gravity, 0.0], dtype=float)

    def wind_vector(self) -> np.ndarray:
        return np.array([self.wind_x, self.wind_y, 0.0], dtype=float)
