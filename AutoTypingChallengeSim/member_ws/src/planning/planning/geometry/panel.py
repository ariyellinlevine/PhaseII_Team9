"""A flat panel (the board or the key grid) as a frame in the world, and rays hitting it."""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

Z_AXIS = np.array([0.0, 0.0, 1.0])


class Hit(NamedTuple):
    xy: np.ndarray
    range: float
    incidence: float


@dataclass(frozen=True)
class Panel:
    """A flat panel and its frame: x right, y down, z into the panel, so the arm sits on the -z side."""

    origin: np.ndarray
    rotation: Rotation

    @property
    def normal(self):
        """Unit normal pointing back toward the arm."""
        return self.rotation.apply(-Z_AXIS)

    def to_world(self, xy):
        return self.origin + self.rotation.apply([*xy, 0.0])

    def to_local(self, point):
        return self.rotation.inv().apply(np.asarray(point) - self.origin)

    def shifted(self, xy):
        """The same panel with its frame moved to `xy` in its own plane, e.g. from the board to the key grid."""
        return Panel(self.to_world(xy), self.rotation)

    def hit(self, start, direction):
        """Where a ray from `start` along the unit vector `direction` meets the plane, or None if it never does."""
        facing = direction @ self.normal
        if facing >= 0.0:
            return None
        stroke = (self.origin - start) @ self.normal / facing
        if stroke <= 0.0:
            return None
        return Hit(self.to_local(start + stroke * direction)[:2], stroke, np.arccos(np.clip(-facing, -1.0, 1.0)))
