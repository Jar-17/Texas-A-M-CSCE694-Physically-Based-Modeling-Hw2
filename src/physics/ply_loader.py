"""
ply_loader.py
=============
Pure NumPy binary / ASCII PLY parser and point-cloud mesh container.
No extra dependencies beyond numpy (already required by the project).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level PLY parser
# ---------------------------------------------------------------------------

# Map PLY scalar type names -> (struct format char, byte size)
_PLY_SCALAR = {
    "char":   ("b", 1), "uchar":  ("B", 1),
    "int8":   ("b", 1), "uint8":  ("B", 1),
    "short":  ("h", 2), "ushort": ("H", 2),
    "int16":  ("h", 2), "uint16": ("H", 2),
    "int":    ("i", 4), "uint":   ("I", 4),
    "int32":  ("i", 4), "uint32": ("I", 4),
    "float":  ("f", 4), "float32":("f", 4),
    "double": ("d", 8), "float64":("d", 8),
}


def _parse_header(f) -> tuple[dict, int, str]:
    """
    Read PLY header lines and return:
      elements  – dict keyed by element name, value = (count, [(prop_name, fmt_char, byte_size)])
      header_end_offset – byte offset immediately after 'end_header\n'
      endian_char – '<' little-endian, '>' big-endian, or '' for ASCII
    """
    magic = f.readline().strip()
    if magic != b"ply":
        raise ValueError("Not a PLY file (missing 'ply' magic bytes)")

    fmt_line = f.readline().decode("ascii").strip()
    if "binary_little_endian" in fmt_line:
        endian_char = "<"
    elif "binary_big_endian" in fmt_line:
        endian_char = ">"
    elif "ascii" in fmt_line:
        endian_char = ""
    else:
        raise ValueError(f"Unsupported PLY format: {fmt_line}")

    elements: dict[str, tuple[int, list]] = {}
    current_element: Optional[str] = None
    current_count: int = 0

    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header ended without 'end_header'")
        line = raw.decode("ascii", errors="replace").strip()
        if line == "end_header":
            if current_element is not None:
                elements[current_element] = (current_count, elements[current_element][1])
            break
        if line.startswith("element "):
            if current_element is not None:
                elements[current_element] = (current_count, elements[current_element][1])
            parts = line.split()
            current_element = parts[1]
            current_count = int(parts[2])
            elements[current_element] = (current_count, [])
        elif line.startswith("property ") and current_element is not None:
            parts = line.split()
            if parts[1] == "list":
                # list properties – mark with None so we can skip them later
                elements[current_element][1].append((parts[4], None, 0))
            else:
                type_name = parts[1]
                prop_name = parts[2]
                if type_name in _PLY_SCALAR:
                    fmt_c, sz = _PLY_SCALAR[type_name]
                    elements[current_element][1].append((prop_name, fmt_c, sz))
                else:
                    elements[current_element][1].append((prop_name, None, 0))
        # ignore all other header lines (comments, obj_info, etc.)

    header_end_offset = f.tell()
    return elements, header_end_offset, endian_char


def load_ply(path: str | Path) -> np.ndarray:
    """
    Parse a PLY file and return the vertex XYZ positions as a float32 (N, 3) array.

    Handles:
    * binary_little_endian and binary_big_endian
    * ASCII
    * Any extra per-vertex properties (normals, colours, Gaussian-splat attributes …)
    * No face/edge data is required – the file may be a pure point cloud.

    Raises ValueError on malformed input.
    """
    path = Path(path)
    with path.open("rb") as f:
        elements, _end_offset, endian_char = _parse_header(f)

    if "vertex" not in elements:
        raise ValueError("PLY file has no 'vertex' element")

    n_verts, props = elements["vertex"]
    if n_verts == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Locate x, y, z property indices and build a flat row layout
    prop_names = [p[0] for p in props]
    for required in ("x", "y", "z"):
        if required not in prop_names:
            raise ValueError(f"PLY vertex element has no '{required}' property")

    xi = prop_names.index("x")
    yi = prop_names.index("y")
    zi = prop_names.index("z")

    if endian_char == "":
        return _load_ascii_vertices(path, n_verts, xi, yi, zi, props)
    else:
        return _load_binary_vertices(path, n_verts, xi, yi, zi, props, endian_char)


def _load_binary_vertices(
    path: Path,
    n_verts: int,
    xi: int, yi: int, zi: int,
    props: list,
    endian_char: str,
) -> np.ndarray:
    """Read binary vertex data; extract x,y,z."""
    # Build a numpy structured dtype for the vertex row so we can use np.frombuffer
    # Skip list properties and unknown properties (byte-size == 0).
    dtype_fields = []
    row_bytes = 0
    for name, fmt_c, sz in props:
        if fmt_c is None:
            # Variable-length or unknown – we cannot easily vectorise; fall back
            raise ValueError(
                f"PLY property '{name}' is a list or unknown type; "
                "cannot read binary data with mixed-width properties"
            )
        dtype_fields.append((name, f"{endian_char}{fmt_c}"))
        row_bytes += sz

    dt = np.dtype(dtype_fields)
    assert dt.itemsize == row_bytes, "Dtype size mismatch — header parse error"

    # Re-open and seek to the data block
    with path.open("rb") as f:
        _parse_header(f)  # advances f past 'end_header'
        raw = f.read(n_verts * row_bytes)

    if len(raw) < n_verts * row_bytes:
        raise ValueError(
            f"PLY binary data truncated: expected {n_verts * row_bytes} bytes, "
            f"got {len(raw)}"
        )

    data = np.frombuffer(raw, dtype=dt, count=n_verts)
    xyz = np.column_stack(
        (data["x"].astype(np.float32),
         data["y"].astype(np.float32),
         data["z"].astype(np.float32))
    )
    return xyz.astype(np.float32)


def _load_ascii_vertices(
    path: Path,
    n_verts: int,
    xi: int, yi: int, zi: int,
    props: list,
) -> np.ndarray:
    """Read ASCII vertex data; extract x,y,z."""
    xyz = np.empty((n_verts, 3), dtype=np.float32)
    with path.open("r", encoding="ascii", errors="replace") as f:
        # Skip header
        for line in f:
            if line.strip() == "end_header":
                break
        for i in range(n_verts):
            parts = f.readline().split()
            xyz[i, 0] = float(parts[xi])
            xyz[i, 1] = float(parts[yi])
            xyz[i, 2] = float(parts[zi])
    return xyz


# ---------------------------------------------------------------------------
# Point-cloud mesh container
# ---------------------------------------------------------------------------

MAX_RENDER_POINTS = 20_000   # max points drawn per frame
MAX_COLLISION_POINTS = 3_000 # max points used for collision queries
GRID_CELL_SIZE = 0.12        # world units per spatial-hash cell


def _build_spatial_grid(pts: np.ndarray, cell_size: float) -> dict[tuple, np.ndarray]:
    """
    Build a simple uniform spatial-hash grid.
    Returns a dict mapping integer 3-tuple cell coords -> (K,3) array of points in that cell.
    """
    inv = 1.0 / cell_size
    cells_idx = np.floor(pts * inv).astype(np.int32)  # (N, 3)
    grid: dict[tuple, list] = {}
    for i in range(len(pts)):
        key = (int(cells_idx[i, 0]), int(cells_idx[i, 1]), int(cells_idx[i, 2]))
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    # Convert lists to arrays for fast indexing
    return {k: pts[v] for k, v in grid.items()}


@dataclass
class PointCloudMesh:
    """
    Container for a loaded PLY point cloud.

    render_points   : (R, 3) float32 – a subsample for rendering (≤ MAX_RENDER_POINTS)
    collision_points: (C, 3) float32 – a subsample for collision queries (≤ MAX_COLLISION_POINTS)
    grid            : spatial hash mapping int-tuple -> (K, 3) pts, built on collision_points
    aabb_min/max    : axis-aligned bounding box of collision_points
    name            : filename stem, shown in the UI
    flash_timer     : set > 0 when a collision fires this frame; decremented each step
    """
    render_points: np.ndarray        # (R, 3) float32
    collision_points: np.ndarray     # (C, 3) float32
    grid: dict = field(repr=False)
    aabb_min: np.ndarray             # (3,)
    aabb_max: np.ndarray             # (3,)
    name: str = ""
    flash_timer: float = 0.0


def prepare_point_cloud(
    raw_xyz: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
) -> PointCloudMesh:
    """
    Center and uniformly scale a raw XYZ point cloud so it fits inside 70 % of
    the simulation box, then build the render and collision subsamples.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(raw_xyz) == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        return PointCloudMesh(empty, empty, {}, zero, zero)

    # ---- center and scale ------------------------------------------------
    raw_min = raw_xyz.min(axis=0)
    raw_max = raw_xyz.max(axis=0)
    center = (raw_min + raw_max) * 0.5
    raw_extent = (raw_max - raw_min).max()          # largest dimension
    box_extent = ((box_max - box_min) * 0.70).min() # 70 % of smallest box dim
    scale = box_extent / raw_extent if raw_extent > 1e-9 else 1.0
    pts = ((raw_xyz - center) * scale).astype(np.float32)

    # ---- render subsample ------------------------------------------------
    if len(pts) > MAX_RENDER_POINTS:
        idx = rng.choice(len(pts), MAX_RENDER_POINTS, replace=False)
        render_pts = pts[idx]
    else:
        render_pts = pts.copy()

    # ---- collision subsample --------------------------------------------
    if len(pts) > MAX_COLLISION_POINTS:
        idx = rng.choice(len(pts), MAX_COLLISION_POINTS, replace=False)
        coll_pts = pts[idx]
    else:
        coll_pts = pts.copy()

    # ---- spatial grid ---------------------------------------------------
    grid = _build_spatial_grid(coll_pts, GRID_CELL_SIZE)

    aabb_min = coll_pts.min(axis=0)
    aabb_max = coll_pts.max(axis=0)

    return PointCloudMesh(
        render_points=render_pts,
        collision_points=coll_pts,
        grid=grid,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )
