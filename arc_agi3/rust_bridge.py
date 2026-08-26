"""Bridge to the Rust ``goldsnnail`` compositional solver.

When the Rust crate is compiled and ``cargo`` is available on PATH, this
module can invoke the ``arc_compositional_solver`` example to find a program
that explains a grid-to-grid transformation.  A pure-Python fallback is
provided for the most common operations so the world model remains usable
even without Rust.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from arc_agi3.types import FrameData, Transition

# ---------------------------------------------------------------------------
# Rust availability
# ---------------------------------------------------------------------------

RUST_AVAILABLE: bool = False
"""``True`` only if ``cargo`` is on PATH and ``GOLDSNNAIL_RUST`` is not ``0``."""

_env_override = os.environ.get("GOLDSNNAIL_RUST", "").strip()
if _env_override != "0" and shutil.which("cargo") is not None:
    RUST_AVAILABLE = True

# ---------------------------------------------------------------------------
# Pure-Python apply fallback
# ---------------------------------------------------------------------------


def _apply_identity(grid: np.ndarray) -> Optional[np.ndarray]:
    return grid.copy()


def _apply_rotate(grid: np.ndarray, angle: int) -> Optional[np.ndarray]:
    if angle == 0:  # 90° clockwise
        return np.rot90(grid, k=-1)
    if angle == 1:  # 180°
        return np.rot90(grid, k=2)
    if angle == 2:  # 270° clockwise (90° CCW)
        return np.rot90(grid, k=1)
    return None


def _apply_flip(grid: np.ndarray, axis: int) -> Optional[np.ndarray]:
    if axis == 0:  # horizontal
        return np.fliplr(grid)
    if axis == 1:  # vertical
        return np.flipud(grid)
    return None


def _apply_move(grid: np.ndarray, dx: int, dy: int) -> Optional[np.ndarray]:
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                continue
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = grid[r, c]
    return out


def _apply_fill(grid: np.ndarray, color: int, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    out = grid.copy()
    hh, ww = grid.shape
    for r in range(y, min(y + h, hh)):
        for c in range(x, min(x + w, ww)):
            out[r, c] = color
    return out


def _apply_copy(
    grid: np.ndarray, src_x: int, src_y: int, dst_x: int, dst_y: int, w: int, h: int
) -> Optional[np.ndarray]:
    out = grid.copy()
    hh, ww = grid.shape
    if src_x + w > ww or src_y + h > hh or dst_x + w > ww or dst_y + h > hh:
        return None
    for dr in range(h):
        for dc in range(w):
            out[dst_y + dr, dst_x + dc] = grid[src_y + dr, src_x + dc]
    return out


def _apply_gravity(grid: np.ndarray, direction: int) -> Optional[np.ndarray]:
    h, w = grid.shape
    out = np.zeros_like(grid)
    if direction == 0:  # down
        for c in range(w):
            wr = h - 1
            for r in range(h - 1, -1, -1):
                if grid[r, c] != 0:
                    out[wr, c] = grid[r, c]
                    wr -= 1
    elif direction == 1:  # up
        for c in range(w):
            wr = 0
            for r in range(h):
                if grid[r, c] != 0:
                    out[wr, c] = grid[r, c]
                    wr += 1
    elif direction == 2:  # left
        for r in range(h):
            wc = 0
            for c in range(w):
                if grid[r, c] != 0:
                    out[r, wc] = grid[r, c]
                    wc += 1
    elif direction == 3:  # right
        for r in range(h):
            wc = w - 1
            for c in range(w - 1, -1, -1):
                if grid[r, c] != 0:
                    out[r, wc] = grid[r, c]
                    wc -= 1
    else:
        return None
    return out


def _apply_mirror(grid: np.ndarray, axis_x: int, axis_y: int) -> Optional[np.ndarray]:
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h):
        for c in range(w):
            mr = abs(r - axis_y)
            mc = abs(c - axis_x)
            if mr < h and mc < w:
                out[r, c] = grid[mr, mc]
    return out


def _apply_tile(grid: np.ndarray, n: int, m: int) -> Optional[np.ndarray]:
    if n == 0 or m == 0 or n > 4 or m > 4:
        return None
    gh, gw = grid.shape
    if gh == 0 or gw == 0:
        return None
    nh, nw = gh * m, gw * n
    if nh > 30 or nw > 30:
        return None
    out = np.zeros((nh, nw), dtype=grid.dtype)
    for ty in range(m):
        for tx in range(n):
            out[ty * gh : (ty + 1) * gh, tx * gw : (tx + 1) * gw] = grid
    return out


def _apply_crop(grid: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    hh, ww = grid.shape
    if x + w > ww or y + h > hh or w == 0 or h == 0:
        return None
    return grid[y : y + h, x : x + w].copy()


def _apply_replace_color(grid: np.ndarray, src: int, dst: int) -> Optional[np.ndarray]:
    if src == dst:
        return None
    out = grid.copy()
    out[out == src] = dst
    return out


def _apply_scale(grid: np.ndarray, factor: int) -> Optional[np.ndarray]:
    if factor == 0 or factor > 3:
        return None
    gh, gw = grid.shape
    if gh == 0 or gw == 0:
        return None
    nh, nw = gh * factor, gw * factor
    if nh > 30 or nw > 30:
        return None
    out = np.zeros((nh, nw), dtype=grid.dtype)
    for r in range(gh):
        for c in range(gw):
            out[r * factor : (r + 1) * factor, c * factor : (c + 1) * factor] = grid[r, c]
    return out


def _apply_crop_content(grid: np.ndarray) -> Optional[np.ndarray]:
    h, w = grid.shape
    if h == 0 or w == 0:
        return grid.copy()
    bg = int(np.bincount(grid.flatten()).argmax())
    mask = grid != bg
    if not np.any(mask):
        return grid.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin : rmax + 1, cmin : cmax + 1].copy()


_OP_APPLY = {
    0: _apply_identity,
    1: _apply_rotate,
    2: _apply_flip,
    3: _apply_move,
    4: _apply_fill,
    5: _apply_copy,
    6: _apply_gravity,
    7: _apply_mirror,
    8: _apply_tile,
    9: _apply_crop,
    10: _apply_replace_color,
    11: _apply_scale,
    12: _apply_crop_content,
}


def rust_apply_program(
    grid: list[list[int]], program: list[list[int]]
) -> Optional[list[list[int]]]:
    """Best-effort pure-Python apply of an ArcProgram to a grid.

    Returns the resulting grid as ``list[list[int]]``, or ``None`` on failure.
    """
    if not program:
        return grid
    arr = np.array(grid, dtype=int)
    try:
        for token in program:
            if not isinstance(token, (list, tuple)) or len(token) < 8:
                return None
            op_code = int(token[0])
            p1, p2, p3, p4, p5, p6, p7 = (int(v) for v in token[1:8])
            fn = _OP_APPLY.get(op_code)
            if fn is None:
                return None
            if op_code == 1:  # Rotate
                arr = fn(arr, p1)
            elif op_code == 2:  # Flip
                arr = fn(arr, p1)
            elif op_code == 3:  # Move
                arr = fn(arr, p1, p2)
            elif op_code == 4:  # Fill
                arr = fn(arr, p1, p2, p3, p4)
            elif op_code == 5:  # Copy
                arr = fn(arr, p1, p2, p3, p4, p5)
            elif op_code == 6:  # Gravity
                arr = fn(arr, p1)
            elif op_code == 7:  # Mirror
                arr = fn(arr, p1, p2)
            elif op_code == 8:  # Tile
                arr = fn(arr, p1, p2)
            elif op_code == 9:  # Crop
                arr = fn(arr, p1, p2, p3, p4)
            elif op_code == 10:  # ReplaceColor
                arr = fn(arr, p1, p2)
            elif op_code == 11:  # Scale
                arr = fn(arr, p1)
            elif op_code == 12:  # CropContent
                arr = fn(arr)
            if arr is None:
                return None
        return arr.tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rust invocation
# ---------------------------------------------------------------------------


def solve_grid_transform(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    timeout: float = 20.0,
) -> Optional[list[list[int]]]:
    """Ask the Rust solver to find a program mapping ``input_grid`` to ``output_grid``.

    Returns a list of 8-byte tokens (each token is ``list[int]`` of length 8)
    or ``None`` if Rust is unavailable or the solver fails.
    """
    if not RUST_AVAILABLE:
        return None

    task_id = "py_bridge_adhoc"
    tmpdir = tempfile.mkdtemp(prefix="goldsnnail_solve_")
    try:
        task_data = {
            "train": [
                {"input": input_grid, "output": output_grid}
            ],
            "test": [
                {"input": input_grid}
            ],
        }
        task_path = Path(tmpdir) / f"{task_id}.json"
        task_path.write_text(json.dumps(task_data), encoding="utf-8")

        data_dir = Path(tmpdir) / "data" / "arc" / "training"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / f"{task_id}.json").write_text(json.dumps(task_data), encoding="utf-8")

        cmd = [
            "cargo", "run", "--release", "--example", "arc_compositional_solver",
            "--", task_id,
        ]
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None

        return _parse_program_from_stdout(result.stdout, result.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_program_from_stdout(stdout: str, stderr: str) -> Optional[list[list[int]]]:
    """Parse the program from the Rust solver's stdout."""
    combined = stdout + "\n" + stderr
    # Look for "Program: ..." or JSON output
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith("Program:"):
            desc = line[len("Program:"):].strip()
            # desc looks like "Rotate([0, 0, 0, 0, 0, 0, 0]) -> ..."
            tokens = _parse_program_description(desc)
            if tokens:
                return tokens
        # Try JSON array of tokens
        if line.startswith("[") and line.endswith("]"):
            try:
                raw = json.loads(line)
                tokens = []
                for t in raw:
                    if isinstance(t, list) and len(t) == 8:
                        tokens.append([int(v) for v in t])
                if tokens:
                    return tokens
            except json.JSONDecodeError:
                pass
    return None


def _parse_program_description(desc: str) -> Optional[list[list[int]]]:
    """Parse something like 'Rotate([0,0,0,0,0,0,0]) -> Fill([1,0,0,0,0,0,0])'."""
    import re
    tokens = []
    for m in re.finditer(r"(\w+)\(\[([0-9,\s]+)\]\)", desc):
        name = m.group(1)
        params_str = m.group(2)
        params = [int(v.strip()) for v in params_str.split(",") if v.strip()]
        op_map = {
            "Identity": 0,
            "Rotate": 1,
            "Flip": 2,
            "Move": 3,
            "Fill": 4,
            "Copy": 5,
            "Gravity": 6,
            "Mirror": 7,
            "Tile": 8,
            "Crop": 9,
            "ReplaceColor": 10,
            "Scale": 11,
            "CropContent": 12,
        }
        op_code = op_map.get(name)
        if op_code is None:
            continue
        token = [op_code] + params + [0] * (7 - len(params))
        token = token[:8]
        tokens.append(token)
    return tokens if tokens else None
