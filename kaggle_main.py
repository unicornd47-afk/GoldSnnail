"""ARC-AGI-3 -- GoldSnnail Interactive Agent (Kaggle submission)."""

from __future__ import annotations

import os, sys, logging, json, re, random, time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Tuple, List, Dict
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# OPENROUTER_API_KEY from Kaggle secrets or environment
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
USE_LLM = bool(OPENROUTER_API_KEY)
USE_RUST = False
BUDGET_MULTIPLIER = 2.0
SEED = 0
MAX_STEPS = 300


# === Embedded: arc_agi3/types.py ===
"""Core types for the ARC-AGI-3 interactive-agent framework.

Defines the internal, toolkit-agnostic data model used across all agent modules.
All imports must work without the real ``arc_agi`` / ``arcengine`` packages installed.
"""


import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GameAction(enum.Enum):
    """Action codes available in ARC-AGI-3 environments.

    Mirrors the real toolkit's ``GameAction``.  Simple actions (1-5) are
    parameter-free; complex actions (6-7) require a ``(x, y)`` payload.
    """

    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7
    RESET = 0  # RESET is code 0 in many toolkits; treated as complex-is=False

    @classmethod
    def from_int(cls, code: int) -> GameAction:
        mapping = {member.value: member for member in cls}
        if code not in mapping:
            raise ValueError(f"Unknown action code: {code}")
        return mapping[code]

    def to_int(self) -> int:
        return self.value

    def is_simple(self) -> bool:
        return self in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                        GameAction.ACTION4, GameAction.ACTION5)

    def is_complex(self) -> bool:
        return self in (GameAction.ACTION6, GameAction.ACTION7)


ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7, GameAction.RESET,
]
SIMPLE_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5,
]
COMPLEX_ACTIONS = [
    GameAction.ACTION6, GameAction.ACTION7,
]


def action_is_simple(a: GameAction) -> bool:
    """Module-level helper mirroring ``GameAction.is_simple()``."""
    return a.is_simple()


def action_is_complex(a: GameAction) -> bool:
    """Module-level helper mirroring ``GameAction.is_complex()``."""
    return a.is_complex()


class GameState(enum.Enum):
    """Possible states of an ARC-AGI-3 environment."""

    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    PLAYING = "PLAYING"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"
    LOSE = "LOSE"
    UNKNOWN = "UNKNOWN"

    def is_terminal(self) -> bool:
        return self in (GameState.WIN, GameState.GAME_OVER, GameState.LOSE)


def is_terminal(s: GameState) -> bool:
    """Module-level helper mirroring ``GameState.is_terminal()``."""
    return s.is_terminal()


class OperationMode(enum.Enum):
    """Operational mode for the Arcade toolkit."""

    NORMAL = "normal"
    OFFLINE = "offline"
    ONLINE = "online"
    COMPETITION = "competition"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """High-level agent action wrapping a ``GameAction`` and optional payload.

    Complex actions (6, 7) carry a ``data`` dict with ``x`` and ``y`` in
    ``[0, 63]``.  ``reasoning`` is free-form metadata the agent may attach.
    """

    action: GameAction
    data: Optional[dict[str, Any]] = None
    reasoning: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.action.is_complex():
            if self.data is None or "x" not in self.data or "y" not in self.data:
                raise ValueError(
                    f"Complex action {self.action} requires data with 'x' and 'y' keys."
                )
            x, y = self.data["x"], self.data["y"]
            if not (0 <= x <= 63 and 0 <= y <= 63):
                raise ValueError(f"Action coordinates must be in [0, 63], got ({x}, {y}).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "data": self.data,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Action:
        return cls(
            action=GameAction(d["action"]),
            data=d.get("data"),
            reasoning=d.get("reasoning"),
        )


@dataclass
class FrameData:
    """Observation returned by ``env.step()``.

    Mirrors the real toolkit's ``FrameDataRaw``.  The ``frame`` field holds
    the 2-D grid as ``list[list[int]]``; ``grid()`` exposes it as a NumPy array.
    """

    game_id: str
    state: GameState = GameState.NOT_PLAYED
    levels_completed: int = 0
    win_levels: int = 0
    available_actions: list[int] = field(default_factory=list)
    full_reset: bool = False
    guid: Optional[str] = None
    action_input: Optional[dict[str, Any]] = None
    frame: Optional[list[list[int]]] = None
    step: int = 0
    score: float = 0.0

    def grid(self) -> Optional[np.ndarray]:
        if self.frame is None:
            return None
        return np.array(self.frame, dtype=int)


@dataclass
class EnvironmentInfo:
    """Static metadata about an environment (game)."""

    game_id: str
    title: str
    tags: list[str]
    n_levels: int
    baseline_actions: int
    is_simulated: bool = True


@dataclass
class ScorecardEntry:
    """Record for a single environment play-through."""

    game_id: str
    seed: int
    level_scores: list[float]
    total_score: float
    steps_used: int
    budget: int
    won: bool


@dataclass
class Transition:
    """A single environment step: ``state_before`` -> ``action`` -> ``state_after``."""

    state_before: FrameData
    action: Action
    state_after: FrameData
    step: int

    def grid_delta(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.state_before.grid(), self.state_after.grid()


@dataclass
class SceneObject:
    """A single connected component extracted from a frame grid."""

    object_id: int
    color: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    centroid: tuple[float, float]
    area: int
    pixels: list[tuple[int, int]]


@dataclass
class SceneGraph:
    """Structured perception of a single frame."""

    objects: list[SceneObject]
    width: int
    height: int
    frame: Optional[np.ndarray] = None


@dataclass
class Hypothesis:
    """A candidate rule explaining an observed grid transformation."""

    id: str
    action: Optional[GameAction]
    description: str
    program: Optional[list[list[int]]]  # list of 8-byte tokens (Rust style)
    confidence: float
    supporting_transitions: int
    contradicting_transitions: int
    created_step: int


@dataclass
class PlanStep:
    """One step in an agent's plan."""

    action: Action
    expected_state: Optional[FrameData] = None
    description: str = ""


@dataclass
class Plan:
    """A sequence of ``PlanStep`` objects representing a full agent plan."""

    steps: list[PlanStep]
    goal_state: Optional[FrameData] = None
    confidence: float = 1.0

# === Embedded: arc_agi3/sim_env.py ===
"""Simulated ARC-AGI-3 environments.

Each :class:`SimGame` obeys the same protocol as the real toolkit so that
downstream agent code is identical for simulated and real environments.

All games are deterministic given a seed (``random.Random(seed)``) and
solvable by construction.
"""


import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np



# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SimGame(ABC):
    """Abstract base for simulated ARC-AGI-3 games."""

    def __init__(self, game_id: str, seed: int = 0) -> None:
        self.game_id = game_id
        self.seed = seed
        self._rng = random.Random(seed)
        self.level: int = 0
        self.step_count: int = 0
        self.levels_completed: int = 0
        self.win_levels: int = 0
        self.state: GameState = GameState.NOT_PLAYED
        self.baseline_actions: int = 0
        self._grid: list[list[int]] = []
        self._height: int = 0
        self._width: int = 0
        self._available_actions: list[int] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> FrameData:
        """Reset to the start of level 0 (or a provided seed)."""

    @abstractmethod
    def step(self, action: Action) -> FrameData:
        """Advance one step; mutate internal state and return new FrameData."""

    @abstractmethod
    def action_space(self) -> list[int]:
        """Return list of valid action codes for the current state."""

    @abstractmethod
    def info(self) -> EnvironmentInfo:
        """Return static environment metadata."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_frame(self, state: GameState, levels_completed: int) -> FrameData:
        return FrameData(
            game_id=self.game_id,
            state=state,
            levels_completed=levels_completed,
            win_levels=self.win_levels,
            available_actions=self.action_space(),
            full_reset=False,
            guid=None,
            action_input=None,
            frame=[row[:] for row in self._grid],
            step=self.step_count,
            score=float(levels_completed),
        )

    def _advance_level(self) -> None:
        self.levels_completed += 1
        self.level += 1
        if self.levels_completed >= self.win_levels:
            self.state = GameState.WIN

    def _build_level(self) -> None:
        """Override to populate ``self._grid`` for the current level."""


# ---------------------------------------------------------------------------
# NavigateGame
# ---------------------------------------------------------------------------


class NavigateGame(SimGame):
    """Grid navigation: move the player (color 9) to the target (color 4).

    Actions: ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=no-op.
    Grid is 8x8; empty cells are 0.  Reaching the target completes the level.
    """

    def __init__(self, game_id: str = "sim_nav", seed: int = 0) -> None:
        super().__init__(game_id, seed)
        self.win_levels = 3
        self.baseline_actions = 20
        self._height = 8
        self._width = 8
        self._player_pos: tuple[int, int] = (0, 0)
        self._target_pos: tuple[int, int] = (0, 0)

    def info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id=self.game_id,
            title="Navigate Game",
            tags=["navigation", "spatial"],
            n_levels=self.win_levels,
            baseline_actions=self.baseline_actions,
            is_simulated=True,
        )

    def reset(self, seed: Optional[int] = None) -> FrameData:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.level = 0
        self.step_count = 0
        self.levels_completed = 0
        self.state = GameState.PLAYING
        self._build_level()
        return self._make_frame(self.state, self.levels_completed)

    def action_space(self) -> list[int]:
        if self.state == GameState.WIN:
            return [-1]  # RESET only
        return [a.value for a in ALL_ACTIONS]

    def step(self, action: Action) -> FrameData:
        before = self._make_frame(self.state, self.levels_completed)
        if self.state in (GameState.WIN, GameState.GAME_OVER):
            self.step_count += 1
            return self._make_frame(self.state, self.levels_completed)

        self.step_count += 1
        act = action.action

        if act == GameAction.RESET:
            self.state = GameState.PLAYING
            self._build_level()
            return self._make_frame(self.state, self.levels_completed)

        if act in (GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4):
            pr, pc = self._player_pos
            if act == GameAction.ACTION1:  # up
                nr, nc = max(0, pr - 1), pc
            elif act == GameAction.ACTION2:  # down
                nr, nc = min(self._height - 1, pr + 1), pc
            elif act == GameAction.ACTION3:  # left
                nr, nc = pr, max(0, pc - 1)
            else:  # ACTION4 right
                nr, nc = pr, min(self._width - 1, pc + 1)
            self._grid[pr][pc] = 0
            self._player_pos = (nr, nc)
            self._grid[nr][nc] = 9

        if self._player_pos == self._target_pos:
            if self.level < self.win_levels - 1:
                self._advance_level()
                self._build_level()
            else:
                self.levels_completed = self.win_levels
                self.state = GameState.WIN

        return self._make_frame(self.state, self.levels_completed)

    def _build_level(self) -> None:
        self._grid = [[0 for _ in range(self._width)] for _ in range(self._height)]
        # Deterministic positions based on level + seed
        rng = random.Random(self.seed + self.level)
        self._player_pos = (rng.randint(0, self._height - 1), rng.randint(0, self._width - 1))
        self._target_pos = (rng.randint(0, self._height - 1), rng.randint(0, self._width - 1))
        while self._target_pos == self._player_pos:
            self._target_pos = (rng.randint(0, self._height - 1), rng.randint(0, self._width - 1))
        pr, pc = self._player_pos
        tr, tc = self._target_pos
        self._grid[pr][pc] = 9
        self._grid[tr][tc] = 4


# ---------------------------------------------------------------------------
# ColorMatchGame
# ---------------------------------------------------------------------------


class ColorMatchGame(SimGame):
    """Match a target cell to the level's required color.

    The grid is 8x8.  A central cell (3,3) needs to match a target color.
    ACTION1..ACTION5 set that cell to colors 1..5 respectively.
    ACTION5 is a no-op (keeps whatever color is there).
    Each level specifies a different required color in [1, 5].
    """

    def __init__(self, game_id: str = "sim_cm", seed: int = 0) -> None:
        super().__init__(game_id, seed)
        self.win_levels = 3
        self.baseline_actions = 5
        self._height = 8
        self._width = 8
        self._target_color: int = 0

    def info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id=self.game_id,
            title="Color Match Game",
            tags=["color", "pattern"],
            n_levels=self.win_levels,
            baseline_actions=self.baseline_actions,
            is_simulated=True,
        )

    def reset(self, seed: Optional[int] = None) -> FrameData:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.level = 0
        self.step_count = 0
        self.levels_completed = 0
        self.state = GameState.PLAYING
        self._build_level()
        return self._make_frame(self.state, self.levels_completed)

    def action_space(self) -> list[int]:
        if self.state == GameState.WIN:
            return [-1]
        return [a.value for a in ALL_ACTIONS]

    def step(self, action: Action) -> FrameData:
        before = self._make_frame(self.state, self.levels_completed)
        if self.state in (GameState.WIN, GameState.GAME_OVER):
            self.step_count += 1
            return self._make_frame(self.state, self.levels_completed)

        self.step_count += 1
        act = action.action

        if act == GameAction.RESET:
            self.state = GameState.PLAYING
            self._build_level()
            return self._make_frame(self.state, self.levels_completed)

        color_map = {
            GameAction.ACTION1: 1,
            GameAction.ACTION2: 2,
            GameAction.ACTION3: 3,
            GameAction.ACTION4: 4,
            GameAction.ACTION5: 5,
        }
        if act in color_map:
            self._grid[3][3] = color_map[act]

        if self._grid[3][3] == self._target_color:
            if self.level < self.win_levels - 1:
                self._advance_level()
                self._build_level()
            else:
                self.levels_completed = self.win_levels
                self.state = GameState.WIN

        return self._make_frame(self.state, self.levels_completed)

    def _build_level(self) -> None:
        self._grid = [[0 for _ in range(self._width)] for _ in range(self._height)]
        rng = random.Random(self.seed + self.level)
        self._target_color = rng.randint(1, 5)
        # Show the target color in the border so the agent can observe it
        for c in range(self._width):
            self._grid[0][c] = self._target_color
            self._grid[self._height - 1][c] = self._target_color
        for r in range(self._height):
            self._grid[r][0] = self._target_color
            self._grid[r][self._width - 1] = self._target_color
        # Central cell starts at a random wrong color (never the target)
        wrong = (self._target_color % 5) + 1
        self._grid[3][3] = wrong


# ---------------------------------------------------------------------------
# ClickTargetGame
# ---------------------------------------------------------------------------


class ClickTargetGame(SimGame):
    """Find and click a hidden target using a complex (x, y) action.

    The grid is 8x8, mostly empty (0).  A single target cell is color 4.
    ACTION6 is complex with ``{"x": int, "y": int}``; if coordinates match
    the target, the level is completed.  ACTION1..ACTION5 are no-ops.
    """

    def __init__(self, game_id: str = "sim_click", seed: int = 0) -> None:
        super().__init__(game_id, seed)
        self.win_levels = 3
        self.baseline_actions = 10
        self._height = 8
        self._width = 8
        self._target_pos: tuple[int, int] = (0, 0)

    def info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            game_id=self.game_id,
            title="Click Target Game",
            tags=["click", "precision"],
            n_levels=self.win_levels,
            baseline_actions=self.baseline_actions,
            is_simulated=True,
        )

    def reset(self, seed: Optional[int] = None) -> FrameData:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.level = 0
        self.step_count = 0
        self.levels_completed = 0
        self.state = GameState.PLAYING
        self._build_level()
        return self._make_frame(self.state, self.levels_completed)

    def action_space(self) -> list[int]:
        if self.state == GameState.WIN:
            return [-1]
        return [a.value for a in ALL_ACTIONS]

    def step(self, action: Action) -> FrameData:
        before = self._make_frame(self.state, self.levels_completed)
        if self.state in (GameState.WIN, GameState.GAME_OVER):
            self.step_count += 1
            return self._make_frame(self.state, self.levels_completed)

        self.step_count += 1
        act = action.action

        if act == GameAction.RESET:
            self.state = GameState.PLAYING
            self._build_level()
            return self._make_frame(self.state, self.levels_completed)

        if act == GameAction.ACTION6 and action.data is not None:
            x = int(action.data["x"])
            y = int(action.data["y"])
            if (x, y) == self._target_pos:
                if self.level < self.win_levels - 1:
                    self._advance_level()
                    self._build_level()
                else:
                    self.levels_completed = self.win_levels
                    self.state = GameState.WIN

        return self._make_frame(self.state, self.levels_completed)

    def _build_level(self) -> None:
        self._grid = [[0 for _ in range(self._width)] for _ in range(self._height)]
        rng = random.Random(self.seed + self.level)
        self._target_pos = (rng.randint(0, self._width - 1), rng.randint(0, self._height - 1))
        tx, ty = self._target_pos
        self._grid[ty][tx] = 4


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIM_GAMES: dict[str, type[SimGame]] = {
    "sim_nav": NavigateGame,
    "sim_cm": ColorMatchGame,
    "sim_click": ClickTargetGame,
}


def get_sim_info(game_id: str) -> EnvironmentInfo:
    game_cls = SIM_GAMES[game_id]
    return game_cls(game_id).info()


def list_sim_games() -> list[EnvironmentInfo]:
    return [get_sim_info(gid) for gid in SIM_GAMES]

# === Embedded: arc_agi3/rust_bridge.py ===
"""Bridge to the Rust ``goldsnnail`` compositional solver.

When the Rust crate is compiled and ``cargo`` is available on PATH, this
module can invoke the ``arc_compositional_solver`` example to find a program
that explains a grid-to-grid transformation.  A pure-Python fallback is
provided for the most common operations so the world model remains usable
even without Rust.
"""


import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


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
            cwd=str(Path.cwd()),
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

# === Embedded: arc_agi3/perception.py ===
"""Perception module for ARC-AGI-3 interactive-agent framework.

Extracts structured scene representations from raw grid observations using
4-connected connected-components, computes frame-level features, diffs
between frames, and produces stable state signatures.

Imports: only standard library + numpy.  No torch / arc_agi.
"""


import hashlib
from collections import defaultdict
from typing import Any, Optional

import numpy as np



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cc4(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping color -> list of (r,c) for 4-connected components."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val == 0 or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == val:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components[val] = pixels
    return components


def _bbox(pixels: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return (min(rs), min(cs), max(rs), max(cs))


def _centroid(pixels: list[tuple[int, int]]) -> tuple[float, float]:
    n = len(pixels)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in pixels) / n, sum(p[1] for p in pixels) / n)


def _overlap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = bbox_a
    r0b, c0b, r1b, c1b = bbox_b
    r0 = max(r0a, r0b)
    c0 = max(c0a, c0b)
    r1 = min(r1a, r1b)
    c1 = min(c1a, c1b)
    if r1 < r0 or c1 < c0:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


# ---------------------------------------------------------------------------
# Perception class
# ---------------------------------------------------------------------------

class Perception:
    """Extract structured scene data from ARC-AGI-3 frame observations."""

    def __init__(self, ignore_border: int = 0) -> None:
        self.ignore_border = ignore_border

    def _detect_border(self, grid: np.ndarray) -> int:
        """Auto-detect uniform border rows/cols and return border width."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return 0
        top_row = grid[0, :]
        bottom_row = grid[h - 1, :]
        left_col = grid[:, 0]
        right_col = grid[:, w - 1]
        # Border if first/last rows are uniform non-zero.
        if (
            np.all(top_row == top_row[0])
            and np.all(bottom_row == bottom_row[0])
            and top_row[0] != 0
            and bottom_row[0] != 0
        ):
            return 1
        return 0

    def _apply_border_mask(self, grid: np.ndarray) -> np.ndarray:
        """Return a copy of grid with border cells zeroed out."""
        if self.ignore_border <= 0:
            return grid
        masked = grid.copy()
        h, w = masked.shape
        b = self.ignore_border
        if h > 2 * b:
            masked[:b, :] = 0
            masked[h - b :, :] = 0
        if w > 2 * b:
            masked[:, :b] = 0
            masked[:, w - b :] = 0
        return masked

    def extract_objects(self, grid: Optional[np.ndarray]) -> list[SceneObject]:
        if grid is None or grid.size == 0:
            return []
        # Auto-detect border if not explicitly set.
        if self.ignore_border == 0:
            self.ignore_border = self._detect_border(grid)
        masked = self._apply_border_mask(grid)
        components = _cc4(masked)
        objects: list[SceneObject] = []
        oid = 0
        for color, pixels in sorted(components.items()):
            if not pixels:
                continue
            bbox = _bbox(pixels)
            cen = _centroid(pixels)
            objects.append(SceneObject(
                object_id=oid,
                color=int(color),
                bbox=bbox,
                centroid=cen,
                area=len(pixels),
                pixels=pixels,
            ))
            oid += 1
        return objects

    def observe(self, frame: Optional[FrameData]) -> SceneGraph:
        if frame is None:
            return SceneGraph(objects=[], width=0, height=0, frame=None)
        g = frame.grid()
        if g is None:
            return SceneGraph(objects=[], width=0, height=0, frame=None)
        return SceneGraph(
            objects=self.extract_objects(g),
            width=int(g.shape[1]),
            height=int(g.shape[0]),
            frame=g,
        )

    def frame_features(self, frame: Optional[FrameData]) -> dict[str, Any]:
        if frame is None:
            return {"width": 0, "height": 0, "color_histogram": {}, "n_objects": 0, "colors_present": [], "nonzero_count": 0}
        g = frame.grid()
        if g is None:
            return {"width": 0, "height": 0, "color_histogram": {}, "n_objects": 0, "colors_present": [], "nonzero_count": 0}
        colors, counts = np.unique(g[g != 0], return_counts=True)
        hist = {int(c): int(n) for c, n in zip(colors, counts)}
        sg = self.extract_objects(g)
        return {
            "width": int(g.shape[1]),
            "height": int(g.shape[0]),
            "color_histogram": hist,
            "n_objects": len(sg),
            "colors_present": sorted(hist.keys()),
            "nonzero_count": int(np.count_nonzero(g)),
        }

    def diff(self, before: Optional[FrameData], after: Optional[FrameData]) -> dict[str, Any]:
        if before is None or after is None:
            return {"compatible": False}
        gb = before.grid()
        ga = after.grid()
        if gb is None or ga is None:
            return {"compatible": False}
        if gb.shape != ga.shape:
            return {"compatible": False}

        h, w = gb.shape
        # Mask border region before diffing (Task 7).
        masked_gb = self._apply_border_mask(gb)
        masked_ga = self._apply_border_mask(ga)
        changed_cells: list[tuple[int, int, int, int]] = []
        color_changes: list[dict[str, int]] = []
        for r in range(h):
            for c in range(w):
                ov = int(masked_gb[r, c])
                nv = int(masked_ga[r, c])
                if ov != nv:
                    changed_cells.append((r, c, ov, nv))
                    if ov != 0 and nv != 0:
                        color_changes.append({"r": r, "c": c, "old": ov, "new": nv})

        sg_b = self.observe(before)
        sg_a = self.observe(after)

        # Build color-keyed lists
        before_by_color: dict[int, list[SceneObject]] = defaultdict(list)
        for obj in sg_b.objects:
            before_by_color[obj.color].append(obj)
        after_by_color: dict[int, list[SceneObject]] = defaultdict(list)
        for obj in sg_a.objects:
            after_by_color[obj.color].append(obj)

        moved_objects: list[dict[str, Any]] = []
        added_objects: list[SceneObject] = []
        removed_objects: list[SceneObject] = []
        matched_before: set[int] = set()
        matched_after: set[int] = set()

        all_colors = set(before_by_color) | set(after_by_color)
        for color in sorted(all_colors):
            blist = before_by_color.get(color, [])
            alist = after_by_color.get(color, [])
            # Greedy match by maximal overlap
            used_a = set()
            for bo in blist:
                best_ao = None
                best_overlap = -1
                for idx, ao in enumerate(alist):
                    if idx in used_a:
                        continue
                    overlap = _overlap(bo.bbox, ao.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_ao = (idx, ao)
                if best_ao is not None:
                    idx, ao = best_ao
                    used_a.add(idx)
                    matched_before.add(bo.object_id)
                    matched_after.add(ao.object_id)
                    dr = round(ao.centroid[0] - bo.centroid[0])
                    dc = round(ao.centroid[1] - bo.centroid[1])
                    if bo.bbox != ao.bbox or dr != 0 or dc != 0:
                        moved_objects.append({
                            "color": color,
                            "from_bbox": bo.bbox,
                            "to_bbox": ao.bbox,
                            "dr": dr,
                            "dc": dc,
                            "from_centroid": bo.centroid,
                            "to_centroid": ao.centroid,
                        })
            # Unmatched before -> removed
            for bo in blist:
                if bo.object_id not in matched_before:
                    removed_objects.append(bo)
            # Unmatched after -> added
            for idx, ao in enumerate(alist):
                if idx not in used_a:
                    added_objects.append(ao)

        return {
            "compatible": True,
            "changed_cells": changed_cells,
            "moved_objects": moved_objects,
            "added_objects": added_objects,
            "removed_objects": removed_objects,
            "color_changes": color_changes,
        }

    def signature(self, frame: Optional[FrameData]) -> str:
        if frame is None:
            return "none"
        g = frame.grid()
        if g is None:
            return "none"
        # Auto-detect border on first call.
        if self.ignore_border == 0:
            self.ignore_border = self._detect_border(g)
        masked = self._apply_border_mask(g)
        # Hash the masked grid bytes directly (shape included to avoid
        # cross-shape collisions). Same dedup semantics as the previous
        # per-cell string format, but ~30x faster on dense 64x64 grids.
        h = hashlib.md5()
        h.update(str(masked.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(masked).tobytes())
        return h.hexdigest()

    def describe_scene(self, scene: SceneGraph) -> str:
        if not scene.objects:
            return "empty scene"
        parts = []
        for obj in scene.objects:
            cr, cc = obj.centroid
            parts.append(f"color{obj.color}@({int(cr)},{int(cc)}) area{obj.area}")
        return "objects: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience aliases
# ---------------------------------------------------------------------------

_perception = Perception()

extract_objects = _perception.extract_objects
observe = _perception.observe
frame_features = _perception.frame_features
diff_frames = _perception.diff
scene_signature = _perception.signature


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# === Embedded: arc_agi3/memory.py ===
"""Memory module for the ARC-AGI-3 interactive-agent framework.

Provides :class:`MemoryStore`, an append-only, searchable JSONL store of
transitions, hypotheses, and events keyed by ``(game_id, seed)``.

Only the standard library and :mod:`numpy` are used, and the module imports
cleanly even when the real ``arc_agi`` / ``arcengine`` packages are absent.
"""


import hashlib
import json
import os
from typing import Any, Callable, Optional

import numpy as np


try:  # Perception lives in arc_agi3.perception; tolerate its absence.
    pass
except Exception:  # pragma: no cover - defensive against missing module
    Perception = None


def _local_signature(frame: FrameData) -> str:
    """Fallback state hash based on the grid pixels (deterministic)."""
    g = frame.grid()
    if g is None:
        return "none"
    return hashlib.md5(g.tobytes()).hexdigest()


class _FallbackPerception:
    """Minimal stand-in providing ``signature`` when perception is missing."""

    def signature(self, frame: FrameData) -> str:
        return _local_signature(frame)

    def observe(self, frame: FrameData):  # pragma: no cover - unused by memory
        raise NotImplementedError("Fallback Perception does not implement observe()")


_PERCEPTION = Perception() if Perception is not None else _FallbackPerception()


class MemoryStore:
    """Append-only, searchable JSONL memory store for a single (game, seed).

    Every ``record_*`` call appends a JSON-safe line to a ``.jsonl`` file and
    flushes, while keeping in-memory indices so queries are O(n) at worst and
    often O(1). Reopening a store on the same path resumes from disk.
    """

    def __init__(self, game_id: str, seed: int = 0, memory_dir: str = "memory") -> None:
        self.game_id = game_id
        self.seed = seed
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.path = os.path.join(memory_dir, f"{game_id}_{seed}.jsonl")

        # Raw JSONL records (dicts) kept in memory for scanning/search.
        self._records: list[dict[str, Any]] = []
        # Derived in-memory indices.
        self._transitions: list[Transition] = []
        self._hypotheses: list[Hypothesis] = []
        self._events: list[dict[str, Any]] = []

        self._load_existing()
        # Open for appending; flushed on every write.
        self._fh = open(self.path, "a", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Loading / persistence helpers
    # ------------------------------------------------------------------ #

    def _load_existing(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._records.append(rec)
                self._index_record(rec)

    def _index_record(self, rec: dict[str, Any]) -> None:
        rtype = rec.get("type")
        if rtype == "transition":
            self._transitions.append(self._transition_from_json(rec))
        elif rtype == "hypothesis":
            self._hypotheses.append(self._hypothesis_from_json(rec))
        elif rtype == "hypothesis_update":
            self._apply_hypothesis_update(rec)
        elif rtype == "event":
            self._events.append(rec)

    def _append(self, rec: dict[str, Any]) -> None:
        self._fh.write(json.dumps(rec))
        self._fh.write("\n")
        self._fh.flush()
        self._records.append(rec)
        self._index_record(rec)

    # ------------------------------------------------------------------ #
    # (De)serialization
    # ------------------------------------------------------------------ #

    def _frame_to_json(self, frame: FrameData) -> dict[str, Any]:
        action_input = frame.action_input
        if action_input is not None and hasattr(action_input, "model_dump"):
            action_input = action_input.model_dump(mode="json")
        return {
            "game_id": frame.game_id,
            "state": frame.state.value,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "available_actions": list(frame.available_actions),
            "full_reset": frame.full_reset,
            "guid": frame.guid,
            "action_input": action_input,
            "frame": frame.frame,  # list[list[int]] or None
            "step": frame.step,
            "score": frame.score,
        }

    def _frame_from_json(self, d: dict[str, Any]) -> FrameData:
        return FrameData(
            game_id=d["game_id"],
            state=GameState(d["state"]),
            levels_completed=d.get("levels_completed", 0),
            win_levels=d.get("win_levels", 0),
            available_actions=list(d.get("available_actions", [])),
            full_reset=d.get("full_reset", False),
            guid=d.get("guid"),
            action_input=d.get("action_input"),
            frame=d.get("frame"),
            step=d.get("step", 0),
            score=d.get("score", 0.0),
        )

    def _transition_from_json(self, rec: dict[str, Any]) -> Transition:
        return Transition(
            state_before=self._frame_from_json(rec["before"]),
            action=Action.from_dict(rec["action"]),
            state_after=self._frame_from_json(rec["after"]),
            step=rec["step"],
        )

    def _hypothesis_from_json(self, rec: dict[str, Any]) -> Hypothesis:
        action_val = rec.get("action")
        return Hypothesis(
            id=rec["id"],
            action=GameAction(action_val) if action_val is not None else None,
            description=rec["description"],
            program=rec.get("program"),
            confidence=rec["confidence"],
            supporting_transitions=rec["supporting_transitions"],
            contradicting_transitions=rec["contradicting_transitions"],
            created_step=rec["created_step"],
        )

    def _apply_hypothesis_update(self, rec: dict[str, Any]) -> None:
        hyp_id = rec.get("id")
        for h in self._hypotheses:
            if h.id == hyp_id:
                if "confidence" in rec:
                    h.confidence = rec["confidence"]
                if "contradicting_transitions" in rec:
                    h.contradicting_transitions = rec["contradicting_transitions"]
                if "supporting_transitions" in rec:
                    h.supporting_transitions = rec["supporting_transitions"]
                return

    # ------------------------------------------------------------------ #
    # Recording API
    # ------------------------------------------------------------------ #

    def record_transition(self, t: Transition) -> None:
        rec = {
            "type": "transition",
            "step": t.step,
            "action": t.action.to_dict(),
            "before": self._frame_to_json(t.state_before),
            "after": self._frame_to_json(t.state_after),
        }
        self._append(rec)

    def record_hypothesis(self, h: Hypothesis) -> None:
        rec = {
            "type": "hypothesis",
            "id": h.id,
            "action": h.action.value if h.action is not None else None,
            "description": h.description,
            "program": h.program,
            "confidence": h.confidence,
            "supporting_transitions": h.supporting_transitions,
            "contradicting_transitions": h.contradicting_transitions,
            "created_step": h.created_step,
        }
        self._append(rec)

    def record_event(self, event_type: str, data: dict) -> None:
        rec = {"type": "event", "event": event_type, "data": data}
        self._append(rec)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    def hypotheses(self, action: Optional[GameAction] = None) -> list[Hypothesis]:
        if action is None:
            return list(self._hypotheses)
        return [h for h in self._hypotheses if h.action == action]

    def best_hypothesis(
        self, action: Optional[GameAction] = None
    ) -> Optional[Hypothesis]:
        cands = self.hypotheses(action)
        if not cands:
            return None
        return max(
            cands,
            key=lambda h: (h.confidence, h.supporting_transitions),
        )

    def contradict(self, hyp_id: str, transition: Transition) -> None:
        target = None
        for h in self._hypotheses:
            if h.id == hyp_id:
                target = h
                break
        if target is None:
            return

        target.contradicting_transitions += 1
        target.confidence = max(0.0, target.confidence * 0.8)

        # Event record documenting the contradiction.
        self.record_event(
            "contradiction",
            {
                "hypothesis_id": hyp_id,
                "transition_step": getattr(transition, "step", None),
                "new_confidence": target.confidence,
                "contradicting_transitions": target.contradicting_transitions,
            },
        )

        # Persisted hypothesis update so resume keeps consistent state.
        update = {
            "type": "hypothesis_update",
            "id": hyp_id,
            "confidence": target.confidence,
            "contradicting_transitions": target.contradicting_transitions,
        }
        self._append(update)

    def _grid_similarity(
        self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> float:
        if a is None and b is None:
            return 1.0
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=int)
        b = np.asarray(b, dtype=int)
        if a.shape != b.shape:
            # Compare on the overlapping top-left region only.
            h = min(a.shape[0], b.shape[0]) if a.ndim and b.ndim else 0
            w = min(a.shape[1], b.shape[1]) if a.ndim and b.ndim else 0
            if h <= 0 or w <= 0:
                return 0.0
            a = a[:h, :w]
            b = b[:h, :w]
        nonzero = (a != 0) | (b != 0)
        total = int(nonzero.sum())
        if total == 0:
            return 1.0
        equal = int(((a == b) & nonzero).sum())
        return equal / total

    def similar_transitions(
        self, frame: FrameData, action: GameAction, k: int = 5
    ) -> list[Transition]:
        query_sig = _PERCEPTION.signature(frame)
        query_grid = frame.grid()

        scored: list[tuple[float, Transition]] = []
        for t in self._transitions:
            if t.action.action != action:
                continue
            before_sig = _PERCEPTION.signature(t.state_before)
            if before_sig == query_sig:
                # Exact state match ranks above any partial overlap.
                overlap = self._grid_similarity(query_grid, t.state_before.grid())
                scored.append((2.0 + overlap, t))
            else:
                overlap = self._grid_similarity(query_grid, t.state_before.grid())
                scored.append((overlap, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def search(self, predicate: Callable[[dict], bool]) -> list[dict]:
        return [rec for rec in self._records if predicate(rec)]

    def summary(self) -> dict:
        signatures: set[str] = set()
        for t in self._transitions:
            signatures.add(_PERCEPTION.signature(t.state_before))
            signatures.add(_PERCEPTION.signature(t.state_after))
        actions_seen = []
        seen = set()
        for t in self._transitions:
            code = t.action.action.value
            if code not in seen:
                seen.add(code)
                actions_seen.append(code)
        return {
            "n_transitions": len(self._transitions),
            "n_hypotheses": len(self._hypotheses),
            "n_events": len(self._events),
            "actions_seen": actions_seen,
            "distinct_states": len(signatures),
        }

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def create_memory(
    game_id: str, seed: int = 0, memory_dir: str = "memory"
) -> MemoryStore:
    """Convenience factory returning a configured :class:`MemoryStore`."""
    return MemoryStore(game_id=game_id, seed=seed, memory_dir=memory_dir)

# === Embedded: arc_agi3/env.py ===
"""Toolkit adapter bridging the real ``arc_agi`` package and our simulations.

Provides :class:`Environment` (wraps a single game backend) and :class:`Arcade`
(the main entry point for creating environments).
"""


import json
import os
import time
from typing import Any, Optional


try:
    import arc_agi  # type: ignore
    from arc_agi import Arcade as _RealArcade  # type: ignore
    from arc_agi import OperationMode as _RealOM  # type: ignore

    _REAL_AVAILABLE = True
except ImportError:
    _REAL_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Environment:
    """Wraps a single ARC-AGI-3 environment (real toolkit or simulated)."""

    def __init__(
        self,
        backend: Any,
        game_id: str,
        seed: int = 0,
        save_recording: bool = False,
        recordings_dir: str = "recordings",
        renderer: Any = None,
        include_frame_data: bool = True,
    ) -> None:
        self._backend = backend
        self.game_id = game_id
        self.seed = seed
        self.save_recording = save_recording
        self.recordings_dir = recordings_dir
        self.renderer = renderer
        self.include_frame_data = include_frame_data
        self._step_count = 0
        self._recording_path: Optional[str] = None
        self._frame_before: Optional[FrameData] = None

        if self.save_recording:
            os.makedirs(self.recordings_dir, exist_ok=True)
            self._recording_path = os.path.join(
                self.recordings_dir, f"{game_id}_{seed}.jsonl"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, action: Action) -> FrameData:
        """Execute one step in the environment."""
        self._frame_before = self._snapshot()
        result = self._backend_step(action)
        self._step_count += 1
        if self.save_recording and self._recording_path:
            self._append_recording(action, result)
        return result

    def reset(self) -> FrameData:
        """Reset the environment to its initial state."""
        self._step_count = 0
        self._frame_before = None
        if hasattr(self._backend, "reset") and callable(self._backend.reset):
            return self._backend.reset(self.seed)
        # Fallback: issue RESET action
        reset_action = Action(action=GameAction.RESET)
        return self.step(reset_action)

    def action_space(self) -> list[int]:
        if hasattr(self._backend, "action_space") and callable(self._backend.action_space):
            return self._backend.action_space()
        return [a.value for a in ALL_ACTIONS]

    def info(self) -> EnvironmentInfo:
        if hasattr(self._backend, "info") and callable(self._backend.info):
            return self._backend.info()
        return get_sim_info(self.game_id)

    def close(self) -> None:
        if hasattr(self._backend, "close") and callable(self._backend.close):
            self._backend.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _backend_step(self, action: Action) -> FrameData:
        backend = self._backend

        if isinstance(backend, SimGame):
            return backend.step(action)

        if _REAL_AVAILABLE:
            try:
                result = backend.step(action)
                # _RealToolkitWrapper.step already returns a FrameData (converted
                # from the raw toolkit frame), so avoid a redundant second
                # conversion via _from_toolkit_frame.
                if isinstance(result, FrameData):
                    return result
                return _from_toolkit_frame(result) if result is not None else _empty_frame(self.game_id)
            except Exception as exc:
                logger.warning("Real toolkit step failed: %s", exc)
                raise

        raise RuntimeError("No valid backend available.")

    def _snapshot(self) -> Optional[FrameData]:
        if hasattr(self._backend, "_make_frame"):
            return self._backend._make_frame(
                self._backend.state, self._backend.levels_completed
            )
        return None

    def _append_recording(self, action: Action, frame_after: FrameData) -> None:
        if not self._recording_path:
            return
        entry = {
            "game_id": self.game_id,
            "seed": self.seed,
            "step": self._step_count,
            "action": action.to_dict(),
            "frame_before": self._frame_before.frame if self._frame_before else None,
            "frame_after": frame_after.frame,
            "state": frame_after.state.value,
        }
        with open(self._recording_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Arcade
# ---------------------------------------------------------------------------


class Arcade:
    """Main entry point for creating ARC-AGI-3 environments.

    Usage::

        arc = Arcade(operation_mode=OperationMode.OFFLINE)
        env = arc.make("sim_nav", seed=0)
        obs = env.reset()
        obs = env.step(Action(GameAction.ACTION1))
    """

    def __init__(
        self,
        operation_mode: OperationMode = OperationMode.OFFLINE,
        environments_dir: str = "environment_files",
        recordings_dir: str = "recordings",
        logger: Any = None,
        arc_api_key: str = "",
    ) -> None:
        self.operation_mode = operation_mode
        self.environments_dir = environments_dir
        self.recordings_dir = recordings_dir
        # Forward the key explicitly (env var / .env) so the toolkit never
        # silently falls back to an anonymous key.
        self.arc_api_key = arc_api_key or os.environ.get("ARC_API_KEY", "")
        self.logger = logger or logging.getLogger(__name__)
        self._entries: list[ScorecardEntry] = []
        self._real_arcade: Any = None
        if _REAL_AVAILABLE:
            try:
                self._real_arcade = _RealArcade(
                    operation_mode=_RealOM(operation_mode.value),
                    environments_dir=environments_dir,
                    recordings_dir=recordings_dir,
                    arc_api_key=self.arc_api_key,
                )
                self.logger.info("Real arc_agi toolkit loaded (mode=%s).", operation_mode.value)
            except Exception as exc:
                self.logger.warning("Failed to load real arc_agi toolkit: %s", exc)

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def make(
        self,
        game_id: str,
        seed: int = 0,
        scorecard_id: Optional[str] = None,
        save_recording: bool = False,
        include_frame_data: bool = True,
        render_mode: Optional[str] = None,
        renderer: Any = None,
    ) -> Environment:
        """Create an environment for ``game_id`` with the given ``seed``."""
        backend = self._build_backend(game_id, seed)
        return Environment(
            backend=backend,
            game_id=game_id,
            seed=seed,
            save_recording=save_recording,
            recordings_dir=self.recordings_dir,
            renderer=renderer,
            include_frame_data=include_frame_data,
        )

    # ------------------------------------------------------------------
    # Metadata / scorecard
    # ------------------------------------------------------------------

    def get_environments(self) -> list[EnvironmentInfo]:
        infos: list[EnvironmentInfo] = list(list_sim_games())
        if self._real_arcade is not None and hasattr(self._real_arcade, "get_environments"):
            try:
                for raw in self._real_arcade.get_environments():
                    infos.append(
                        EnvironmentInfo(
                            game_id=getattr(raw, "game_id", "unknown"),
                            title=getattr(raw, "title", ""),
                            tags=getattr(raw, "tags", []),
                            n_levels=getattr(raw, "n_levels", 0),
                            baseline_actions=getattr(raw, "baseline_actions", 0),
                            is_simulated=False,
                        )
                    )
            except Exception as exc:
                self.logger.warning("Could not fetch real environments: %s", exc)
        return infos

    def record_entry(self, entry: ScorecardEntry) -> None:
        self._entries.append(entry)

    def get_scorecard(self) -> dict[str, Any]:
        total = sum(e.total_score for e in self._entries)
        return {
            "entries": self._entries,
            "total": total,
            "n_games": len(self._entries),
        }

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _build_backend(self, game_id: str, seed: int) -> Any:
        if game_id in SIM_GAMES:
            return SIM_GAMES[game_id](game_id=game_id, seed=seed)

        if self._real_arcade is not None:
            try:
                real_env = self._real_arcade.make(game_id=game_id, seed=seed)
                return _RealToolkitWrapper(real_env)
            except Exception as exc:
                self.logger.warning("Real toolkit failed for %s: %s", game_id, exc)

        if game_id in SIM_GAMES:
            return SIM_GAMES[game_id](game_id=game_id, seed=seed)

        raise ValueError(
            f"Game '{game_id}' not found in simulations and real toolkit is unavailable."
        )


# ---------------------------------------------------------------------------
# Real-toolkit wrapper
# ---------------------------------------------------------------------------


class _RealToolkitWrapper:
    """Thin wrapper translating between real toolkit objects and our types."""

    def __init__(self, real_env: Any) -> None:
        self._real = real_env
        self._level = 0
        self._levels_completed = 0
        self._win_levels = 0
        self.state = GameState.PLAYING

    def reset(self, seed: int) -> FrameData:
        if hasattr(self._real, "reset") and callable(self._real.reset):
            raw = self._real.reset()
        else:
            tk_action, _data = _to_toolkit_action(Action(GameAction.RESET))
            raw = self._real.step(tk_action, data=_data)
        return _from_toolkit_frame(raw) if raw is not None else _empty_frame(getattr(self._real, "game_id", "unknown"))

    def step(self, action: Action) -> FrameData:
        tk_action, data = _to_toolkit_action(action)
        reasoning = action.reasoning
        raw = self._real.step(tk_action, data=data, reasoning=reasoning)
        if raw is None:
            # Server-side session may have expired (~30-35s timeout).
            # Refresh by re-resetting and retry the step once.
            game_id = getattr(self._real, "game_id", "unknown")
            logger.info("Step returned None for %s, refreshing session and retrying", game_id)
            try:
                self._real.reset()
            except Exception as exc:
                logger.warning("Re-reset failed for %s: %s", game_id, exc)
            raw = self._real.step(tk_action, data=data, reasoning=reasoning)
            if raw is None:
                return _empty_frame(game_id)
        fd = _from_toolkit_frame(raw)
        self.state = fd.state
        self._levels_completed = fd.levels_completed
        return fd

    def action_space(self) -> list[int]:
        if hasattr(self._real, "action_space"):
            return self._real.action_space()
        return [a.value for a in ALL_ACTIONS]

    def info(self) -> EnvironmentInfo:
        if hasattr(self._real, "info"):
            raw = self._real.info
            ri = raw() if callable(raw) else raw
            return EnvironmentInfo(
                game_id=getattr(ri, "game_id", getattr(self._real, "game_id", "unknown")),
                title=getattr(ri, "title", ""),
                tags=getattr(ri, "tags", []),
                n_levels=getattr(ri, "n_levels", 0),
                baseline_actions=getattr(ri, "baseline_actions", 0),
                is_simulated=False,
            )
        return EnvironmentInfo(
            game_id=getattr(self._real, "game_id", "unknown"),
            title="",
            tags=[],
            n_levels=0,
            baseline_actions=0,
            is_simulated=False,
        )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_toolkit_action(action: Action) -> Any:
    """Convert our Action to an arcengine.GameAction enum member.

    Returns a tuple of (game_action, data) so callers can pass data separately
    to RemoteEnvironmentWrapper.step which expects (action, data, reasoning).
    """
    if not _REAL_AVAILABLE:
        raise RuntimeError("Real toolkit not installed.")
    from arcengine import GameAction as _TKAction  # type: ignore
    tk = _TKAction[action.action.name]
    data = action.data
    if data is not None:
        tk.set_data(data)
    return tk, data


def _from_toolkit_frame(raw: Any) -> FrameData:
    frame_list = getattr(raw, "frame", None)
    if frame_list is None and hasattr(raw, "frame"):
        # Some toolkits expose .frame lazily
        frame_list = raw.frame  # type: ignore
    if isinstance(frame_list, list) and frame_list and hasattr(frame_list[0], "tolist"):
        frame_list = frame_list[0].tolist()
    elif hasattr(frame_list, "tolist"):
        frame_list = frame_list.tolist()
    return FrameData(
        game_id=getattr(raw, "game_id", ""),
        state=GameState(getattr(raw, "state", GameState.UNKNOWN.value)),
        levels_completed=getattr(raw, "levels_completed", 0),
        win_levels=getattr(raw, "win_levels", 0),
        available_actions=list(getattr(raw, "available_actions", [])),
        full_reset=bool(getattr(raw, "full_reset", False)),
        guid=getattr(raw, "guid", None),
        action_input=getattr(raw, "action_input", None),
        frame=frame_list if isinstance(frame_list, list) else None,
        step=getattr(raw, "step", 0),
        score=float(getattr(raw, "score", 0.0)),
    )


def _empty_frame(game_id: str) -> FrameData:
    return FrameData(game_id=game_id, state=GameState.UNKNOWN)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def competition_score(level_scores: list[float]) -> float:
    """Competition scoring: each level score is squared, then weighted by level index.

    Matches the plan's rule: ``total = sum(s_i**2 * (i+1) for i, s_i in enumerate(level_scores))``.

    The result is not normalized to [0, 1]; higher is better.
    """
    if not level_scores:
        return 0.0
    total = sum((float(s) ** 2) * (i + 1) for i, s in enumerate(level_scores))
    return total


def simple_score(level_scores: list[float]) -> float:
    """Simple unweighted sum of level scores (for dev comparison)."""
    return float(sum(level_scores))


# ---------------------------------------------------------------------------
# Default arcade
# ---------------------------------------------------------------------------


def get_default_arcade(
    operation_mode: OperationMode = OperationMode.OFFLINE, **kw: Any
) -> Arcade:
    return Arcade(operation_mode=operation_mode, **kw)

# === Embedded: arc_agi3/world_model.py ===
"""World-model / rule-learner module for the ARC-AGI-3 interactive-agent framework.

Learns parametric transition rules from observed (state, action) -> state
transitions so the agent can predict and simulate future states for planning.
"""


import os
import sys

# === Embedded: arc_agi3/verifier.py ===
"""Verification module for the ARC-AGI-3 interactive-agent framework.

Implements the predict-act-compare verification loop. Before committing to
planned actions, the agent predicts the outcome via the world model, executes
the action in the real environment, and compares. A mismatch marks the world
model stale and records a contradiction.
"""


import os
import sys

# === Embedded: arc_agi3/planner.py ===
"""Planner for the ARC-AGI-3 interactive-agent framework.

Two-phase behaviour (per the rebuild plan):

* Phase 1 — Exploration: ``directed_explore`` probes the environment with a
  Thompson-sampling / epsilon-greedy bandit to gather initial transitions that
  the world model can learn from.
* Phase 3 — Goal search: ``search_plan`` runs a breadth-first search over
  action sequences, using the :class:`WorldModel` as a simulator, and
  ``next_action`` chooses between a planned action, a perception-driven complex
  action, or another exploration step.
"""


import os
import random
import sys
from typing import Callable, Optional, Tuple

# Allow importing the repo-root bandit if present.
_REPO_ROOT = os.path.join(os.getcwd(), "repo_root")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_SIMPLE_CODES = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Directed-exploration bandit
# ---------------------------------------------------------------------------


class _SimpleBandit:
    """Epsilon-greedy bandit with per-arm success/total counts."""

    def __init__(self, n_arms: int, epsilon: float = 0.15) -> None:
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.success = [0] * n_arms
        self.total = [0] * n_arms
        self._rng = random.Random(1234)

    def select_arm(self, context=None) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_arms)
        means = [
            (self.success[i] / self.total[i]) if self.total[i] > 0 else 0.5
            for i in range(self.n_arms)
        ]
        return int(max(range(self.n_arms), key=lambda i: means[i]))

    def update(self, arm: int, reward: float) -> None:
        if 0 <= arm < self.n_arms:
            self.total[arm] += 1
            if reward > 0.5:
                self.success[arm] += 1


def _make_bandit(n_arms: int = 7):
    """Prefer the repo-root ThompsonSamplingContextualBandit; fall back to simple."""
    try:
        from positional_bandit import ThompsonSamplingContextualBandit  # type: ignore

        real = ThompsonSamplingContextualBandit(n_arms=n_arms, context_dim=10)

        class _Wrap:
            def select_arm(self, context):
                # The real bandit expects a context vector; returns an arm index.
                try:
                    return int(real.select_arm(context))
                except Exception:
                    return random.randrange(n_arms)

            def update(self, arm, reward):
                try:
                    real.update(arm, reward)
                except Exception:
                    pass

        return _Wrap()
    except Exception:
        return _SimpleBandit(n_arms)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Directed exploration + goal-directed BFS planning."""

    def __init__(
        self,
        world_model: WorldModel,
        perception: Perception,
        memory: MemoryStore,
        explore_probes: int = 6,
    ) -> None:
        self.world_model = world_model
        self.perception = perception
        self.memory = memory
        self.explore_probes = explore_probes
        self.bandit = _make_bandit(7)
        # State-action visited set (Task 3)
        self._visited: set[tuple[str, int]] = set()
        self._state_outcomes: dict[tuple[str, int], str] = {}
        self._last_levels_completed: int = -1

    def clear_visited(self) -> None:
        """Clear the visited set (call on level transition)."""
        self._visited.clear()
        self._state_outcomes.clear()

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _context(self, frame: FrameData) -> list[float]:
        ctx: list[float] = [
            float(frame.levels_completed),
            float(frame.win_levels),
            float(len(frame.available_actions)),
        ]
        grid = frame.grid()
        if grid is not None:
            flat = grid.flatten()
            counts = [int((flat == c).sum()) for c in range(1, 6)]
            ctx.extend(float(c) for c in counts)
        while len(ctx) < 10:
            ctx.append(0.0)
        return ctx[:10]

    def directed_explore(
        self, env: Environment, current_frame: FrameData, n_probes: Optional[int] = None
    ) -> list[Transition]:
        """Take a few directed/random actions to gather transitions."""
        n = n_probes if n_probes is not None else self.explore_probes
        collected: list[Transition] = []
        frame = current_frame
        for _ in range(n):
            arm = self.bandit.select_arm(self._context(frame))
            ga = GameAction.from_int(arm + 1)  # arm 0 -> ACTION1 ... 6 -> ACTION7
            if ga.is_complex():
                # Keep exploration on simple actions; complex handled by planning.
                ga = GameAction.ACTION1
            action = Action(ga)
            sig = self.perception.signature(frame)
            key = (sig, action.action.value)
            if key in self._visited:
                # Try a different untried simple action.
                tried = {a for (s, a) in self._visited if s == sig}
                untried = [c for c in frame.available_actions if c in _SIMPLE_CODES and c not in tried]
                if not untried:
                    break
                code = random.choice(untried)
                action = Action(GameAction.from_int(code))
            before = frame
            after = env.step(action)
            if after is None:
                break
            t = Transition(before, action, after, before.step)
            self.world_model.learn(t)
            sig_before = self.perception.signature(before)
            sig_after = self.perception.signature(after)
            reward = 1.0 if (
                after.levels_completed > before.levels_completed or sig_before != sig_after
            ) else 0.1
            self.bandit.update(arm, reward)
            collected.append(t)
            self._visited.add((sig_before, action.action.value))
            self._state_outcomes[(sig_before, action.action.value)] = sig_after
            frame = after
            if after.state.is_terminal():
                break
        return collected

    # ------------------------------------------------------------------
    # Goal-directed search
    # ------------------------------------------------------------------

    def search_plan(
        self,
        start_frame: FrameData,
        goal_test: Callable[[FrameData], bool],
        max_depth: int = 12,
        max_expansions: int = 4000,
    ) -> Optional[Plan]:
        """BFS over action sequences using the world model as a simulator."""
        from collections import deque

        start_sig = self.perception.signature(start_frame)
        if goal_test(start_frame):
            return Plan(steps=[], goal_state=start_frame, confidence=1.0)

        visited = {start_sig}
        parent: dict[str, tuple[str, Action, FrameData]] = {}
        queue: deque = deque()
        queue.append((start_frame, start_sig, 0))
        expansions = 0

        while queue and expansions < max_expansions:
            frame, sig, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for code in frame.available_actions:
                if code not in _SIMPLE_CODES:
                    continue
                key = (sig, code)
                if key in self._visited:
                    # Skip already-expanded edge.
                    continue
                action = Action(GameAction.from_int(code))
                nxt = self.world_model.predict(action, frame)
                if nxt is None:
                    continue
                nsig = self.perception.signature(nxt)
                # Record every expanded edge so it is simulated at most once
                # across all search_plan calls (previously edges leading to
                # already-visited states were re-expanded on every call).
                self._visited.add(key)
                self._state_outcomes[key] = nsig
                if nsig in visited:
                    continue
                visited.add(nsig)
                parent[nsig] = (sig, action, frame)
                expansions += 1
                if goal_test(nxt):
                    return self._reconstruct(start_sig, nsig, parent, nxt)
                queue.append((nxt, nsig, depth + 1))
        return None

    def _reconstruct(
        self,
        start_sig: str,
        goal_sig: str,
        parent: dict,
        goal_frame: FrameData,
    ) -> Plan:
        actions: list[Action] = []
        conf = 1.0
        sig = goal_sig
        while sig != start_sig:
            prev_sig, action, _frame = parent[sig]
            actions.append(action)
            c = self.world_model.confidence(action, _frame)
            conf *= max(0.05, c)
            sig = prev_sig
        actions.reverse()
        steps = [PlanStep(action=a, expected_state=None, description=f"sim ACTION{a.action.value}") for a in actions]
        return Plan(steps=steps, goal_state=goal_frame, confidence=conf)

    # ------------------------------------------------------------------
    # High-level action selection
    # ------------------------------------------------------------------

    def next_action(
        self,
        env: Environment,
        current_frame: FrameData,
        goal_test: Callable[[FrameData], bool],
        budget: int,
        allow_complex: bool = True,
    ) -> Tuple[Action, str]:
        # 1) Complex / perception-driven targeting (e.g. click-the-target games).
        avail = current_frame.available_actions or []
        if allow_complex and (6 in avail or 7 in avail):
            scene = self.perception.observe(current_frame)
            target = None
            for obj in scene.objects:
                if obj.color == 0:
                    continue
                if target is None or obj.area > target.area:
                    target = obj
            if target is not None:
                x = int(round(target.centroid[1]))
                y = int(round(target.centroid[0]))
                if 0 <= x <= 63 and 0 <= y <= 63:
                    return Action(GameAction.ACTION6, data={"x": x, "y": y}), "perception_targeted"

        # 2) Plan using the world model as a simulator.
        plan = self.search_plan(current_frame, goal_test, max_depth=min(14, max(4, budget)))
        if plan is not None and plan.steps:
            return plan.steps[0].action, "planned"

        # 2.5) Hill-climbing fallback: pick simple action with highest world-model confidence.
        best_action, best_score = None, -1.0
        for code in (current_frame.available_actions or []):
            if code not in _SIMPLE_CODES:
                continue
            a = Action(GameAction.from_int(code))
            score = self.world_model.confidence(a, current_frame)
            if score > best_score:
                best_score = score
                best_action = a
        if best_action is not None and best_score > 0.0:
            return best_action, "hill_climb"

        # 3) Explore one more step.
        ts = self.directed_explore(env, current_frame, n_probes=1)
        if ts:
            return ts[-1].action, "explore"

        # 4) Fallback: pick an untried simple action for this state (Fix 3).
        sig = self.perception.signature(current_frame)
        tried = {a for (s, a) in self._visited if s == sig}
        untried = [c for c in (current_frame.available_actions or []) if c in _SIMPLE_CODES and c not in tried]
        if untried:
            code = random.choice(untried)
            return Action(GameAction.from_int(code)), "untried_fallback"

        return Action(GameAction.ACTION1), "default"


def create_planner(
    world_model: WorldModel,
    perception: Perception,
    memory: MemoryStore,
    explore_probes: int = 6,
) -> Planner:
    return Planner(world_model, perception, memory, explore_probes=explore_probes)

# === Embedded: arc_agi3/llm_planner.py ===
"""Optional LLM-based action proposer for ARC-AGI-3.

Leading ARC-AGI-3 solutions drive an LLM to reason about each frame and choose
the next action. This module provides that capability **optionally**: if an API
key is present it proposes actions; otherwise ``available`` is ``False`` and the
agent transparently falls back to the rule-based :class:`Planner`.

Configuration (environment variables):
    OPENROUTER_API_KEY / OPENAI_API_KEY / ARC_LLM_API_KEY : API key (enables).
    ARC_LLM_BASE_URL / OPENROUTER_BASE_URL / OPENAI_BASE_URL : OpenAI-compatible base URL.
    ARC_LLM_MODEL / OPENROUTER_MODEL                            : model id (default ``z-ai/glm-5.2``).

OpenRouter is OpenAI-compatible: set ``OPENROUTER_API_KEY`` and use the slug
``z-ai/glm-5.2`` with base ``https://openrouter.ai/api/v1``.

The ``openai`` package is imported lazily, so this module never breaks the
framework when the dependency or key is absent.
"""


import json
import os
import re
from typing import Any, Optional



def render_frame(frame: FrameData) -> str:
    """Render a frame's grid as readable ASCII for the LLM prompt."""
    grid = frame.grid()
    if grid is None:
        return "(no grid / non-visual state)"
    rows = [" ".join(str(int(c)) for c in row) for row in grid.tolist()]
    return "\n".join(rows)


class LLMPlanner:
    """Proposes the next action using an OpenAI-compatible API (e.g. OpenRouter)."""

    def __init__(
        self,
        model: Optional[str] = None,
        world_model: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        self.api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ARC_LLM_API_KEY")
            or ""
        )
        self.base_url = (
            os.environ.get("ARC_LLM_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        self.model = (
            model
            or os.environ.get("ARC_LLM_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or "z-ai/glm-5.2"
        )
        self.world_model = world_model
        self.memory = memory
        self.available = bool(self.api_key)
        self._client: Any = None
        if self.available:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self.available = False
                self._client = None

        # Persistent CoT memory (Task 1)
        self._cot_memory: list[str] = []
        self._compaction_threshold = int(
            os.environ.get("ARC_LLM_COMPACT_THRESHOLD", "20")
        )
        self._total_steps = 0

    def _call(self, prompt: str) -> Optional[str]:
        """Call the model; prefer the Responses API, fall back to Chat Completions."""
        if self._client is None:
            return None
        try:
            resp = self._client.responses.create(model=self.model, input=prompt, timeout=30)
            text = getattr(resp, "output_text", None)
            if text:
                return text
        except Exception:
            pass
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30,
            )
            return (resp.choices[0].message.content or "") if resp.choices else ""
        except Exception:
            return None

    def _compact_history(self, history: list[str]) -> str:
        """Summarize history into 3-5 bullet points when it exceeds the threshold."""
        if len(history) <= self._compaction_threshold:
            return "\n".join(f"{i}: {h}" for i, h in enumerate(history[-6:])) or "(none)"
        try:
            compact_prompt = (
                "Summarize the following action history into 3-5 bullet points of "
                "what was tried and what happened. Be concise.\n\n"
                + "\n".join(f"{i}: {h}" for i, h in enumerate(history))
            )
            summary = self._call(compact_prompt)
            if summary:
                return f"(compacted history)\n{summary}"
        except Exception:
            pass
        # Fallback: show last 6 entries
        return "\n".join(f"{i}: {h}" for i, h in enumerate(history[-6:])) or "(none)"

    def _build_prompt(
        self,
        frame: FrameData,
        history: list[str],
        available_actions: list[int],
        known_rules: str = "",
        failed_attempts: str = "",
    ) -> str:
        grid_txt = render_frame(frame)
        avail = ", ".join(str(a) for a in available_actions if a not in (-1, 0))
        hist_txt = self._compact_history(history)

        # CoT memory: last 3 rationales (Task 1)
        cot_txt = "(none)"
        if self._cot_memory:
            cot_txt = "\n".join(
                f"- {r}" for r in self._cot_memory[-3:]
            )

        # Structured scene (Task 2)
        scene_txt = "n/a"
        try:

            sc = Perception().observe(frame)
            objs = [
                f"color{o.color}@({int(o.centroid[0])},{int(o.centroid[1])}) area{o.area}"
                for o in sc.objects
                if o.color != 0
            ]
            scene_txt = "; ".join(objs) if objs else "none"
        except Exception:
            pass

        prompt = (
            "You are playing an interactive ARC-AGI-3 grid game. The grid below "
            "shows the current state as rows of integers (0 = background).\n\n"
            f"{grid_txt}\n\n"
            f"Distinct non-background objects: {scene_txt}\n"
            f"Available action codes (ignore -1 and 0): {avail}\n"
            "Simple actions 1-5 need no arguments. Complex actions 6-7 require a "
            'JSON "data" field with {"x": column, "y": row} where x,y are in [0, 63].\n\n'
            f"Action history:\n{hist_txt}\n\n"
            f"Prior reasoning:\n{cot_txt}\n\n"
        )
        if known_rules:
            prompt += f"Known rules from world model:\n{known_rules}\n\n"
        if failed_attempts:
            prompt += f"Failed approaches (verifier feedback):\n{failed_attempts}\n\n"
        prompt += (
            "Reason step by step about how each action changes the grid and which "
            "action most increases progress toward winning (e.g. move a distinct "
            "marker onto a target, match a pattern, or clear an obstruction). "
            "Then choose the single best next action. "
            "Respond with ONLY a JSON object of the form "
            '{"action": <int>, "data": {"x": int, "y": int} or null, "reason": "..."}.'
        )
        return prompt

    def choose_action(
        self, frame: FrameData, history: list[str], available_actions: list[int]
    ) -> Optional[Action]:
        """Return a proposed :class:`Action`, or ``None`` on any failure."""
        if not self.available or self._client is None:
            return None

        self._total_steps += 1

        # Enrich prompt with world-model hypotheses and verifier feedback (Task 2)
        known_rules = ""
        failed_attempts = ""
        try:
            if self.world_model is not None:
                hyps = self.world_model.hypotheses()
                strong = [h for h in hyps if h.confidence > 0.3]
                if strong:
                    known_rules = "\n".join(
                        f"- {h.description} (confidence {h.confidence:.2f})"
                        for h in strong[:10]
                    )
            if self.memory is not None:
                failures = self.memory.search(
                    lambda r: r.get("type") == "event"
                    and r.get("event") == "verification_failure"
                )
                if failures:
                    recent = failures[-5:]
                    failed_attempts = "\n".join(
                        f"- ACTION{f.get('action')} at step {f.get('step')}: "
                        f"predicted {f.get('predicted_state')}, got {f.get('actual_state')}"
                        for f in recent
                    )
        except Exception:
            pass

        prompt = self._build_prompt(
            frame, history, available_actions, known_rules, failed_attempts
        )
        text = self._call(prompt)
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            code = int(obj["action"])
            ga = GameAction.from_int(code)
            data = obj.get("data") if ga.is_complex() else None
            if data is not None:
                data = {"x": int(data.get("x", 0)), "y": int(data.get("y", 0))}
            reason = obj.get("reason", "")
            if reason:
                self._cot_memory.append(reason)
                # Cap CoT memory (Task 1)
                if len(self._cot_memory) > 10:
                    self._cot_memory = self._cot_memory[-10:]
            return Action(ga, data=data, reasoning={"reason": reason})
        except Exception:
            return None


def create_llm_planner(
    model: Optional[str] = None,
    world_model: Optional[Any] = None,
    memory: Optional[Any] = None,
) -> LLMPlanner:
    return LLMPlanner(model=model, world_model=world_model, memory=memory)

# === Embedded: arc_agi3/agent.py ===
"""Agent orchestrator for ARC-AGI-3.

Coordinates perception, memory, world model, verifier and planner in an
observe -> plan -> act -> verify -> update loop, managing the per-level
action budget and falling back to exploration when the world model is stale
(see the rebuild plan's Phase 4 integration).
"""


import logging
import random
from typing import Callable, Optional


logger = logging.getLogger(__name__)

# Known simulated games and whether their primary mechanic is a complex
# (coordinate) action. Real ARC-AGI-3 environments would be detected at runtime
# from `available_actions` + observation; here we seed it per game id.
_COMPLEX_GAMES = {"sim_click"}


class ARCAgent:
    """Self-contained interactive ARC-AGI-3 agent."""

    def __init__(
        self,
        arcade: Arcade,
        game_id: str,
        seed: int = 0,
        budget_multiplier: float = 2.0,
        save_recording: bool = False,
        memory_dir: str = "memory",
        use_rust: bool = True,
        verbose: bool = False,
        budget_override: Optional[int] = None,
        use_llm: bool = True,
    ) -> None:
        self.arcade = arcade
        self.game_id = game_id
        self.seed = seed
        self.verbose = verbose

        self.env: Environment = arcade.make(
            game_id,
            seed=seed,
            save_recording=save_recording,
        )
        info = self.env.info()
        raw_base = getattr(info, "baseline_actions", 0) or 0
        if isinstance(raw_base, list):
            raw_base = max(raw_base) if raw_base else 0
        self.baseline_actions = max(raw_base, 1)
        n_levels = max(1, getattr(info, "n_levels", 1))
        computed = int(self.baseline_actions * budget_multiplier * n_levels) + 1
        # Real games often don't expose a baseline; give a sane floor so the
        # agent isn't starved of actions. An explicit override always wins.
        floor = 60 if raw_base == 0 else computed
        self.budget = budget_override if budget_override is not None else max(computed, floor)

        self.perception = Perception()
        self.memory = MemoryStore(game_id, seed, memory_dir=memory_dir)
        self.world_model = WorldModel(self.memory, self.perception, use_rust=use_rust)
        self.verifier = Verifier(self.world_model, self.memory)
        self.planner = Planner(self.world_model, self.perception, self.memory)
        self.llm = LLMPlanner(world_model=self.world_model, memory=self.memory) if use_llm else None
        self.history: list[str] = []
        self.allow_complex = game_id in _COMPLEX_GAMES
        self.goal_test: Callable[[FrameData], bool] = self._make_goal_test(game_id)
        # Tuning 2: penalize actions that recently produced no grid change.
        self._inert_actions: set[int] = set()
        # Tuning 4: detect when stuck in the same state.
        self._consecutive_same_state: int = 0
        self._last_state_sig: str = ""

    # ------------------------------------------------------------------
    # Goal definitions (per simulated game)
    # ------------------------------------------------------------------

    def _make_goal_test(self, game_id: str) -> Callable[[FrameData], bool]:
        if game_id == "sim_nav":
            def goal(frame: FrameData) -> bool:
                scene = self.perception.observe(frame)
                objs = {o.color: o for o in scene.objects}
                if 9 in objs and 4 in objs:
                    return tuple(objs[9].centroid) == tuple(objs[4].centroid)
                # Player consumed the target: target gone, player remains.
                if 9 in objs and 4 not in objs:
                    return True
                return False
            return goal
        if game_id == "sim_cm":
            def goal(frame: FrameData) -> bool:
                grid = frame.grid()
                if grid is None:
                    return False
                border = int(grid[0][0]) if grid.shape[1] > 0 else 0
                h, w = grid.shape
                if h > 3 and w > 3:
                    return int(grid[3][3]) == border
                return False
            return goal
        # Default: only a full win is a goal.
        return lambda frame: frame.state == GameState.WIN

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _systematic_bootstrap(self, frame: FrameData) -> tuple[FrameData, int]:
        """Systematic bootstrap: try each simple action once, then random probes.

        Returns (final_frame, steps_used).
        """
        steps_used = 0
        avail = frame.available_actions or []
        simple_actions = [a for a in avail if a in (1, 2, 3, 4, 5)]
        if not simple_actions:
            boot = self.planner.directed_explore(self.env, frame, n_probes=3)
            if boot:
                return boot[-1].state_after, len(boot)
            return frame, 0

        # Phase 1: systematic coverage of simple actions.
        changed_actions: list[int] = []
        for code in simple_actions:
            ga = GameAction.from_int(code)
            action = Action(ga)
            actual = self.env.step(action)
            steps_used += 1
            if actual is not None and actual.state != GameState.UNKNOWN:
                transition = Transition(frame, action, actual, frame.step)
                self.world_model.learn(transition)
                sig_before = self.perception.signature(frame)
                sig_after = self.perception.signature(actual)
                if sig_before != sig_after:
                    changed_actions.append(code)
                frame = actual
                if frame.state.is_terminal():
                    break

        # Phase 2: extra random probes (prioritize actions that changed state).
        budget = self.budget - steps_used
        extra_n = min(5, max(2, budget // 5))
        rng = random.Random(self.seed)
        for _ in range(extra_n):
            if changed_actions:
                code = rng.choice(changed_actions)
            else:
                code = rng.choice(simple_actions)
            ga = GameAction.from_int(code)
            action = Action(ga)
            actual = self.env.step(action)
            steps_used += 1
            if actual is not None and actual.state != GameState.UNKNOWN:
                transition = Transition(frame, action, actual, frame.step)
                self.world_model.learn(transition)
                frame = actual
                if frame.state.is_terminal():
                    break

        return frame, steps_used

    def run(self, max_steps: int = 300) -> ScorecardEntry:
        frame = self.env.reset()
        steps_used = 0
        start_levels = 0
        last_levels = 0

        # Bootstrap: gather initial transitions so the world model has rules.
        boot_n = min(10, max(5, self.budget // 5))
        if boot_n > 0:
            frame, boot_steps = self._systematic_bootstrap(frame)
            steps_used += boot_steps
            last_levels = frame.levels_completed
            self.planner.clear_visited()

        while steps_used < self.budget and not frame.state.is_terminal():
            if steps_used >= max_steps:
                break
            # Clear visited set and inert penalties on level transition.
            if frame.levels_completed > last_levels:
                self.planner.clear_visited()
                self._inert_actions.clear()
                self._consecutive_same_state = 0
                self._last_state_sig = ""
                last_levels = frame.levels_completed

            remaining = self.budget - steps_used
            action, why = None, "default"

            # Tuning 4: stuck-state detection — if same state for > 5 steps, randomize.
            current_sig = self.perception.signature(frame)
            if current_sig == self._last_state_sig and current_sig != "none":
                self._consecutive_same_state += 1
            else:
                self._consecutive_same_state = 0
            self._last_state_sig = current_sig
            stuck = self._consecutive_same_state > 5

            if stuck:
                avail = frame.available_actions or []
                simple = [a for a in avail if a in (1, 2, 3, 4, 5) and a not in self._inert_actions]
                if simple:
                    code = random.Random(self.seed + steps_used).choice(simple)
                    action = Action(GameAction.from_int(code))
                    why = "stuck_random"
                else:
                    simple_candidates = [a for a in avail if a in (1, 2, 3, 4, 5)]
                    if simple_candidates:
                        code = random.choice(simple_candidates)
                        action = Action(GameAction.from_int(code))
                    else:
                        action = Action(GameAction.ACTION1)
                    why = "stuck_default"

            if action is None and self.llm is not None and self.llm.available:
                action = self.llm.choose_action(frame, self.history, frame.available_actions)
                if action is not None:
                    why = "llm"
            if action is None:
                action, why = self.planner.next_action(
                    self.env, frame, self.goal_test, remaining, allow_complex=self.allow_complex
                )

            # Tuning 2: avoid recently inert actions.
            if action is not None and action.action.value in self._inert_actions:
                avail = frame.available_actions or []
                alternatives = [a for a in avail if a not in self._inert_actions and a in (1, 2, 3, 4, 5)]
                if alternatives:
                    action = Action(GameAction.from_int(random.choice(alternatives)))
                    why = "inert_avoidance"
            if self.verbose:
                logger.info("step %d: %s (%s) budget_left=%d", steps_used, action.action, why, remaining)

            executed_action = action
            executed_ok, correct, actual = self.verifier.verify_action(self.env, executed_action, frame)
            steps_used += 1

            if not executed_ok or actual is None:
                self.memory.record_event("step_failed", {"action": executed_action.action.value})
                # Retry with a different simple action up to 2 times (Fix 4).
                retries = 0
                while retries < 2 and (not executed_ok or actual is None):
                    avail = frame.available_actions or []
                    fallback_codes = [c for c in avail if c in (1, 2, 3, 4, 5)]
                    if not fallback_codes:
                        break
                    fallback_code = random.Random(self.seed + steps_used).choice(fallback_codes)
                    fallback_action = Action(GameAction.from_int(fallback_code))
                    executed_ok, correct, actual = self.verifier.verify_action(self.env, fallback_action, frame)
                    steps_used += 1
                    retries += 1
                    if executed_ok and actual is not None:
                        executed_action = fallback_action
                        break
                if not executed_ok or actual is None:
                    frame = actual if actual is not None else frame
                    continue

            # Learn the real outcome (skip when exploration already learned it).
            if why != "explore":
                transition = Transition(frame, executed_action, actual, frame.step)
                self.world_model.learn(transition)

            self.history.append(
                f"a={executed_action.action.value} -> {actual.state.value} L{actual.levels_completed}"
            )
            # Tuning 2: track inert actions (no grid change).
            try:
                sig_before = self.perception.signature(frame)
                sig_after = self.perception.signature(actual)
                if sig_before == sig_after:
                    self._inert_actions.add(executed_action.action.value)
                else:
                    self._inert_actions.discard(executed_action.action.value)
            except Exception:
                pass
            frame = actual

            if correct is False:
                # Prediction mismatch -> world model marked stale by verifier.
                self.memory.record_event("replan_after_failure", {})
                re = self.planner.directed_explore(self.env, frame, n_probes=2)
                if re:
                    frame = re[-1].state_after
                    steps_used += len(re)
                    # Reset stale flag after learning new transitions.
                    self.world_model.reset_stale()

            if frame.state == GameState.WIN:
                break

        won = frame.state == GameState.WIN
        level_scores = [1.0 for _ in range(frame.levels_completed)]
        total = competition_score(level_scores)
        entry = ScorecardEntry(
            game_id=self.game_id,
            seed=self.seed,
            level_scores=level_scores,
            total_score=total,
            steps_used=steps_used,
            budget=self.budget,
            won=won,
        )
        self.arcade.record_entry(entry)
        self.env.close()
        return entry


def create_agent(arcade: Arcade, game_id: str, seed: int = 0, **kw) -> ARCAgent:
    return ARCAgent(arcade, game_id, seed=seed, **kw)

# ---------------------------------------------------------------------------
# Main run: COMPETITION mode
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)s | %(message)s')

arcade = Arcade(
    operation_mode=OperationMode.COMPETITION,
    environments_dir='environment_files',
)

infos = arcade.get_environments()
game_ids = [info.game_id for info in infos if not info.is_simulated]
print(f"Discovered {len(game_ids)} non-simulated environments: {game_ids}")

results = []
for gid in game_ids:
    try:
        agent = ARCAgent(
            arcade,
            gid,
            seed=SEED,
            budget_multiplier=BUDGET_MULTIPLIER,
            use_rust=USE_RUST,
            use_llm=USE_LLM,
            verbose=False,
        )
        entry = agent.run(max_steps=MAX_STEPS)
        rec = {
            'game_id': gid,
            'won': entry.won,
            'levels': len(entry.level_scores),
            'score': round(entry.total_score, 4),
            'steps': entry.steps_used,
            'budget': entry.budget,
        }
        results.append(rec)
        print(f"[{gid}] won={rec['won']} levels={rec['levels']} "
              f"score={rec['score']} steps={rec['steps']}/{rec['budget']}")
    except Exception as exc:
        print(f"[{gid}] ERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        results.append({'game_id': gid, 'won': False, 'levels': 0,
                        'score': 0.0, 'steps': 0, 'budget': 0, 'error': str(exc)})

# --- Summary
n = len(results)
wins = sum(1 for r in results if r['won'])
total_score = sum(r['score'] for r in results)
avg_steps = sum(r['steps'] for r in results) / max(n, 1)
print("\n=== Summary ===")
print(f"games={n} wins={wins} total_score={round(total_score, 4)} avg_steps={round(avg_steps, 2)}")
print(f"USE_LLM={USE_LLM} USE_RUST={USE_RUST} seed={SEED} budget_multiplier={BUDGET_MULTIPLIER}")
