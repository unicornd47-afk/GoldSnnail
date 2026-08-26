"""Simulated ARC-AGI-3 environments.

Each :class:`SimGame` obeys the same protocol as the real toolkit so that
downstream agent code is identical for simulated and real environments.

All games are deterministic given a seed (``random.Random(seed)``) and
solvable by construction.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from arc_agi3.types import (
    ALL_ACTIONS,
    Action,
    EnvironmentInfo,
    FrameData,
    GameAction,
    GameState,
)


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
