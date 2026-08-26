"""Core types for the ARC-AGI-3 interactive-agent framework.

Defines the internal, toolkit-agnostic data model used across all agent modules.
All imports must work without the real ``arc_agi`` / ``arcengine`` packages installed.
"""

from __future__ import annotations

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

    NOT_PLAYED = "not_played"
    PLAYING = "playing"
    WIN = "win"
    GAME_OVER = "game_over"
    LOSE = "lose"
    UNKNOWN = "unknown"

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
