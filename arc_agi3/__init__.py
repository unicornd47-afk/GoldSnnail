"""ARC-AGI-3 interactive-agent framework.

Provides offline-simulated environments that mirror the real ``arc_agi`` toolkit
API, plus types, scoring helpers, and a Rust bridge for grid-transform solving.
"""

from arc_agi3.env import (
    Arcade,
    Environment,
    competition_score,
    get_default_arcade,
    simple_score,
)
from arc_agi3.rust_bridge import RUST_AVAILABLE, solve_grid_transform
from arc_agi3.sim_env import SIM_GAMES, get_sim_info, list_sim_games
from arc_agi3.types import (
    Action,
    EnvironmentInfo,
    FrameData,
    GameAction,
    GameState,
    Hypothesis,
    OperationMode,
    Plan,
    PlanStep,
    SceneGraph,
    SceneObject,
    ScorecardEntry,
    Transition,
)

__all__ = [
    "GameAction",
    "GameState",
    "OperationMode",
    "Action",
    "FrameData",
    "EnvironmentInfo",
    "ScorecardEntry",
    "Transition",
    "SceneGraph",
    "SceneObject",
    "Hypothesis",
    "PlanStep",
    "Plan",
    "Arcade",
    "Environment",
    "get_default_arcade",
    "competition_score",
    "simple_score",
    "RUST_AVAILABLE",
    "solve_grid_transform",
    "SIM_GAMES",
    "get_sim_info",
    "list_sim_games",
]
