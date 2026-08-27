"""Toolkit adapter bridging the real ``arc_agi`` package and our simulations.

Provides :class:`Environment` (wraps a single game backend) and :class:`Arcade`
(the main entry point for creating environments).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from arc_agi3.sim_env import SIM_GAMES, SimGame, get_sim_info, list_sim_games
from arc_agi3.types import (
    ALL_ACTIONS,
    Action,
    EnvironmentInfo,
    FrameData,
    GameAction,
    GameState,
    OperationMode,
    ScorecardEntry,
)

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
                raw = backend.step(action)
                return _from_toolkit_frame(raw) if raw is not None else _empty_frame(self.game_id)
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
    ) -> None:
        self.operation_mode = operation_mode
        self.environments_dir = environments_dir
        self.recordings_dir = recordings_dir
        self.logger = logger or logging.getLogger(__name__)
        self._entries: list[ScorecardEntry] = []
        self._real_arcade: Any = None
        if _REAL_AVAILABLE:
            try:
                self._real_arcade = _RealArcade(
                    operation_mode=_RealOM(operation_mode.value),
                    environments_dir=environments_dir,
                    recordings_dir=recordings_dir,
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
            raw = self._real.step(_to_toolkit_action(Action(GameAction.RESET)))
        return _from_toolkit_frame(raw) if raw is not None else _empty_frame(getattr(self._real, "game_id", "unknown"))

    def step(self, action: Action) -> FrameData:
        tk_action = _to_toolkit_action(action)
        raw = self._real.step(tk_action)
        if raw is None:
            return _empty_frame(getattr(self._real, "game_id", "unknown"))
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
    if not _REAL_AVAILABLE:
        raise RuntimeError("Real toolkit not installed.")
    from arcengine import GameAction as _TKAction  # type: ignore
    tk = _TKAction[action.action.name]
    if action.data is not None:
        tk = tk.set_data(action.data)
    return tk


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
