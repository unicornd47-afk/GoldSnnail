"""Agent orchestrator for ARC-AGI-3.

Coordinates perception, memory, world model, verifier and planner in an
observe -> plan -> act -> verify -> update loop, managing the per-level
action budget and falling back to exploration when the world model is stale
(see the rebuild plan's Phase 4 integration).
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

from arc_agi3.env import Arcade, Environment, ScorecardEntry, competition_score
from arc_agi3.memory import MemoryStore
from arc_agi3.perception import Perception
from arc_agi3.planner import Planner
from arc_agi3.types import (
    Action,
    FrameData,
    GameAction,
    GameState,
    OperationMode,
    ScorecardEntry as _ScorecardEntry,
    Transition,
)
from arc_agi3.verifier import Verifier
from arc_agi3.world_model import WorldModel
from arc_agi3.llm_planner import LLMPlanner

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
                    action = Action(GameAction.from_int(random.choice(avail) if avail else 1))
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
                alternatives = [a for a in avail if a not in self._inert_actions and a not in (0, -1)]
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
