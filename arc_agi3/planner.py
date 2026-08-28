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

from __future__ import annotations

import os
import random
import sys
from typing import Callable, Optional, Tuple

# Allow importing the repo-root bandit if present.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arc_agi3.env import Environment
from arc_agi3.memory import MemoryStore
from arc_agi3.perception import Perception
from arc_agi3.types import (
    Action,
    FrameData,
    GameAction,
    Plan,
    PlanStep,
    SIMPLE_ACTIONS,
    Transition,
)
from arc_agi3.world_model import WorldModel

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
                if nsig in visited:
                    continue
                visited.add(nsig)
                parent[nsig] = (sig, action, frame)
                self._visited.add(key)
                self._state_outcomes[key] = nsig
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
