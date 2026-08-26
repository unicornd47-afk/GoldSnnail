"""Verification module for the ARC-AGI-3 interactive-agent framework.

Implements the predict-act-compare verification loop. Before committing to
planned actions, the agent predicts the outcome via the world model, executes
the action in the real environment, and compares. A mismatch marks the world
model stale and records a contradiction.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    root = r"C:\Users\Student\Documents\Goldsnnail\Goldsnnail"
    if root not in sys.path:
        sys.path.insert(0, root)

from typing import List, Optional, Tuple

import numpy as np

from arc_agi3.types import Action, FrameData, GameAction, GameState, Transition


# ---------------------------------------------------------------------------
# Frame comparison helpers
# ---------------------------------------------------------------------------


def frames_match(
    a: Optional[FrameData],
    b: Optional[FrameData],
    grid_tolerance: float = 1.0,
) -> bool:
    """Return True if two frames are considered equal within *grid_tolerance*.

    Rules (in order):
    1. ``None`` inputs are always mismatched.
    2. ``state`` and ``levels_completed`` must match exactly.
    3. Grid overlap fraction must meet or exceed ``grid_tolerance``.
       - If both grids are ``None`` the scalar checks suffice (return True).
       - If shapes differ, return False.
       - Otherwise require ``float((ga == gb).sum()) / ga.size >= grid_tolerance``.
    """
    if a is None or b is None:
        return False
    if a.state != b.state:
        return False
    if a.levels_completed != b.levels_completed:
        return False
    ga, gb = a.grid(), b.grid()
    if ga is None and gb is None:
        return True
    if ga is None or gb is None:
        return False
    if ga.shape != gb.shape:
        return False
    overlap = float((ga == gb).sum()) / ga.size
    return overlap >= grid_tolerance


def frames_similarity(a: Optional[FrameData], b: Optional[FrameData]) -> float:
    """Return a similarity score in [0, 1] combining state and grid overlap.

    - State match contributes 0.5 (1.0 if equal, 0.0 otherwise).
    - Grid overlap fraction contributes 0.5 (guards against ``None`` grids).
    """
    if a is None or b is None:
        return 0.0
    state_score = 1.0 if a.state == b.state else 0.0
    ga, gb = a.grid(), b.grid()
    if ga is None and gb is None:
        grid_score = 1.0
    elif ga is None or gb is None:
        grid_score = 0.0
    elif ga.shape != gb.shape:
        grid_score = 0.0
    else:
        grid_score = float((ga == gb).sum()) / ga.size
    return 0.5 * state_score + 0.5 * grid_score


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class Verifier:
    """Runs the predict-act-compare verification loop for an agent.

    Each planned action is first predicted by the world model, then executed
    in the real environment. Mismatches mark the model stale and record a
    contradiction in memory.
    """

    def __init__(self, world_model: "WorldModel", memory: "MemoryStore") -> None:
        self.world_model = world_model
        self.memory = memory

    def verify_action(
        self,
        env: "Environment",
        action: Action,
        current_frame: FrameData,
    ) -> Tuple[bool, Optional[bool], FrameData]:
        """Verify a single action by comparing prediction to execution.

        Returns ``(executed_ok, correct, actual)`` where:
        - ``executed_ok`` is True when the environment returned a non-UNKNOWN frame.
        - ``correct`` is True/False when a prediction was available, else None.
        - ``actual`` is the frame returned by ``env.step(action)``.
        """
        prediction = self.world_model.predict(action, current_frame)
        actual: FrameData = env.step(action)
        executed_ok = actual is not None and actual.state != GameState.UNKNOWN
        correct: Optional[bool] = None if prediction is None else frames_match(prediction, actual)

        if prediction is not None and correct is False:
            self.world_model.mark_stale()
            self.memory.record_event(
                "verification_failure",
                {
                    "action": action.action.value,
                    "predicted_state": prediction.state.value,
                    "actual_state": actual.state.value,
                    "step": current_frame.step,
                },
            )
            hyp = self.memory.best_hypothesis(action.action)
            if hyp is not None:
                self.memory.contradict(hyp.id, Transition(current_frame, action, actual, current_frame.step))
        elif prediction is not None and correct is True:
            self.memory.record_event(
                "verification_success",
                {
                    "action": action.action.value,
                    "step": current_frame.step,
                },
            )

        return executed_ok, correct, actual

    def verify_plan(
        self,
        env: "Environment",
        plan: object,
        current_frame: FrameData,
    ) -> List[Tuple[bool, Optional[bool], FrameData]]:
        """Verify every step in a plan, breaking early on terminal or mismatch.

        Iterates over ``plan.steps``, feeding each action through
        :meth:`verify_action` and advancing ``current_frame`` to the actual
        outcome.  Stops early if the environment reaches a terminal state or
        a prediction mismatch occurs.
        """
        results: List[Tuple[bool, Optional[bool], FrameData]] = []
        for step in plan.steps:
            executed_ok, correct, actual = self.verify_action(env, step.action, current_frame)
            results.append((executed_ok, correct, actual))
            current_frame = actual
            if actual.state.is_terminal() or correct is False:
                break
        return results


def create_verifier(world_model: "WorldModel", memory: "MemoryStore") -> Verifier:
    """Factory returning a configured :class:`Verifier`."""
    return Verifier(world_model=world_model, memory=memory)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tempfile
    import os
    import sys

    root = r"C:\Users\Student\Documents\Goldsnnail\Goldsnnail"
    if root not in sys.path:
        sys.path.insert(0, root)

    from arc_agi3.memory import MemoryStore
    from arc_agi3.types import (
        Action,
        FrameData,
        GameAction,
        GameState,
        Hypothesis,
        Plan,
        PlanStep,
        Transition,
    )

    # -- 1. frames_match / frames_similarity with identical frames -------------
    grid_identical = [[0] * 8 for _ in range(8)]
    grid_identical[4][2] = 9
    grid_identical[4][5] = 4

    f1 = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_identical, step=0)
    f2 = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_identical, step=1)

    assert frames_match(f1, f2), "identical frames should match"
    sim = frames_similarity(f1, f2)
    assert abs(sim - 1.0) < 1e-9, f"expected similarity ~1.0, got {sim}"

    # -- 2. One-cell difference, default tolerance fails, relaxed passes ----------
    grid_diff = [row[:] for row in grid_identical]
    grid_diff[4][2] = 1  # change player color

    f3 = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_diff, step=0)

    assert not frames_match(f1, f3, grid_tolerance=1.0), "1/64 cells differ -> should not match at 1.0"
    assert frames_match(f1, f3, grid_tolerance=0.9), "63/64 match should pass at 0.9"
    sim2 = frames_similarity(f1, f3)
    assert 0.0 < sim2 < 1.0, f"partial similarity expected, got {sim2}"

    # -- 3. verify_action with mock environment / world model / memory -----------
    class MockEnv:
        def __init__(self, response: FrameData) -> None:
            self._response = response

        def step(self, action: Action) -> FrameData:
            return self._response

        def reset(self) -> FrameData:
            return self._response

        def action_space(self) -> list:
            return [1, 2, 3, 4, 5, 6, 7, 0]

        def close(self) -> None:
            pass

    class MockWorldModel:
        def __init__(self, prediction: Optional[FrameData]) -> None:
            self._prediction = prediction
            self.stale_called = False

        def predict(self, action: Action, frame: FrameData) -> Optional[FrameData]:
            return self._prediction

        def mark_stale(self) -> None:
            self.stale_called = True

        def confidence(self, action: Action, frame: FrameData) -> float:
            return 0.5

        def hypotheses(self) -> list:
            return []

    with tempfile.TemporaryDirectory() as tmpdir:
        mem = MemoryStore(game_id="test", seed=0, memory_dir=tmpdir)
        actual_frame = FrameData(
            game_id="test", state=GameState.PLAYING, frame=grid_identical, step=1
        )

        # Case A: correct prediction
        wm_ok = MockWorldModel(prediction=actual_frame)
        env_ok = MockEnv(response=actual_frame)
        verifier_ok = create_verifier(wm_ok, mem)
        exec_ok, correct, actual = verifier_ok.verify_action(env_ok, Action(GameAction.ACTION1), f1)
        assert exec_ok is True, "execution should succeed"
        assert correct is True, "prediction should match actual"
        assert wm_ok.stale_called is False, "mark_stale should not fire on match"

        events = [r for r in mem._records if r.get("type") == "event"]
        assert any(e.get("event") == "verification_success" for e in events), \
            "verification_success event should be recorded"

        # Case B: incorrect prediction
        wrong_pred = FrameData(
            game_id="test", state=GameState.PLAYING, frame=grid_diff, step=1
        )
        wm_bad = MockWorldModel(prediction=wrong_pred)
        env_bad = MockEnv(response=actual_frame)
        verifier_bad = create_verifier(wm_bad, mem)
        exec_ok2, correct2, actual2 = verifier_bad.verify_action(env_bad, Action(GameAction.ACTION1), f1)
        assert exec_ok2 is True, "execution should succeed"
        assert correct2 is False, "prediction should mismatch actual"
        assert wm_bad.stale_called is True, "mark_stale should fire on mismatch"

        # A hypothesis should have been contradicted (or at least the code ran without error)
        events2 = [r for r in mem._records if r.get("type") == "event"]
        assert any(e.get("event") == "verification_failure" for e in events2), \
            "verification_failure event should be recorded"

        # Case C: verify_plan breaks early on mismatch
        plan = Plan(
            steps=[
                PlanStep(action=Action(GameAction.ACTION1)),
                PlanStep(action=Action(GameAction.ACTION2)),
            ],
        )
        env_plan = MockEnv(response=actual_frame)
        results = verifier_bad.verify_plan(env_plan, plan, f1)
        assert len(results) == 1, "plan should break after first mismatch"
        assert results[0][1] is False

        mem.close()

    print("ok")
