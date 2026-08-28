"""World-model / rule-learner module for the ARC-AGI-3 interactive-agent framework.

Learns parametric transition rules from observed (state, action) -> state
transitions so the agent can predict and simulate future states for planning.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    root = r"C:\Users\Student\Documents\Goldsnnail\Goldsnnail"
    arc_agi3_dir = os.path.join(root, "arc_agi3")
    if sys.path and os.path.normpath(sys.path[0]) == os.path.normpath(arc_agi3_dir):
        sys.path = sys.path[1:]
    if root not in sys.path:
        sys.path.insert(0, root)

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from arc_agi3.memory import MemoryStore
from arc_agi3.perception import Perception
from arc_agi3.rust_bridge import RUST_AVAILABLE, rust_apply_program, solve_grid_transform
from arc_agi3.types import Action, FrameData, GameAction, Hypothesis, Transition


def create_world_model(
    memory: MemoryStore,
    perception: Perception,
    use_rust: bool = True,
) -> WorldModel:
    """Convenience factory returning a configured :class:`WorldModel`."""
    return WorldModel(memory=memory, perception=perception, use_rust=use_rust)


class WorldModel:
    """Learns and applies transition rules from observed frame transitions."""

    def __init__(
        self,
        memory: MemoryStore,
        perception: Perception,
        use_rust: bool = True,
    ) -> None:
        self.memory = memory
        self.perception = perception
        self.use_rust = use_rust
        self._stale = False
        self.motion_rules: dict[tuple[int, int], tuple[int, int]] = {}
        self.motion_counts: dict[tuple[int, int], int] = {}
        self.paint_rules: dict[tuple[int, tuple[int, int]], int] = {}
        self.paint_counts: dict[tuple[int, tuple[int, int]], int] = {}
        self._hypotheses: list[Hypothesis] = []
        self._rust_confidence: float = 0.0
        # Minimum supporting transitions before a hypothesis is created / surfaced.
        self.min_evidence = int(os.environ.get("ARC_WM_MIN_EVIDENCE", "2"))

    def learn(self, t: Transition) -> None:
        """Ingest a transition and update parametric rules and hypotheses."""
        self.memory.record_transition(t)
        before_grid = t.state_before.grid()
        after_grid = t.state_after.grid()
        if before_grid is None or after_grid is None:
            return

        d = self.perception.diff(t.state_before, t.state_after)
        action_code = t.action.action.value

        # Detect contradictions with existing motion/paint rules (Task 4).
        for m in d.get("moved_objects", []):
            key = (action_code, int(m["color"]))
            dr, dc = int(m["dr"]), int(m["dc"])
            if key in self.motion_rules:
                old_dr, old_dc = self.motion_rules[key]
                if (old_dr, old_dc) != (dr, dc):
                    # Contradiction: decay conflicting motion hypotheses.
                    self._prune_contradictions("motion", key, action_code)
            self.motion_rules[key] = (dr, dc)
            self.motion_counts[key] = self.motion_counts.get(key, 0) + 1
            count = self.motion_counts[key]
            desc = f"ACTION{action_code} translates color {m['color']} by ({dr},{dc})"
            self._upsert_hypothesis(
                hyp_type="motion",
                key=key,
                action=t.action.action,
                description=desc,
                confidence=min(1.0, 0.2 + count * 0.3),
                supporting_transitions=count,
            )

        for c in d.get("color_changes", []):
            key = (action_code, (int(c["r"]), int(c["c"])))
            new_color = int(c["new"])
            if key in self.paint_rules and self.paint_rules[key] != new_color:
                self._prune_contradictions("paint", key, action_code)
            self.paint_rules[key] = new_color
            self.paint_counts[key] = self.paint_counts.get(key, 0) + 1
            count = self.paint_counts[key]
            desc = f"ACTION{action_code} sets cell ({c['r']},{c['c']}) to color {c['new']}"
            self._upsert_hypothesis(
                hyp_type="paint",
                key=key,
                action=t.action.action,
                description=desc,
                confidence=min(1.0, 0.2 + count * 0.3),
                supporting_transitions=count,
            )

        if t.state_after.levels_completed > t.state_before.levels_completed:
            self.memory.record_event(
                "level_progress",
                {
                    "action": action_code,
                    "levels_before": t.state_before.levels_completed,
                    "levels_after": t.state_after.levels_completed,
                },
            )

    def _prune_contradictions(
        self, hyp_type: str, key: tuple, action_code: int
    ) -> None:
        """Decay confidence of the specific contradicted hypothesis only (Fix 2)."""
        action = GameAction.from_int(action_code)
        hyp_id = f"{hyp_type}_{action.value}_{hash(key) & 0xFFFFFFFF:08x}"
        target = None
        for h in self._hypotheses:
            if h.id == hyp_id:
                target = h
                break
        if target is None:
            return
        target.confidence = max(0.0, target.confidence * 0.5)
        target.contradicting_transitions += 1
        self.memory.record_event(
            "contradiction_pruned",
            {
                "hypothesis_id": hyp_id,
                "new_confidence": target.confidence,
            },
        )

    def reset_stale(self) -> None:
        """Clear the stale flag after learning new transitions."""
        self._stale = False

    def _upsert_hypothesis(
        self,
        hyp_type: str,
        key: tuple,
        action: GameAction,
        description: str,
        confidence: float,
        supporting_transitions: int,
    ) -> None:
        hyp_id = f"{hyp_type}_{action.value}_{hash(key) & 0xFFFFFFFF:08x}"
        for h in self._hypotheses:
            if h.id == hyp_id:
                h.confidence = confidence
                h.supporting_transitions = supporting_transitions
                return
        # Only create a hypothesis when we have minimum evidence (Fix 1).
        if supporting_transitions < self.min_evidence:
            return
        h = Hypothesis(
            id=hyp_id,
            action=action,
            description=description,
            program=None,
            confidence=confidence,
            supporting_transitions=supporting_transitions,
            contradicting_transitions=0,
            created_step=0,
        )
        self._hypotheses.append(h)
        self.memory.record_hypothesis(h)

    def predict(self, action: Action, current_frame: FrameData) -> Optional[FrameData]:
        """Predict the next frame given an action applied to the current frame."""
        current_grid = current_frame.grid()
        if current_grid is None:
            return None

        new_grid = current_grid.copy()
        a = action.action.value
        applied = False

        for (akey, color), (dr, dc) in list(self.motion_rules.items()):
            if akey != a:
                continue
            H, W = new_grid.shape
            pixels = np.argwhere(new_grid == color)
            if pixels.size == 0:
                continue
            result = np.zeros_like(new_grid)
            mask = new_grid != color
            result[mask] = new_grid[mask]
            for r, c in pixels:
                nr, nc = int(r) + dr, int(c) + dc
                # Clamp to grid bounds (matches env semantics, e.g. nav walls).
                nr = max(0, min(H - 1, nr))
                nc = max(0, min(W - 1, nc))
                result[nr, nc] = color
            new_grid = result
            applied = True

        if not applied:
            for (akey, pos), color in list(self.paint_rules.items()):
                if akey != a:
                    continue
                r, c = pos
                new_grid[r, c] = color
                applied = True

        if not applied:
            self._rust_confidence = 0.0
            templates = self.memory.similar_transitions(current_frame, action.action, k=1)
            if templates and RUST_AVAILABLE and self.use_rust:
                tt = templates[0]
                if tt.state_before.frame is not None and tt.state_after.frame is not None:
                    prog = solve_grid_transform(
                        tt.state_before.frame, tt.state_after.frame, timeout=10
                    )
                    if prog:
                        res = rust_apply_program(current_frame.frame, prog)
                        if res is not None:
                            new_grid = np.array(res, dtype=int)
                            applied = True
                            self._rust_confidence = 0.4

        if not applied:
            return None

        return replace(
            current_frame,
            frame=new_grid.tolist(),
            step=current_frame.step + 1,
        )

    def confidence(self, action: Action, current_frame: FrameData) -> float:
        a = action.action.value
        for (akey, _color), count in self.motion_counts.items():
            if akey == a:
                return min(1.0, 0.2 + count * 0.3)
        for (akey, _pos), count in self.paint_counts.items():
            if akey == a:
                return min(1.0, 0.2 + count * 0.3)
        return self._rust_confidence

    def mark_stale(self) -> None:
        self._stale = True

    def is_stale(self) -> bool:
        return self._stale

    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hypotheses)


if __name__ == "__main__":
    import tempfile
    import sys

    root = r"C:\Users\Student\Documents\Goldsnnail\Goldsnnail"
    if root not in sys.path:
        sys.path.insert(0, root)

    from arc_agi3.memory import MemoryStore
    from arc_agi3.perception import Perception
    from arc_agi3.types import Action, FrameData, GameAction, GameState, Transition

    with tempfile.TemporaryDirectory() as tmpdir:
        mem = MemoryStore(game_id="test", seed=0, memory_dir=tmpdir)
        perception = Perception()
        wm = WorldModel(memory=mem, perception=perception, use_rust=False)

        # Motion rule test
        grid_a = [[0] * 8 for _ in range(8)]
        grid_a[4][2] = 9
        grid_a[4][5] = 4
        fa = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_a, step=0)

        grid_b = [[0] * 8 for _ in range(8)]
        grid_b[3][2] = 9
        grid_b[4][5] = 4
        fb = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_b, step=1)

        t = Transition(
            state_before=fa, action=Action(GameAction.ACTION1), state_after=fb, step=0
        )
        wm.learn(t)

        assert (1, 9) in wm.motion_rules
        assert wm.motion_rules[(1, 9)] == (-1, 0)

        grid_c = [[0] * 8 for _ in range(8)]
        grid_c[6][3] = 9
        grid_c[4][5] = 4
        fc = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_c, step=0)

        pred = wm.predict(Action(GameAction.ACTION1), fc)
        assert pred is not None
        assert pred.frame[5][3] == 9
        assert pred.frame[4][5] == 4

        # Paint rule test
        grid_d = [[3] * 5 for _ in range(5)]
        grid_d[2][2] = 2
        fd = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_d, step=0)

        grid_e = [[3] * 5 for _ in range(5)]
        grid_e[2][2] = 3
        fe = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_e, step=1)

        t2 = Transition(
            state_before=fd, action=Action(GameAction.ACTION3), state_after=fe, step=1
        )
        wm.learn(t2)

        assert (3, (2, 2)) in wm.paint_rules
        assert wm.paint_rules[(3, (2, 2))] == 3

        grid_f = [[0] * 5 for _ in range(5)]
        grid_f[2][2] = 1
        ff = FrameData(game_id="test", state=GameState.PLAYING, frame=grid_f, step=0)

        pred2 = wm.predict(Action(GameAction.ACTION3), ff)
        assert pred2 is not None
        assert pred2.frame[2][2] == 3

        # Stale test
        assert not wm.is_stale()
        wm.mark_stale()
        assert wm.is_stale()

        mem.close()
        print("ok")
