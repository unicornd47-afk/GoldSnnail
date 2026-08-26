"""Memory module for the ARC-AGI-3 interactive-agent framework.

Provides :class:`MemoryStore`, an append-only, searchable JSONL store of
transitions, hypotheses, and events keyed by ``(game_id, seed)``.

Only the standard library and :mod:`numpy` are used, and the module imports
cleanly even when the real ``arc_agi`` / ``arcengine`` packages are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Optional

import numpy as np

from arc_agi3.types import (
    Action,
    FrameData,
    GameAction,
    GameState,
    Hypothesis,
    Transition,
)

try:  # Perception lives in arc_agi3.perception; tolerate its absence.
    from arc_agi3.perception import Perception
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
        return {
            "game_id": frame.game_id,
            "state": frame.state.value,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "available_actions": list(frame.available_actions),
            "full_reset": frame.full_reset,
            "guid": frame.guid,
            "action_input": frame.action_input,
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
