"""Optional LLM-based action proposer for ARC-AGI-3.

Leading ARC-AGI-3 solutions drive an LLM to reason about each frame and choose
the next action. This module provides that capability **optionally**: if an API
key is present it proposes actions; otherwise ``available`` is ``False`` and the
agent transparently falls back to the rule-based :class:`Planner`.

Configuration (environment variables):
    OPENAI_API_KEY / ARC_LLM_API_KEY : API key (required to enable).
    ARC_LLM_MODEL                   : model id (default ``gpt-4o-mini``).
    ARC_LLM_BASE_URL                : optional OpenAI-compatible base URL.

The ``openai`` package is imported lazily, so this module never breaks the
framework when the dependency or key is absent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from arc_agi3.types import Action, FrameData, GameAction


def render_frame(frame: FrameData) -> str:
    """Render a frame's grid as readable ASCII for the LLM prompt."""
    grid = frame.grid()
    if grid is None:
        return "(no grid / non-visual state)"
    rows = [" ".join(str(int(c)) for c in row) for row in grid.tolist()]
    return "\n".join(rows)


class LLMPlanner:
    """Proposes the next action using an OpenAI-compatible Responses API."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ARC_LLM_API_KEY") or ""
        self.model = model or os.environ.get("ARC_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = os.environ.get("ARC_LLM_BASE_URL") or None
        self.available = bool(self.api_key)
        self._client: Any = None
        if self.available:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self.available = False
                self._client = None

    def choose_action(
        self, frame: FrameData, history: list[str], available_actions: list[int]
    ) -> Optional[Action]:
        """Return a proposed :class:`Action`, or ``None`` on any failure."""
        if not self.available or self._client is None:
            return None
        grid_txt = render_frame(frame)
        avail = ", ".join(str(a) for a in available_actions if a not in (-1, 0))
        hist_txt = "\n".join(f"{i}: {h}" for i, h in enumerate(history[-6:])) or "(none)"
        prompt = (
            "You are playing an interactive ARC-AGI-3 grid game. The grid below "
            "shows the current state as rows of integers (0 = background).\n\n"
            f"{grid_txt}\n\n"
            f"Available action codes (ignore -1 and 0): {avail}\n"
            "Simple actions 1-5 need no arguments. Complex actions 6-7 require a "
            'JSON "data" field with {"x": column, "y": row} where x,y are in [0, 63].\n'
            f"Recent history (action -> outcome):\n{hist_txt}\n\n"
            "Choose the single best next action to make progress toward winning. "
            "Respond with ONLY a JSON object of the form "
            '{"action": <int>, "data": {"x": int, "y": int} or null, "reason": "..."}.'
        )
        try:
            resp = self._client.responses.create(model=self.model, input=prompt, timeout=30)
            text = getattr(resp, "output_text", None)
            if text is None:
                text = str(resp)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            obj = json.loads(m.group(0))
            code = int(obj["action"])
            ga = GameAction.from_int(code)
            data = obj.get("data") if ga.is_complex() else None
            if data is not None:
                data = {"x": int(data.get("x", 0)), "y": int(data.get("y", 0))}
            return Action(ga, data=data, reasoning={"reason": obj.get("reason")})
        except Exception:
            return None


def create_llm_planner(model: Optional[str] = None) -> LLMPlanner:
    return LLMPlanner(model=model)
