"""Optional LLM-based action proposer for ARC-AGI-3.

Leading ARC-AGI-3 solutions drive an LLM to reason about each frame and choose
the next action. This module provides that capability **optionally**: if an API
key is present it proposes actions; otherwise ``available`` is ``False`` and the
agent transparently falls back to the rule-based :class:`Planner`.

Configuration (environment variables):
    OPENROUTER_API_KEY / OPENAI_API_KEY / ARC_LLM_API_KEY : API key (enables).
    ARC_LLM_BASE_URL / OPENROUTER_BASE_URL / OPENAI_BASE_URL : OpenAI-compatible base URL.
    ARC_LLM_MODEL / OPENROUTER_MODEL                            : model id (default ``z-ai/glm-5.2``).

OpenRouter is OpenAI-compatible: set ``OPENROUTER_API_KEY`` and use the slug
``z-ai/glm-5.2`` with base ``https://openrouter.ai/api/v1``.

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
    """Proposes the next action using an OpenAI-compatible API (e.g. OpenRouter)."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ARC_LLM_API_KEY")
            or ""
        )
        self.base_url = (
            os.environ.get("ARC_LLM_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        self.model = (
            model
            or os.environ.get("ARC_LLM_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or "z-ai/glm-5.2"
        )
        self.available = bool(self.api_key)
        self._client: Any = None
        if self.available:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self.available = False
                self._client = None

    def _call(self, prompt: str) -> Optional[str]:
        """Call the model; prefer the Responses API, fall back to Chat Completions."""
        if self._client is None:
            return None
        try:
            resp = self._client.responses.create(model=self.model, input=prompt, timeout=30)
            text = getattr(resp, "output_text", None)
            if text:
                return text
        except Exception:
            pass
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30,
            )
            return (resp.choices[0].message.content or "") if resp.choices else ""
        except Exception:
            return None

    def choose_action(
        self, frame: FrameData, history: list[str], available_actions: list[int]
    ) -> Optional[Action]:
        """Return a proposed :class:`Action`, or ``None`` on any failure."""
        if not self.available or self._client is None:
            return None
        grid_txt = render_frame(frame)
        avail = ", ".join(str(a) for a in available_actions if a not in (-1, 0))
        hist_txt = "\n".join(f"{i}: {h}" for i, h in enumerate(history[-6:])) or "(none)"
        # Structured scene gives the model far more signal than raw ints.
        scene_txt = "n/a"
        try:
            from arc_agi3.perception import Perception

            sc = Perception().observe(frame)
            objs = [
                f"color{o.color}@({int(o.centroid[0])},{int(o.centroid[1])}) area{o.area}"
                for o in sc.objects
                if o.color != 0
            ]
            scene_txt = "; ".join(objs) if objs else "none"
        except Exception:
            pass
        prompt = (
            "You are playing an interactive ARC-AGI-3 grid game. The grid below "
            "shows the current state as rows of integers (0 = background).\n\n"
            f"{grid_txt}\n\n"
            f"Distinct non-background objects: {scene_txt}\n"
            f"Available action codes (ignore -1 and 0): {avail}\n"
            "Simple actions 1-5 need no arguments. Complex actions 6-7 require a "
            'JSON "data" field with {"x": column, "y": row} where x,y are in [0, 63].\n'
            f"Recent history (action -> outcome):\n{hist_txt}\n\n"
            "Reason step by step about how each action changes the grid and which "
            "action most increases progress toward winning (e.g. move a distinct "
            "marker onto a target, match a pattern, or clear an obstruction). "
            "Then choose the single best next action. "
            "Respond with ONLY a JSON object of the form "
            '{"action": <int>, "data": {"x": int, "y": int} or null, "reason": "..."}.'
        )
        text = self._call(prompt)
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
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
