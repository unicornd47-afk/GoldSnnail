"""Perception module for ARC-AGI-3 interactive-agent framework.

Extracts structured scene representations from raw grid observations using
4-connected connected-components, computes frame-level features, diffs
between frames, and produces stable state signatures.

Imports: only standard library + numpy.  No torch / arc_agi.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from arc_agi3.types import FrameData, SceneGraph, SceneObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cc4(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping color -> list of (r,c) for 4-connected components."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val == 0 or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == val:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components[val] = pixels
    return components


def _bbox(pixels: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return (min(rs), min(cs), max(rs), max(cs))


def _centroid(pixels: list[tuple[int, int]]) -> tuple[float, float]:
    n = len(pixels)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in pixels) / n, sum(p[1] for p in pixels) / n)


def _overlap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = bbox_a
    r0b, c0b, r1b, c1b = bbox_b
    r0 = max(r0a, r0b)
    c0 = max(c0a, c0b)
    r1 = min(r1a, r1b)
    c1 = min(c1a, c1b)
    if r1 < r0 or c1 < c0:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


# ---------------------------------------------------------------------------
# Perception class
# ---------------------------------------------------------------------------

class Perception:
    """Extract structured scene data from ARC-AGI-3 frame observations."""

    def __init__(self, ignore_border: int = 0) -> None:
        self.ignore_border = ignore_border

    def _detect_border(self, grid: np.ndarray) -> int:
        """Auto-detect uniform border rows/cols and return border width."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return 0
        top_row = grid[0, :]
        bottom_row = grid[h - 1, :]
        left_col = grid[:, 0]
        right_col = grid[:, w - 1]
        # Border if first/last rows are uniform non-zero.
        if (
            np.all(top_row == top_row[0])
            and np.all(bottom_row == bottom_row[0])
            and top_row[0] != 0
            and bottom_row[0] != 0
        ):
            return 1
        return 0

    def _apply_border_mask(self, grid: np.ndarray) -> np.ndarray:
        """Return a copy of grid with border cells zeroed out."""
        if self.ignore_border <= 0:
            return grid
        masked = grid.copy()
        h, w = masked.shape
        b = self.ignore_border
        if h > 2 * b:
            masked[:b, :] = 0
            masked[h - b :, :] = 0
        if w > 2 * b:
            masked[:, :b] = 0
            masked[:, w - b :] = 0
        return masked

    def extract_objects(self, grid: Optional[np.ndarray]) -> list[SceneObject]:
        if grid is None or grid.size == 0:
            return []
        # Auto-detect border if not explicitly set.
        if self.ignore_border == 0:
            self.ignore_border = self._detect_border(grid)
        masked = self._apply_border_mask(grid)
        components = _cc4(masked)
        objects: list[SceneObject] = []
        oid = 0
        for color, pixels in sorted(components.items()):
            if not pixels:
                continue
            bbox = _bbox(pixels)
            cen = _centroid(pixels)
            objects.append(SceneObject(
                object_id=oid,
                color=int(color),
                bbox=bbox,
                centroid=cen,
                area=len(pixels),
                pixels=pixels,
            ))
            oid += 1
        return objects

    def observe(self, frame: Optional[FrameData]) -> SceneGraph:
        if frame is None:
            return SceneGraph(objects=[], width=0, height=0, frame=None)
        g = frame.grid()
        if g is None:
            return SceneGraph(objects=[], width=0, height=0, frame=None)
        return SceneGraph(
            objects=self.extract_objects(g),
            width=int(g.shape[1]),
            height=int(g.shape[0]),
            frame=g,
        )

    def frame_features(self, frame: Optional[FrameData]) -> dict[str, Any]:
        if frame is None:
            return {"width": 0, "height": 0, "color_histogram": {}, "n_objects": 0, "colors_present": [], "nonzero_count": 0}
        g = frame.grid()
        if g is None:
            return {"width": 0, "height": 0, "color_histogram": {}, "n_objects": 0, "colors_present": [], "nonzero_count": 0}
        colors, counts = np.unique(g[g != 0], return_counts=True)
        hist = {int(c): int(n) for c, n in zip(colors, counts)}
        sg = self.extract_objects(g)
        return {
            "width": int(g.shape[1]),
            "height": int(g.shape[0]),
            "color_histogram": hist,
            "n_objects": len(sg),
            "colors_present": sorted(hist.keys()),
            "nonzero_count": int(np.count_nonzero(g)),
        }

    def diff(self, before: Optional[FrameData], after: Optional[FrameData]) -> dict[str, Any]:
        if before is None or after is None:
            return {"compatible": False}
        gb = before.grid()
        ga = after.grid()
        if gb is None or ga is None:
            return {"compatible": False}
        if gb.shape != ga.shape:
            return {"compatible": False}

        h, w = gb.shape
        # Mask border region before diffing (Task 7).
        masked_gb = self._apply_border_mask(gb)
        masked_ga = self._apply_border_mask(ga)
        changed_cells: list[tuple[int, int, int, int]] = []
        color_changes: list[dict[str, int]] = []
        for r in range(h):
            for c in range(w):
                ov = int(masked_gb[r, c])
                nv = int(masked_ga[r, c])
                if ov != nv:
                    changed_cells.append((r, c, ov, nv))
                    if ov != 0 and nv != 0:
                        color_changes.append({"r": r, "c": c, "old": ov, "new": nv})

        sg_b = self.observe(before)
        sg_a = self.observe(after)

        # Build color-keyed lists
        before_by_color: dict[int, list[SceneObject]] = defaultdict(list)
        for obj in sg_b.objects:
            before_by_color[obj.color].append(obj)
        after_by_color: dict[int, list[SceneObject]] = defaultdict(list)
        for obj in sg_a.objects:
            after_by_color[obj.color].append(obj)

        moved_objects: list[dict[str, Any]] = []
        added_objects: list[SceneObject] = []
        removed_objects: list[SceneObject] = []
        matched_before: set[int] = set()
        matched_after: set[int] = set()

        all_colors = set(before_by_color) | set(after_by_color)
        for color in sorted(all_colors):
            blist = before_by_color.get(color, [])
            alist = after_by_color.get(color, [])
            # Greedy match by maximal overlap
            used_a = set()
            for bo in blist:
                best_ao = None
                best_overlap = -1
                for idx, ao in enumerate(alist):
                    if idx in used_a:
                        continue
                    overlap = _overlap(bo.bbox, ao.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_ao = (idx, ao)
                if best_ao is not None:
                    idx, ao = best_ao
                    used_a.add(idx)
                    matched_before.add(bo.object_id)
                    matched_after.add(ao.object_id)
                    dr = round(ao.centroid[0] - bo.centroid[0])
                    dc = round(ao.centroid[1] - bo.centroid[1])
                    if bo.bbox != ao.bbox or dr != 0 or dc != 0:
                        moved_objects.append({
                            "color": color,
                            "from_bbox": bo.bbox,
                            "to_bbox": ao.bbox,
                            "dr": dr,
                            "dc": dc,
                            "from_centroid": bo.centroid,
                            "to_centroid": ao.centroid,
                        })
            # Unmatched before -> removed
            for bo in blist:
                if bo.object_id not in matched_before:
                    removed_objects.append(bo)
            # Unmatched after -> added
            for idx, ao in enumerate(alist):
                if idx not in used_a:
                    added_objects.append(ao)

        return {
            "compatible": True,
            "changed_cells": changed_cells,
            "moved_objects": moved_objects,
            "added_objects": added_objects,
            "removed_objects": removed_objects,
            "color_changes": color_changes,
        }

    def signature(self, frame: Optional[FrameData]) -> str:
        if frame is None:
            return "none"
        g = frame.grid()
        if g is None:
            return "none"
        # Auto-detect border on first call.
        if self.ignore_border == 0:
            self.ignore_border = self._detect_border(g)
        masked = self._apply_border_mask(g)
        cells = []
        for r in range(masked.shape[0]):
            for c in range(masked.shape[1]):
                v = int(masked[r, c])
                if v != 0:
                    cells.append(f"{r},{c},{v}")
        cells.sort()
        raw = ";".join(cells).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def describe_scene(self, scene: SceneGraph) -> str:
        if not scene.objects:
            return "empty scene"
        parts = []
        for obj in scene.objects:
            cr, cc = obj.centroid
            parts.append(f"color{obj.color}@({int(cr)},{int(cc)}) area{obj.area}")
        return "objects: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience aliases
# ---------------------------------------------------------------------------

_perception = Perception()

extract_objects = _perception.extract_objects
observe = _perception.observe
frame_features = _perception.frame_features
diff_frames = _perception.diff
scene_signature = _perception.signature


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    root = r"C:\Users\Student\Documents\Goldsnnail\Goldsnnail"
    if root not in sys.path:
        sys.path.insert(0, root)
    # When running this file directly Python inserts the script's directory
    # (arc_agi3/) into sys.path[0], which shadows stdlib "types" via
    # arc_agi3/types.py.  Remove it to avoid the name collision.
    arc_agi3_dir = os.path.join(root, "arc_agi3")
    if sys.path and os.path.normpath(sys.path[0]) == os.path.normpath(arc_agi3_dir):
        sys.path = sys.path[1:]

    from arc_agi3.types import FrameData, GameState

    # 1. observe
    grid0 = [[0, 9, 0], [0, 0, 0], [0, 0, 4]]
    fd0 = FrameData(game_id="test", state=GameState.PLAYING, frame=grid0, step=0)
    sg = Perception().observe(fd0)
    assert len(sg.objects) == 2, f"Expected 2 objects, got {len(sg.objects)}"
    colors = {o.color for o in sg.objects}
    assert colors == {4, 9}, f"Colors mismatch: {colors}"
    for o in sg.objects:
        if o.color == 9:
            assert o.bbox == (0, 1, 0, 1), f"Wrong bbox for 9: {o.bbox}"
            assert o.centroid == (0.0, 1.0), f"Wrong centroid for 9: {o.centroid}"
            assert o.area == 1
        if o.color == 4:
            assert o.bbox == (2, 2, 2, 2), f"Wrong bbox for 4: {o.bbox}"
            assert o.centroid == (2.0, 2.0), f"Wrong centroid for 4: {o.centroid}"
            assert o.area == 1

    # 2. diff
    grid1 = [[9, 0, 0], [0, 0, 0], [0, 0, 4]]
    fd1 = FrameData(game_id="test", state=GameState.PLAYING, frame=grid1, step=1)
    d = Perception().diff(fd0, fd1)
    assert d["compatible"] is True
    moved = d["moved_objects"]
    assert len(moved) == 1, f"Expected 1 moved, got {len(moved)}"
    m = moved[0]
    assert m["color"] == 9, f"Wrong color in moved: {m['color']}"
    assert m["dr"] == 0, f"Wrong dr: {m['dr']}"
    assert m["dc"] == -1, f"Wrong dc: {m['dc']}"

    # 3. signature
    sig1 = Perception().signature(fd0)
    sig2 = Perception().signature(fd0)
    sig3 = Perception().signature(fd1)
    assert sig1 == sig2, "Identical frames should have equal signatures"
    assert sig1 != sig3, "Different frames should have different signatures"

    # 4. frame_features
    ff = Perception().frame_features(fd0)
    assert ff["width"] == 3
    assert ff["height"] == 3
    assert ff["n_objects"] == 2
    assert ff["colors_present"] == [4, 9]
    assert ff["color_histogram"][9] == 1
    assert ff["color_histogram"][4] == 1
    assert ff["nonzero_count"] == 2

    # 5. describe_scene
    desc = Perception().describe_scene(sg)
    assert "color9" in desc
    assert "color4" in desc

    print("ok")
