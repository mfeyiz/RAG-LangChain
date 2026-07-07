"""Render a structured chart spec to a PNG for AI-generated reports.

The report generator asks the LLM to emit chart data as a fenced ```chart block
holding JSON like:

    {"type": "line", "title": "…", "labels": ["2020", …], "values": [14.6, …]}

We render that with matplotlib (no sandbox needed — it's structured data, not
arbitrary code) and embed the PNG into the report so it exports to PDF/DOCX.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# SOCRadar brand palette — sampled from SOCRadar Template.docx (coral accent
# #FF4562 and the deep navy cover-page banner #191938), extended with tints/
# shades so multi-series charts stay legible while reading as one brand.
_PALETTE = [
    "#FF4562",  # brand coral (primary)
    "#191938",  # brand navy
    "#FF8CA0",  # coral tint
    "#5C5C82",  # navy tint
    "#B23350",  # coral shade
    "#8C8CA8",  # navy tint (lighter)
    "#434343",  # body-copy gray
    "#FFC2CD",  # pale coral
]

# A fenced ```chart … ``` block (language tag "chart").
CHART_BLOCK_RE = re.compile(r"```chart\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def render_chart_spec(spec: dict, out_path: Path) -> bool:
    """Render a chart spec dict to `out_path` (PNG). Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[report_charts] matplotlib unavailable: {exc}")
        return False

    ctype = str(spec.get("type") or "bar").lower()
    title = str(spec.get("title") or "")
    labels = [str(x) for x in (spec.get("labels") or [])]
    try:
        values = [float(v) for v in (spec.get("values") or [])]
    except (TypeError, ValueError):
        return False
    if not labels or not values or len(labels) != len(values):
        return False

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(values))]
    try:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        if ctype in ("pie", "doughnut"):
            wedge = {"width": 0.45} if ctype == "doughnut" else {}
            ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, wedgeprops=wedge)
            ax.axis("equal")
        elif ctype == "line" or ctype == "area":
            ax.plot(labels, values, marker="o", color=_PALETTE[0], linewidth=2)
            if ctype == "area":
                ax.fill_between(range(len(values)), values, color=_PALETTE[0], alpha=0.15)
            ax.grid(True, alpha=0.3)
        else:  # bar (default)
            ax.bar(labels, values, color=colors)
            ax.grid(axis="y", alpha=0.3)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        if ctype not in ("pie", "doughnut"):
            fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        return out_path.exists()
    except Exception as exc:
        print(f"[report_charts] render failed: {exc}")
        return False


def parse_spec(block_text: str) -> dict | None:
    """Parse the JSON inside a ```chart block, tolerating minor noise."""
    text = block_text.strip()
    try:
        return json.loads(text)
    except ValueError:
        # Best-effort: extract the first {...} object.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None
