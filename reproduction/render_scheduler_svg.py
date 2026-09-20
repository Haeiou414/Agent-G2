"""Render the dependency-free scheduler simulation as a publication-style SVG."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from reproduction.simulate_scheduler import run


REPO_ROOT = Path(__file__).resolve().parents[1]
COLORS = {1: "#2563eb", 2: "#f59e0b", 3: "#dc2626"}


def _polyline(
    values: list[tuple[int, float]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    max_step: int,
) -> str:
    return " ".join(
        f"{x0 + width * step / max(max_step, 1):.2f},{y0 + height * (1.0 - value):.2f}"
        for step, value in values
    )


def render(rows: list[dict[str, float | int]], title: str = "Agent-G² scheduler dynamics") -> str:
    width, height = 1100, 520
    margin_left, plot_top, plot_height = 72, 82, 330
    panel_width, gap = 450, 95
    max_step = max(int(row["step"]) for row in rows)
    panels = (("mu", "Gaussian center μ"), ("sigma", "Gaussian spread σ"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="700">{escape(title)}</text>',
        '<text x="550" y="62" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">Synthetic curriculum · paper-faithful online update · seed 7</text>',
    ]

    for panel_index, (field, label) in enumerate(panels):
        x0 = margin_left + panel_index * (panel_width + gap)
        parts.append(
            f'<text x="{x0 + panel_width / 2}" y="76" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="600">{label}</text>'
        )
        for tick in range(6):
            value = tick / 5
            y = plot_top + plot_height * (1 - value)
            parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + panel_width}" y2="{y:.2f}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{value:.1f}</text>')
        parts.append(f'<line x1="{x0}" y1="{plot_top + plot_height}" x2="{x0 + panel_width}" y2="{plot_top + plot_height}" stroke="#334155"/>')
        for cluster_id in (1, 2, 3):
            values = [
                (int(row["step"]), float(row[field]))
                for row in rows
                if int(row["cluster"]) == cluster_id
            ]
            points = _polyline(values, x0, plot_top, panel_width, plot_height, max_step)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{COLORS[cluster_id]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        parts.append(f'<text x="{x0 + panel_width / 2}" y="{plot_top + plot_height + 38}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Training step</text>')

    legend_y = 486
    for index, (cluster_id, label) in enumerate(((1, "easy"), (2, "medium"), (3, "hard"))):
        x = 420 + index * 130
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 26}" y2="{legend_y}" stroke="{COLORS[cluster_id]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 34}" y="{legend_y + 4}" font-family="sans-serif" font-size="13" fill="#334155">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tasks-per-cluster", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reproduction" / "figures" / "scheduler_dynamics.svg",
    )
    args = parser.parse_args()
    rows = run(args.steps, args.tasks_per_cluster, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(f"wrote scheduler figure to {args.output}")


if __name__ == "__main__":
    main()
