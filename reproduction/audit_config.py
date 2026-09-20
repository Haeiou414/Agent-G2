"""Compare the checked-out ALFWorld recipe with paper Appendix B, Table 5."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditField:
    name: str
    paper_value: float
    source_path: str
    pattern: str
    risk: str


FIELDS = (
    AuditField(
        "peak learning rate",
        1e-5,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"actor_rollout_ref\.actor\.optim\.lr=([0-9.eE+-]+)",
        "changes convergence speed",
    ),
    AuditField(
        "max prompt length",
        7000,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"data\.max_prompt_length=([0-9]+)",
        "may truncate long trajectories",
    ),
    AuditField(
        "difficulty clusters K",
        3,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}num_difficulty_groups:\s*([0-9]+)",
        "changes local-statistic granularity",
    ),
    AuditField(
        "center scale lambda",
        1.0,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}lambda_value:\s*([0-9.eE+-]+)",
        "changes the difficulty correction",
    ),
    AuditField(
        "global baseline upper bound",
        1.0,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}mu_global_max:\s*([0-9.eE+-]+)",
        "caps global guidance below the paper range",
    ),
    AuditField(
        "guidance ratio upper bound",
        1.0,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}max_length:\s*([0-9.eE+-]+)",
        "prevents sampling deeper prefixes",
    ),
    AuditField(
        "ALFWorld training epochs",
        200,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"trainer\.total_epochs=([0-9]+)",
        "changes the training budget",
    ),
    AuditField(
        "tasks per step",
        16,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"^train_data_size=([0-9]+)",
        "changes batch statistics",
    ),
    AuditField(
        "rollouts per task",
        8,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"^group_size=([0-9]+)",
        "changes GRPO and schedule estimates",
    ),
    AuditField(
        "cluster EMA rate alpha",
        0.2,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}alpha:\s*([0-9.eE+-]+)",
        "changes adaptation speed",
    ),
    AuditField(
        "variance floor sigma_min",
        0.1,
        "verl/trainer/config/ppo_trainer.yaml",
        r"^\s{2}sigma_min:\s*([0-9.eE+-]+)",
        "changes minimum exploration spread",
    ),
    AuditField(
        "auxiliary loss weight eta",
        0.5,
        "examples/gmsv_trainer/run_alfworld.sh",
        r"gmsv\.prefix_sft\.loss_coef=([0-9.eE+-]+)",
        "changes prefix imitation strength",
    ),
)


def audit(repo_root: Path = REPO_ROOT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache: dict[str, str] = {}
    for field in FIELDS:
        text = cache.setdefault(
            field.source_path,
            (repo_root / field.source_path).read_text(encoding="utf-8"),
        )
        match = re.search(field.pattern, text, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"could not extract {field.name} from {field.source_path}")
        implementation_value = float(match.group(1))
        matches = math.isclose(implementation_value, field.paper_value, rel_tol=1e-12, abs_tol=1e-12)
        rows.append(
            {
                "name": field.name,
                "paper": field.paper_value,
                "implementation": implementation_value,
                "status": "MATCH" if matches else "MISMATCH",
                "source": field.source_path,
                "risk": field.risk,
            }
        )
    return rows


def render_markdown(rows: list[dict[str, object]], commit: str = "e065918") -> str:
    mismatches = sum(row["status"] == "MISMATCH" for row in rows)
    lines = [
        "# Paper-to-code configuration audit",
        "",
        f"Checked upstream commit `{commit}` against Agent-G² Appendix B, Table 5.",
        f"Result: **{mismatches} mismatches across {len(rows)} audited fields**.",
        "",
        "| Field | Paper | Checked-out recipe | Status | Source | Reproduction risk |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | `{paper:g}` | `{implementation:g}` | **{status}** | `{source}` | {risk} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Two runtime discrepancies are audited separately because they are semantic behaviors, not scalar configuration: (1) the paper first averages the R rollouts for each task before computing cluster variance, and (2) prefix length is based on expert action count rather than token count. The reproduction patches enforce both definitions.",
            "",
            "The [released 1.5B checkpoint model card](https://huggingface.co/xiamoent/Agent-G2-alfworld-1.5b) documents the code-side recipe (learning rate `1e-6`, prompt length `4096`, 300 epochs) and states that the exact uploaded checkpoint step is not identified. This project therefore keeps `paper-table5` and `released-recipe` results as separate experiment tracks.",
            "",
            "This report is generated by `python -m reproduction.audit_config`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reproduction" / "reports" / "config_audit.md",
    )
    args = parser.parse_args()
    rows = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    print(f"wrote audit with {len(rows)} fields to {args.output}")


if __name__ == "__main__":
    main()
