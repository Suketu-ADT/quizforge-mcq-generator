"""Small pieces shared by the evaluation suites: the Metric record, number formatting and printing."""
from __future__ import annotations

from dataclasses import dataclass, field

from services.evaluation.stats import wilson_interval  # noqa: F401  (re-exported for the suites)


@dataclass
class Metric:
    name: str
    value: float | None  # None = could not be computed (the note says why); never shown as 0
    better: str  # "higher" / "lower"
    formula: str
    working: list[str] = field(default_factory=list)  # the calculation, with the real numbers substituted
    note: str = ""


def f(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"


def val(x: float | None, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def print_summary(metrics: list[Metric]) -> None:
    print(f"  {'Metric':<46}{'Value':>10}   Better")
    for m in metrics:
        print(f"  {m.name:<46}{val(m.value):>10}   {m.better}")


def print_calculations(metrics: list[Metric]) -> None:
    for m in metrics:
        print(f"\n[{m.name}]  better: {m.better}")
        print(f"  formula : {m.formula}")
        for line in m.working:
            print(f"  | {line}")
        print(f"  = {val(m.value)}")
        if m.note:
            print(f"  note    : {m.note}")


def markdown_calculations(metrics: list[Metric]) -> list[str]:
    out = ["<details><summary>Step-by-step calculations</summary>", ""]
    for m in metrics:
        out += [f"### {m.name}", f"*Formula:* `{m.formula}`  ", "", "```"] + m.working + [f"= {val(m.value)}", "```"]
        if m.note:
            out += [f"*{m.note}*"]
        out += [""]
    return out + ["</details>"]


def markdown_table(metrics: list[Metric]) -> list[str]:
    return ["| Metric | Value | Better |", "|---|---:|---|"] + [f"| {m.name} | {val(m.value)} | {m.better} |" for m in metrics]
