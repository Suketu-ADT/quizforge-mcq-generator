"""Data contracts shared by the evaluation engine, the service API and the dashboard."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Metric states. Missing data is never turned into a zero or a pass.
AVAILABLE = "available"
INSUFFICIENT_DATA = "insufficient_data"
NOT_APPLICABLE = "not_applicable"
EVALUATION_FAILED = "evaluation_failed"
METRIC_STATUSES = (AVAILABLE, INSUFFICIENT_DATA, NOT_APPLICABLE, EVALUATION_FAILED)

METRICS_VERSION = "1.0.0"

# Task types the engine knows how to run.
MCQ_ANSWER = "mcq_answer"  # answer a labelled multiple-choice question
MCQ_GENERATION = "mcq_generation"  # the project's real use case: write MCQs from source material
SAFETY = "safety"  # adversarial material fed through the generation pipeline
TASK_TYPES = (MCQ_ANSWER, MCQ_GENERATION, SAFETY)

# Sample states.
SAMPLE_COMPLETED = "completed"
SAMPLE_ERROR = "error"  # a technical failure stopped the task
SAMPLE_CANCELLED = "cancelled"

# Run states.
RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED = "queued", "running", "completed", "failed", "cancelled"


@dataclass
class MetricResult:
    key: str
    name: str
    value: float | None
    unit: str  # "%", "score 1-5", "score 0-1", "ms", "USD"
    status: str
    formula: str
    numerator: float | None = None
    denominator: float | None = None
    sample_count: int = 0
    evaluator: str = ""
    version: str = METRICS_VERSION
    higher_is_better: bool = True
    group: str = ""  # quality / safety / speed / cost / reliability
    description: str = ""
    working: list[str] = field(default_factory=list)  # the calculation with the real numbers substituted
    limitations: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)  # breakdowns, distributions, failure examples

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MetricResult":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    @property
    def is_available(self) -> bool:
        return self.status == AVAILABLE and self.value is not None


def unavailable(key: str, name: str, unit: str, status: str, formula: str, reason: str, **kw) -> MetricResult:
    """A metric that cannot be computed - value stays None and the reason is recorded."""
    lims = list(kw.pop("limitations", [])) + [reason]
    return MetricResult(key=key, name=name, value=None, unit=unit, status=status, formula=formula, limitations=lims, **kw)
