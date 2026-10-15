"""UCS-Bench-style evaluation harness for DirectMe.

Public API::

    from directme.eval import (
        UCSDimension,
        DimensionResult,
        EvaluationReport,
        UCSBenchEvaluator,
        classify_dimension,
    )
"""

from directme.eval.ucsbench import (
    DimensionResult,
    EvaluationReport,
    UCSBenchEvaluator,
    UCSDimension,
    classify_dimension,
)

__all__ = [
    "DimensionResult",
    "EvaluationReport",
    "UCSBenchEvaluator",
    "UCSDimension",
    "classify_dimension",
]
