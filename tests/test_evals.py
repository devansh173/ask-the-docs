"""Evaluation harness tests: scoring, quota handling, metric summarisation.

None of these call a model. They cover the bookkeeping around the judge, which
is where a silent mistake turns into a wrong number in the README.
"""

from __future__ import annotations

import pandas as pd
import pytest

from askthedocs.evals.ragas_eval import _summarise
from askthedocs.evals.run import QuotaExhausted, _is_quota_error


class _Result:
    """Stands in for a ragas EvaluationResult."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


# --------------------------------------------------------------------------- #
# Metric summarisation
# --------------------------------------------------------------------------- #
def test_metrics_are_averaged_over_valid_rows():
    result = _Result(pd.DataFrame({"faithfulness": [1.0, 0.5, 0.0]}))
    assert _summarise(result, 3)["faithfulness"] == pytest.approx(0.5)


def test_judge_failures_are_reported_not_averaged_as_zero():
    """A failed judgement is not a score of zero.

    RAGAS runs with raise_exceptions=False so one bad response cannot abort a
    paid run; failures come back as NaN. Averaging them in would understate the
    metric and look identical to genuinely poor retrieval - which is exactly
    how a 0.000 ended up in an early result here.
    """
    result = _Result(pd.DataFrame({"faithfulness": [1.0, float("nan"), 1.0]}))
    scores = _summarise(result, 3)

    assert scores["faithfulness"] == pytest.approx(1.0), "NaN must not drag the mean"
    assert scores["faithfulness_failed"] == 1


def test_a_fully_failed_metric_is_none_rather_than_zero():
    result = _Result(pd.DataFrame({"faithfulness": [float("nan"), float("nan")]}))
    scores = _summarise(result, 2)

    assert scores["faithfulness"] is None, "no data is not the same as a zero score"
    assert scores["faithfulness_failed"] == 2


def test_ragas_metric_names_are_mapped_to_readme_names():
    result = _Result(
        pd.DataFrame({
            "llm_context_precision_with_reference": [0.5],
            "context_recall": [0.25],
            "some_unrelated_column": ["ignored"],
        })
    )
    scores = _summarise(result, 1)

    assert scores["context_precision"] == pytest.approx(0.5)
    assert scores["context_recall"] == pytest.approx(0.25)
    assert "some_unrelated_column" not in scores


def test_unreadable_result_degrades_to_empty():
    class Broken:
        def to_pandas(self):
            raise RuntimeError("ragas internals changed")

    assert _summarise(Broken(), 3) == {}


# --------------------------------------------------------------------------- #
# Quota handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED",
        "You exceeded your current quota, please check your plan",
        "insufficient_quota",
        "Your credit balance is too low",
    ],
)
def test_quota_refusals_are_recognised(message):
    assert _is_quota_error(RuntimeError(message))


@pytest.mark.parametrize("message", ["connection reset by peer", "invalid schema"])
def test_ordinary_errors_are_not_treated_as_quota(message):
    """Misreading a transient failure as a quota cap would abort a paid run."""
    assert not _is_quota_error(RuntimeError(message))


def test_quota_exhausted_is_an_error_not_a_silent_skip():
    assert issubclass(QuotaExhausted, RuntimeError)
