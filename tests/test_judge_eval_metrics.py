"""Offline metric checks for the optional evaluation harness; no API or model download."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("judge_eval", Path(__file__).resolve().parents[1] / "eval/judge/run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_confusion_matrix_and_cost():
    rows = [
        {"expected_hit": True, "observed_hit": True, "judge_called": 1, "latency_ms": 20},
        {"expected_hit": False, "observed_hit": True, "judge_called": 1, "latency_ms": 30},
        {"expected_hit": False, "observed_hit": False, "judge_called": 0, "latency_ms": 10},
        {"expected_hit": True, "observed_hit": False, "judge_called": 0, "latency_ms": 40},
    ]
    result = module.summarize(rows)
    assert (result["TP"], result["TN"], result["FP"], result["FN"]) == (1, 1, 1, 1)
    assert result["correct"] == 2 and result["total"] == 4
    assert result["precision"] == result["recall"] == .5
    assert result["judge_calls"] == 2
    assert result["latency_ms_p50"] == 25
    assert result["judged_latency_ms_p50"] == 25
