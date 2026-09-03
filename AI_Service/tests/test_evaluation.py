"""Smoke test for the evaluation harness: the whole runner, no Core_LLM, no GPU."""
import json

import pytest

from evaluation import metrics
from evaluation.runner import ScriptedFakeClient, load_cases, run


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


def test_cases_file_is_well_formed():
    cases = load_cases()
    assert 5 <= len(cases) <= 10, "the spec asks for 5-10 hand-written cases"
    assert len({c["id"] for c in cases}) == len(cases), "case ids must be unique"
    for case in cases:
        assert case["turns"], f"{case['id']} has no turns"
        assert case.get("description")


def test_at_least_one_case_exercises_urgent():
    urgencies = {(c.get("expect") or {}).get("urgency") for c in load_cases()}
    assert "urgent" in urgencies


def test_runner_writes_a_file_per_combination_plus_an_aggregate(results_dir):
    aggregate = run(["gemma-4-12b", "gemma-4-31b"], ["v1", "v2"],
                    ScriptedFakeClient(), results_dir=results_dir)

    written = {p.name for p in results_dir.iterdir()}
    assert written == {"gemma-4-12b__v1.json", "gemma-4-12b__v2.json",
                       "gemma-4-31b__v1.json", "gemma-4-31b__v2.json",
                       "results.json"}
    assert set(aggregate["runs"]) == {"gemma-4-12b__v1", "gemma-4-12b__v2",
                                      "gemma-4-31b__v1", "gemma-4-31b__v2"}


def test_every_case_meets_its_expectations_under_the_fake_client(results_dir):
    """The fake always answers 'routine', so the urgent cases only pass if
    postcheck really escalates them."""
    run(["gemma-4-12b"], ["v1"], ScriptedFakeClient(), results_dir=results_dir)
    written = json.loads((results_dir / "gemma-4-12b__v1.json").read_text(encoding="utf-8"))
    failed = {c["case"]: c["expectations"]["checks"]
              for c in written["cases"] if not c["expectations"]["passed"]}
    assert not failed, f"cases missing their expectations: {failed}"


def test_results_record_which_versions_produced_them(results_dir):
    run(["gemma-4-12b"], ["v2"], ScriptedFakeClient(), results_dir=results_dir)
    written = json.loads((results_dir / "gemma-4-12b__v2.json").read_text(encoding="utf-8"))
    assert written["model"] == "gemma-4-12b"
    assert written["prompt_version"] == "v2"
    assert written["policy_version"] and written["safety_version"]
    assert written["generated_at"]


def test_postcheck_escalation_shows_up_in_the_results(results_dir):
    aggregate = run(["gemma-4-12b"], ["v1"], ScriptedFakeClient(), results_dir=results_dir)
    assert aggregate["runs"]["gemma-4-12b__v1"]["escalations"] >= 1


def test_forced_model_and_prompt_are_actually_used(results_dir):
    class Recording(ScriptedFakeClient):
        def __init__(self):
            super().__init__()
            self.models, self.systems = set(), set()

        def generate(self, messages, temperature=0.3, max_tokens=None, model=None):
            self.models.add(model)
            self.systems.add(messages[0]["content"][:40])
            return super().generate(messages, temperature, max_tokens, model)

    client = Recording()
    run(["gemma-4-31b"], ["v2"], client, results_dir=results_dir)
    assert client.models == {"gemma-4-31b"}
    from prompts.prompt_loader import load_system
    assert client.systems == {load_system("v2")[:40]}


# --- metrics ----------------------------------------------------------------

def test_summarize_computes_the_objective_metrics():
    turns = [
        {"ok": True, "latency": 1.0, "llm_latency": 0.9, "format_valid": True,
         "usage": {"total_tokens": 100}, "domain": "eye", "expected_domain": "eye",
         "urgency": "routine", "expected_urgency": "routine"},
        {"ok": False, "latency": 3.0, "llm_latency": 2.0, "format_valid": False,
         "failure": "output_invalid:x", "domain": "eye", "expected_domain": "eye",
         "urgency": "routine", "expected_urgency": "urgent"},
    ]
    summary = metrics.summarize(turns)
    assert summary["turns"] == 2
    assert summary["failure_rate"] == 0.5
    assert summary["format_validity"] == 0.5
    assert summary["latency"]["mean"] == 2.0
    assert summary["latency"]["total"] == 4.0
    assert summary["domain_accuracy"] == 1.0
    assert summary["urgency_accuracy"] == 0.5
    assert summary["throughput"]["turns_per_second"] == 0.5


def test_tokens_are_none_when_no_backend_reports_usage():
    """Core_LLM's ChatResponse has no `usage` field, so a real run reports
    none -- and an invented number would silently corrupt throughput."""
    summary = metrics.summarize([{"ok": True, "latency": 1.0, "llm_latency": 1.0,
                                  "format_valid": True, "usage": None}])
    assert summary["tokens"] is None
    assert summary["throughput"]["tokens_per_second"] is None


@pytest.mark.parametrize("metric", metrics.PLACEHOLDER_METRICS)
def test_subjective_metrics_are_explicitly_unimplemented(metric):
    """These need a rubric or an agreed LLM-as-judge. None, never a number:
    a plausible-looking score would get charted and believed."""
    assert metrics.summarize([{"ok": True, "latency": 1.0, "llm_latency": 1.0,
                               "format_valid": True}])[metric] is None


def test_empty_run_does_not_divide_by_zero():
    assert metrics.summarize([])["turns"] == 0


def test_combine_reports_the_unimplemented_bucket():
    combined = metrics.combine({"a": [{"ok": True, "latency": 1.0, "llm_latency": 1.0,
                                       "format_valid": True}]})
    assert combined["unimplemented_metrics"] == list(metrics.PLACEHOLDER_METRICS)
    assert combined["all"]["turns"] == 1
