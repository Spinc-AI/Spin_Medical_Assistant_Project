"""Runs every case against every {model} x {prompt version} combination.

    python evaluation/runner.py                     # against a live Core_LLM
    python evaluation/runner.py --fake              # no Core_LLM, no GPU (CI)
    python evaluation/runner.py --models gemma-4-12b,gemma-4-31b --prompts v1,v2

Writes one JSON file per combination into `results/` plus an aggregate
`results.json`. Results are gitignored (they are run output, and a run against
real conversations could contain sensitive text); the folder itself is kept.

Model and prompt version are FORCED per run, not routed: comparing two models
only means something if everything else is held still. That's what
InterviewRequest's `model_key`/`prompt_version` overrides are for -- the
normal routed path used by main.py is untouched.

A note for reading latency numbers: Core_LLM holds one model at a time, so the
first turn after switching models pays the load cost (tens of seconds to
minutes). It is real, it is not per-turn latency, and `latency.p50` is the
number to compare between runs rather than `latency.mean`.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run as a script from AI_Service/ or from evaluation/ -- both work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evaluation import metrics  # noqa: E402
from llm.core_llm_client import CoreLLMClient  # noqa: E402
from llm.interface import LLMClient, LLMResponse  # noqa: E402
from orchestrator import interview  # noqa: E402
from schemas.response import InterviewRequest, Message  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "cases.json"
RESULTS_DIR = HERE / "results"


class ScriptedFakeClient(LLMClient):
    """A stand-in Core_LLM for CI: valid JSON, no network, no GPU.

    It answers the first still-missing slot, so a case's conversation actually
    progresses -- a fake that always returned the same question would make the
    multi-turn cases meaningless. Urgency is deliberately always "routine":
    that is what makes the runs exercise postcheck's escalation rather than
    letting the fake hand back the right answer for free.
    """

    QUESTIONS = {
        "onset": "این مشکل از چه زمانی و چگونه شروع شد؟",
        "redness": "آیا چشم شما قرمز شده است؟",
        "vision_change": "آیا تغییری در دید خود احساس می‌کنید؟",
    }

    def __init__(self, latency: float = 0.01):
        self.latency = latency
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=None, model=None):
        self.calls += 1
        missing = self._missing_from_state_message(messages)
        question = (self.QUESTIONS.get(missing[0], "لطفاً بیشتر توضیح دهید.")
                    if missing else "اطلاعات ثبت شد. خلاصه‌ی شرح‌حال شما ضبط گردید.")
        domain = self._domain_from_state_message(messages) or "general"
        payload = {"domain": domain, "model": model or "fake",
                   "question": question, "urgency": "routine"}
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           model=model or "fake", latency=self.latency,
                           usage={"prompt_tokens": 100, "completion_tokens": 40,
                                  "total_tokens": 140})

    @staticmethod
    def _state_payload(messages) -> dict:
        from prompts.prompt_loader import STATE_HEADER
        for message in reversed(messages):
            content = message["content"] if isinstance(message, dict) else message.content
            if content.startswith(STATE_HEADER):
                return json.loads(content[len(STATE_HEADER):].strip())
        return {}

    def _missing_from_state_message(self, messages) -> list[str]:
        return self._state_payload(messages).get("missing", [])

    def _domain_from_state_message(self, messages) -> str | None:
        return self._state_payload(messages).get("domain")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def run_case(case: dict, model_key: str, prompt_version: str,
             client: LLMClient) -> dict:
    """Play one case's turns through run_turn, recording a row per turn."""
    state = None
    history: list[Message] = []
    turns: list[dict] = []
    expected_domain = case.get("expect", {}).get("domain") or case.get("domain")
    urgency_after = {int(k): v for k, v in (case.get("expect_urgency_after") or {}).items()}

    for index, user_message in enumerate(case["turns"]):
        trace = interview.TurnTrace()
        request = InterviewRequest(user_message=user_message,
                                   session_id=f"eval-{case['id']}",
                                   model_key=model_key, prompt_version=prompt_version)
        try:
            response, state = interview.run_turn(request, state, llm_client=client,
                                                 history=history, trace=trace)
        except Exception as exc:  # a bug in the pipeline is a failed turn, not a crashed run
            turns.append({"case": case["id"], "turn": index, "ok": False,
                          "latency": trace.latency, "llm_latency": trace.llm_latency,
                          "format_valid": False, "repaired": False,
                          "failure": f"exception:{type(exc).__name__}: {exc}"})
            continue

        history.append(Message(role="user", content=user_message))
        history.append(Message(role="assistant", content=response.question))
        turns.append({
            "case": case["id"], "turn": index, "user_message": user_message,
            "ok": trace.ok, "latency": trace.latency, "llm_latency": trace.llm_latency,
            "usage": trace.usage, "format_valid": trace.format_valid,
            "repaired": trace.repaired, "failure": trace.failure,
            "domain": response.domain, "urgency": response.urgency.value,
            "question": response.question, "complete": response.complete,
            "escalated": any("postcheck escalated" in n for n in response.notes),
            "notes": response.notes,
            "expected_domain": expected_domain,
            "expected_urgency": urgency_after.get(index) or (
                case.get("expect", {}).get("urgency")
                if index == len(case["turns"]) - 1 else None),
        })

    return {
        "case": case["id"],
        "description": case.get("description", ""),
        "turns": turns,
        "final_state": state.model_dump() if state else None,
        "expectations": check_expectations(case, state, turns),
    }


def check_expectations(case: dict, state, turns: list[dict]) -> dict:
    """Compare the case's hand-written `expect` block against what happened.

    Reported, not asserted: the runner's job is to measure, and a failing
    expectation is a result to look at, not a reason to stop the run.
    """
    expect = case.get("expect") or {}
    checks: dict[str, bool] = {}
    if state is None:
        return {"ran": False}

    if "domain" in expect:
        checks["domain"] = state.domain == expect["domain"]
    if "complete" in expect:
        checks["complete"] = state.is_complete() == expect["complete"]
    for slot, value in (expect.get("slots") or {}).items():
        checks[f"slot:{slot}"] = state.slots.get(slot) == value
    if "urgency" in expect and turns:
        checks["urgency"] = turns[-1].get("urgency") == expect["urgency"]
    if "expect_failure_on_turn" in case:
        index = case["expect_failure_on_turn"]
        checks[f"turn{index}_failed"] = (
            index < len(turns) and not turns[index].get("ok", True))

    return {"ran": True, "checks": checks, "passed": all(checks.values())}


def run(models: list[str], prompt_versions: list[str], client: LLMClient,
        cases: list[dict] | None = None, results_dir: Path = RESULTS_DIR) -> dict:
    """Every case x every model x every prompt version. Returns the aggregate."""
    cases = cases if cases is not None else load_cases()
    results_dir.mkdir(parents=True, exist_ok=True)
    per_run: dict[str, list[dict]] = {}
    written: list[str] = []

    for model_key in models:
        for prompt_version in prompt_versions:
            label = f"{model_key}__{prompt_version}"
            case_results = [run_case(c, model_key, prompt_version, client) for c in cases]
            turns = [t for cr in case_results for t in cr["turns"]]
            per_run[label] = turns

            path = results_dir / f"{label}.json"
            path.write_text(json.dumps({
                "model": model_key,
                "prompt_version": prompt_version,
                "policy_version": config.ROUTING_POLICY_VERSION,
                "safety_version": config.SAFETY_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cases": case_results,
                "metrics": metrics.summarize(turns),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(path.name)

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "prompt_versions": prompt_versions,
        "cases": [c["id"] for c in cases],
        "files": written,
        **metrics.combine(per_run),
    }
    (results_dir / "results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=config.DEFAULT_MODEL_KEY,
                        help="comma-separated Core_LLM model keys")
    parser.add_argument("--prompts", default="v1,v2",
                        help="comma-separated prompt versions")
    parser.add_argument("--fake", action="store_true",
                        help="use the scripted fake client -- no Core_LLM, no GPU")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    client = ScriptedFakeClient() if args.fake else CoreLLMClient()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    aggregate = run(models, prompts, client, results_dir=Path(args.results_dir))

    for label, summary in aggregate["runs"].items():
        print(f"{label}: {summary['turns']} turns, "
              f"failure_rate={summary['failure_rate']:.2f}, "
              f"format_validity={summary['format_validity']:.2f}, "
              f"latency_p50={summary['latency']['p50']}")
    print(f"\nNOT MEASURED (need a rubric or an agreed judge): "
          f"{', '.join(aggregate['unimplemented_metrics'])}")
    print(f"Wrote {len(aggregate['files']) + 1} files to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
