"""Evaluation harness: `runner.py` plays `cases.json` through the interview
loop for each {model} x {prompt version}, `metrics.py` aggregates the result.

`metrics.py` implements the objective metrics only. question_relevance,
safety_score and clinical_appropriateness are explicit placeholders (always
None) pending a rubric or an agreed LLM-as-judge -- see its module docstring.
"""
