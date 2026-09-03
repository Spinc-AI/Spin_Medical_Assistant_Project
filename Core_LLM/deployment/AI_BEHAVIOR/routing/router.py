"""
router.py

تشخیص domain از روی متن شکایت بیمار (rule-based) و انتخاب مدل مناسب
بر اساس routing_policy.yaml موجود در همین پوشه.

جریان:
    patient complaint -> domain -> policy -> model_key
"""
from __future__ import annotations

import os
from typing import Optional

import yaml

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "eye": ["چشم", "چشمم", "تاری دید", "سوزش چشم", "قرمزی چشم"],
    "musculoskeletal": ["دست", "پا", "کمر", "زانو", "مفصل", "عضله"],
    "respiratory": ["سرفه", "تنگی نفس", "نفس", "خس خس"],
    "headache": ["سردرد", "سرم درد"],
    "gastrointestinal": ["دل درد", "دل‌درد", "تهوع", "اسهال", "معده"],
    "skin": ["پوست", "جوش", "خارش", "بثورات"],
}

DEFAULT_DOMAIN = "general"
EMERGENCY_DOMAIN = "emergency"

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI_BEHAVIOR_DIR = os.path.dirname(_HERE)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_red_flags() -> list[str]:
    path = os.path.join(_AI_BEHAVIOR_DIR, "safety", "red_flags.yaml")
    if not os.path.exists(path):
        return []
    data = _load_yaml(path)
    flags: list[str] = []
    for item in data.get("red_flags", []):
        flags.extend(item.get("keywords", []))
    return flags


def detect_domain(text: str) -> str:
    normalized = (text or "").strip()

    for flag in _load_red_flags():
        if flag in normalized:
            return EMERGENCY_DOMAIN

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                return domain

    return DEFAULT_DOMAIN


class Router:
    def __init__(self, policy_path: Optional[str] = None):
        if policy_path is None:
            policy_path = os.path.join(_HERE, "routing_policy.yaml")
        self.policy_path = policy_path
        self.policy = _load_yaml(policy_path)

    def route(self, domain: str) -> Optional[str]:
        routes = self.policy.get("routes", {})
        entry = routes.get(domain) or routes.get(DEFAULT_DOMAIN, {})
        return entry.get("preferred_model")

    def route_complaint(self, text: str) -> dict:
        domain = detect_domain(text)
        model_key = self.route(domain)
        return {"domain": domain, "selected_model": model_key}