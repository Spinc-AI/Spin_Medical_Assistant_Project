"""تست‌های ساده برای تشخیص شرایط اورژانسی (safety detection)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routing.router import detect_domain


def test_emergency_detected():
    assert detect_domain("نفس نمی‌تونم بکشم") == "emergency"


def test_non_emergency_not_flagged():
    assert detect_domain("چشمم درد می‌کنه") != "emergency"


def test_general_fallback():
    assert detect_domain("حالم زیاد خوب نیست") == "general"


if __name__ == "__main__":
    test_emergency_detected()
    test_non_emergency_not_flagged()
    test_general_fallback()
    print("All tests passed successfully✅")