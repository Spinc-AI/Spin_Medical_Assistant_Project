"""تعریف رسمی ساختار داده‌های red flag، برای اعتبارسنجی فایل red_flags.yaml"""
from pydantic import BaseModel


class RedFlagCondition(BaseModel):
    condition: str          # اسم شرایط خطرناک، مثل respiratory_distress
    keywords: list[str]     # لیست کلمات کلیدی که این شرایط رو نشون می‌دن
    escalation: str         # به کجا escalate بشه، مثلاً "emergency"


class RedFlagsFile(BaseModel):
    red_flags: list[RedFlagCondition]