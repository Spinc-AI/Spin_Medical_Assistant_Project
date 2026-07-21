<div dir="rtl">

# ماژول تبدیل گفتار به متن (<span dir="ltr">STT</span>)

ماژول تبدیل گفتار فارسی به متن. هسته‌ی آن یک سرویس <span dir="ltr">FastAPI</span> است که یک مدل را در حافظه بارگذاری می‌کند و فایل صوتی را به متن تبدیل می‌کند. علاوه بر سرویس، ابزار بنچمارک مدل‌ها و نمونه‌های دمو هم در این ماژول هست.

## ساختار پوشه‌ها

<div dir="ltr">

```
Module_Speech_To_Text/
├─ README.md          # همین فایل
├─ requirements.txt
├─ benchmark/         # ارزیابی و مقایسه‌ی مدل‌ها (WER)
├─ demo/              # نمونه‌های صوتی و نوت‌بوک دمو
└─ deployment/        # سرویس API (FastAPI) — بخش اصلی
   ├─ samples/        # نمونه‌های صوتی نمونه
   ├─ tests/
   ├─ run.bat
   └─ README.md       # مستندات و نحوه‌ی اجرای API
```

</div>

برای راه‌اندازی و جزئیات <span dir="ltr">API</span> به <span dir="ltr">[deployment/README.md](deployment/README.md)</span> مراجعه کنید.

</div>
