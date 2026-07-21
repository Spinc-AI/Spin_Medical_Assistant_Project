<div dir="rtl">

# ماژول مدل زبانی مرکزی (<span dir="ltr">Core_LLM</span>)

سرویس مدل زبانی مرکزیِ پروژه — مدل **<span dir="ltr">Aya Expanse 8B</span>** که با <span dir="ltr">Ollama</span> سرو می‌شود و پشت یک <span dir="ltr">API</span>ِ کوچکِ <span dir="ltr">HTTP</span> قرار دارد. بقیه‌ی اجزا حول <span dir="ltr">`llm_client.chat()`</span> ساخته می‌شوند.

- **سرویس و مستندات <span dir="ltr">API</span>:** [deployment/README.md](deployment/README.md)
- **راه‌اندازی سرور:** [deployment/SERVER_SETUP.md](deployment/SERVER_SETUP.md)

> **مجوز:** مدل <span dir="ltr">Aya Expanse</span> با مجوز **<span dir="ltr">CC-BY-NC</span>** (غیرتجاری) است — پیش از هر استفاده‌ی تجاری یا بالینی بازنگری شود.

## مدل دوم: <span dir="ltr">Gemma 4 (E4B)</span>

سرویس هر مدلی را که با <span dir="ltr">Ollama</span> بار شده باشد می‌پذیرد — کافیست در درخواست <span dir="ltr">`POST /chat`</span> فیلد <span dir="ltr">`model`</span> را ست کنید؛ بدون تغییر کد. برای افزودن <span dir="ltr">Gemma 4 E4B</span> (مجوز <span dir="ltr">Apache 2.0</span> — برخلاف <span dir="ltr">Aya</span>، برای استفاده‌ی تجاری/بالینی محدودیتی ندارد) روی سرور اجرا کنید:

<div dir="ltr">

```bash
ollama pull gemma4:e4b
```

</div>

سپس با <span dir="ltr">`"model": "gemma4:e4b"`</span> در درخواست صدایش بزنید. توجه: <span dir="ltr">Ollama</span> فعلاً ورودی صوتی را پشتیبانی نمی‌کند (فقط متن/تصویر) — با اینکه خودِ <span dir="ltr">Gemma 4 E4B</span> از صوت پشتیبانی می‌کند، از این مسیر (<span dir="ltr">Ollama</span>) قابل استفاده نیست.

</div>
