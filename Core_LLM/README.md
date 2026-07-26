<div dir="rtl">

# ماژول مدل زبانی مرکزی (<span dir="ltr">Core_LLM</span>)

سرویس مدل زبانی مرکزیِ پروژه — مدل **<span dir="ltr">Aya Expanse 8B</span>** که با <span dir="ltr">Ollama</span> سرو می‌شود و پشت یک <span dir="ltr">API</span>ِ کوچکِ <span dir="ltr">HTTP</span> قرار دارد. بقیه‌ی اجزا حول <span dir="ltr">`llm_client.chat()`</span> ساخته می‌شوند.

- **سرویس و مستندات <span dir="ltr">API</span>:** [deployment/README.md](deployment/README.md)
- **راه‌اندازی سرور:** [deployment/SERVER_SETUP.md](deployment/SERVER_SETUP.md)

> **مجوز:** مدل <span dir="ltr">Aya Expanse</span> با مجوز **<span dir="ltr">CC-BY-NC</span>** (غیرتجاری) است — پیش از هر استفاده‌ی تجاری یا بالینی بازنگری شود.

## مدل‌های دیگر و اندازه‌های بزرگ‌تر

سرویس هر مدلی را که با <span dir="ltr">Ollama</span> بار شده باشد می‌پذیرد — کافیست در درخواست <span dir="ltr">`POST /chat`</span> فیلد <span dir="ltr">`model`</span> را ست کنید؛ بدون تغییر کد.

<div dir="ltr">

```bash
ollama pull aya-expanse:32b   # bigger/best Aya Expanse variant (default is the 8B)
ollama pull gemma4:e4b        # Apache 2.0 -- unlike Aya, no commercial/clinical restriction
ollama pull gemma4:31b        # Gemma 4's largest dense variant
```

</div>

سپس با مثلاً <span dir="ltr">`"model": "aya-expanse:32b"`</span> در درخواست صدایش بزنید.

## مدل‌های چندوجهیِ محلی (صوت مستقیم، بدون <span dir="ltr">Ollama</span>)

<span dir="ltr">Ollama</span> فعلاً ورودی صوتی را پشتیبانی نمی‌کند (فقط متن/تصویر) — برای همین یک مسیر جدا و مستقل از <span dir="ltr">Ollama</span> اضافه شده: <span dir="ltr">`POST /chat_audio`</span> — مستقیماً از طریق <span dir="ltr">`transformers`</span>، با انتخاب مدل از طریق فیلد <span dir="ltr">`model`</span>:

- <span dir="ltr">`gemma-4-e4b`</span> (پیش‌فرض) — <span dir="ltr">`google/gemma-4-E4B-it`</span>، سبک‌تر و سریع‌تر.
- <span dir="ltr">`gemma-4-12b`</span> — <span dir="ltr">`google/gemma-4-12B-it`</span>، بزرگ‌ترین نسخه‌ی **صوت‌پذیر** از <span dir="ltr">Gemma 4</span> — نسخه‌های <span dir="ltr">26B-A4B</span> و <span dir="ltr">31B</span> اصلاً ورودی صوتی ندارند (فقط تصویر/ویدیو)، پس این یکی، نه <span dir="ltr">31B</span>، قوی‌ترین گزینه‌ی <span dir="ltr">Gemma 4</span> برای صوت است.
- <span dir="ltr">`qwen3-omni-30b`</span> — <span dir="ltr">`Qwen/Qwen3-Omni-30B-A3B-Instruct`</span>، بهترین گزینه‌ی تست‌شده برای **صوت فارسی** (طبق بنچمارک مستقل <span dir="ltr">[PARSA-Bench](https://arxiv.org/html/2603.14456)</span>) — نیاز به حافظه‌ی <span dir="ltr">GPU</span> بیشتری دارد.

فقط یکی از این دو در حافظه نگه داشته می‌شود؛ تغییر مدل، مدل قبلی را خودکار آزاد می‌کند. جزئیات کامل در [deployment/README.md](deployment/README.md).

</div>
