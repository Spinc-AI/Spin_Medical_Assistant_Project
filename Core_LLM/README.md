<div dir="rtl">

# ماژول مدل زبانی مرکزی (<span dir="ltr">Core_LLM</span>)

سرویس مدل زبانی مرکزیِ پروژه — همه‌ی مدل‌ها مستقیماً از طریق <span dir="ltr">`transformers`</span> سرو می‌شوند (**نه <span dir="ltr">Ollama</span>**) پشت یک <span dir="ltr">API</span>ِ کوچکِ <span dir="ltr">HTTP</span>. <span dir="ltr">Ollama</span> کاملاً کنار گذاشته شد: چون اصلاً ورودی صوتی را پشتیبانی نمی‌کند، نگه‌داشتنش فقط برای نقش متنی یعنی دو مسیر سرویس‌دهیِ موازی به‌جای یکی. حالا <span dir="ltr">`model.py`</span> (رجیستری مدل‌ها + <span dir="ltr">`LLMManager`</span>، یک مدل در هر لحظه در حافظه) پشتِ **هر دو** مسیرِ متنی (<span dir="ltr">`/chat`</span>) و صوتی (<span dir="ltr">`/chat_audio`</span>) قرار دارد — یک مدل را یک‌بار بار کنید، بدون بارگذاریِ دوباره هم برای چت متنی و هم برای صوتِ <span dir="ltr">BuAli</span> قابل استفاده است.

- **سرویس و مستندات <span dir="ltr">API</span>:** [deployment/README.md](deployment/README.md)
- **راه‌اندازی سرور:** [deployment/SERVER_SETUP.md](deployment/SERVER_SETUP.md)

## مدل‌های موجود

| کلید <span dir="ltr">`model`</span> | مدل | نقش |
|---|---|---|
| <span dir="ltr">`aya-expanse-8b`</span> (پیش‌فرض) | <span dir="ltr">`CohereLabs/aya-expanse-8b`</span> | فقط متن |
| <span dir="ltr">`aya-expanse-32b`</span> | <span dir="ltr">`CohereLabs/aya-expanse-32b`</span> | فقط متن، بزرگ‌تر |
| <span dir="ltr">`gemma-4-31b`</span> | <span dir="ltr">`google/gemma-4-31B-it`</span> | فقط متن — قوی‌ترین <span dir="ltr">Gemma 4</span> به‌طور کلی، اما <span dir="ltr">26B-A4B</span>/<span dir="ltr">31B</span> اصلاً ورودی صوتی ندارند |
| <span dir="ltr">`gemma-4-e4b`</span> | <span dir="ltr">`google/gemma-4-E4B-it`</span> | متن **و صوت** — سبک‌تر، سریع‌تر |
| <span dir="ltr">`gemma-4-12b`</span> | <span dir="ltr">`google/gemma-4-12B-it`</span> | متن **و صوت** — بزرگ‌ترین نسخه‌ی صوت‌پذیرِ <span dir="ltr">Gemma 4</span> |
| <span dir="ltr">`qwen3-omni-30b`</span> | <span dir="ltr">`Qwen/Qwen3-Omni-30B-A3B-Instruct`</span> | متن **و صوت** — بهترین گزینه‌ی تست‌شده برای **صوت فارسی** (طبق بنچمارک مستقل <span dir="ltr">[PARSA-Bench](https://arxiv.org/html/2603.14456)</span>) — نیاز به حافظه‌ی <span dir="ltr">GPU</span> بیشتری دارد |

> **مجوز:** خانواده‌ی <span dir="ltr">Aya Expanse</span> با مجوز **<span dir="ltr">CC-BY-NC</span>** (غیرتجاری) است — پیش از هر استفاده‌ی تجاری یا بالینی بازنگری شود. بقیه‌ی مدل‌ها <span dir="ltr">Apache 2.0</span> هستند.

فقط یک مدل در هر لحظه در حافظه نگه داشته می‌شود؛ تغییر مدل (از هر دو مسیرِ <span dir="ltr">`/chat`</span> یا <span dir="ltr">`/chat_audio`</span>) مدل قبلی را خودکار آزاد می‌کند. جزئیات کامل <span dir="ltr">API</span> و مثال‌ها در [deployment/README.md](deployment/README.md).

</div>
