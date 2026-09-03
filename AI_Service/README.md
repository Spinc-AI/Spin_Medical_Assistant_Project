<div dir="rtl">

# ماژول مصاحبه و مسیریابی هوشمند (<span dir="ltr">AI_Service</span>)

لایه‌ی **گفتگوی چندنوبتی (<span dir="ltr">interview</span>) + مسیریابی مدل** که بین «<span dir="ltr">Backend</span>» و <span dir="ltr">Core_LLM</span> می‌نشیند. پیام‌های متوالی بیمار را می‌گیرد، دامنه‌ی بالینی را تشخیص می‌دهد، مدل مناسب را انتخاب می‌کند، پرسش بعدی را می‌سازد و «وضعیت مکالمه» (<span dir="ltr">ConversationState</span>) را نوبت‌به‌نوبت پر می‌کند.

> **این ماژول هرگز مدلی را خودش لود نمی‌کند.** هر پاسخ، یک فراخوانی <span dir="ltr">HTTP</span> به <span dir="ltr">`POST /chat`</span> روی <span dir="ltr">Core_LLM</span> است — هم طبق قاعده‌ی <span dir="ltr">CONTRIBUTING.md</span> («ماژول‌ها فقط از طریق <span dir="ltr">API</span> ارتباط می‌گیرند»)، هم چون نسخه‌ی دومِ مدل روی <span dir="ltr">GPU</span> حافظه را بی‌دلیل دو برابر می‌کند. پس این ماژول به <span dir="ltr">GPU</span> نیاز ندارد و <span dir="ltr">torch</span>/<span dir="ltr">transformers</span> در <span dir="ltr">requirements</span> آن نیست.

## تفاوت با <span dir="ltr">Orchestrator/</span>

این دو **جایگزین هم نیستند**؛ دو ابزار برای دو مسئله‌اند و فعلاً کنار هم می‌مانند.

| | <span dir="ltr">`Orchestrator/`</span> | <span dir="ltr">`AI_Service/`</span> |
|---|---|---|
| کاربرد | اجرای <span dir="ltr">instruction</span>های چندمرحله‌ای ثابت (<span dir="ltr">STT → LLM →</span> پر کردن فرم) | مصاحبه‌ی پویا برای تریاژ علائم |
| انتخاب مدل | کلاینت صریحاً می‌فرستد | خودِ سیستم، بر اساس دامنه و <span dir="ltr">policy</span> نسخه‌دار |
| نسخه‌بندی پرامپت | ندارد | دارد (<span dir="ltr">`prompts/v1`</span>، <span dir="ltr">`v2`</span>) برای تست <span dir="ltr">A/B</span> |
| ارزیابی | ندارد | دارد (<span dir="ltr">`evaluation/`</span>) |
| ایمنی | به عهده‌ی پرامپت سیستمی | <span dir="ltr">precheck</span>/<span dir="ltr">postcheck</span>/<span dir="ltr">output_validator</span> مجزا |

## اجرا

<div dir="ltr">

```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```

</div>

روی <span dir="ltr">`0.0.0.0:9100`</span> بالا می‌آید (مستندات تعاملی در <span dir="ltr">`/docs`</span>). <span dir="ltr">Core_LLM</span> باید در دسترس باشد؛ اگر روی آدرس پیش‌فرض <span dir="ltr">`http://127.0.0.1:8001`</span> نیست، <span dir="ltr">`CORE_LLM_URL`</span> را در <span dir="ltr">`.env`</span> تنظیم کنید (نگاه کنید به <span dir="ltr">`.env.example`</span>).

## <span dir="ltr">API</span>

| متد و مسیر | کاربرد |
|---|---|
| <span dir="ltr">`GET /`</span> | سلامت + نسخه‌های فعال (<span dir="ltr">prompt</span>/<span dir="ltr">policy</span>/<span dir="ltr">safety</span>) و دامنه‌های قابل مسیریابی |
| <span dir="ltr">`GET /health`</span> | همان، به‌علاوه‌ی در دسترس بودن واقعی <span dir="ltr">Core_LLM</span> (هم‌سبک با <span dir="ltr">`GET /health`</span> در <span dir="ltr">Orchestrator</span>) |
| <span dir="ltr">`POST /interview`</span> | یک نوبت مصاحبه: <span dir="ltr">`InterviewRequest`</span> → <span dir="ltr">`InterviewResponse`</span> |
| <span dir="ltr">`DELETE /interview/{session_id}`</span> | پاک کردن یک مکالمه از حافظه |

نوبت اول را بدون <span dir="ltr">`session_id`</span> بفرستید؛ در پاسخ یک <span dir="ltr">`session_id`</span> می‌گیرید که در نوبت‌های بعدی باید همان را بفرستید.

<div dir="ltr">

```bash
curl -X POST http://localhost:9100/interview \
  -H "Content-Type: application/json" \
  -d '{"user_message": "چشم راستم از دیروز قرمز شده"}'
```

```json
{
  "domain": "eye",
  "model": "gemma-4-12b",
  "question": "آیا در دید شما تغییری ایجاد شده است؟",
  "urgency": "routine",
  "session_id": "776246965c6d4e1d...",
  "complete": false,
  "conversation_state": {"domain": "eye",
                         "slots": {"onset": "reported", "redness": "yes",
                                   "vision_change": null},
                         "turn_count": 1},
  "policy_version": "v1", "prompt_version": "v1", "safety_version": "v1",
  "notes": []
}
```

</div>

### وضعیت مکالمه کجا نگه داشته می‌شود؟

پیش‌فرض: **سمت سرور، در حافظه، با کلید <span dir="ltr">`session_id`</span>**. کلاینت فقط یک رشته را نگه می‌دارد. عیبش صریح است: با <span dir="ltr">restart</span> از بین می‌رود و بدون <span dir="ltr">sticky session</span> روی چند <span dir="ltr">replica</span> کار نمی‌کند. این را پذیرفتیم چون هنوز هیچ لایه‌ی <span dir="ltr">persistence</span> در این مخزن وجود ندارد (<span dir="ltr">Orchestrator</span> هم <span dir="ltr">session</span>هایش را در حافظه نگه می‌دارد). راه فرار در خودِ <span dir="ltr">API</span> هست: اگر <span dir="ltr">`conversation_state`</span> را صریحاً بفرستید، همان استفاده می‌شود و سرور نسخه‌ی خودش را نادیده می‌گیرد.

## ساختار

<div dir="ltr">

```
AI_Service/
├─ orchestrator/  interview.py (حلقه‌ی هر نوبت), state.py (اسلات‌ها: استخراج و ادغام)
├─ router/        policy.py (قوانین نسخه‌دار), router.py (پوشش پایدار)
├─ llm/           interface.py, core_llm_client.py (تنها مسیر واقعی), vllm_client.py (بلااستفاده)
├─ prompts/       v1/, v2/, prompt_loader.py
├─ safety/        precheck.py, postcheck.py, output_validator.py
├─ schemas/       state.py, routing.py, response.py
├─ evaluation/    cases.json, runner.py, metrics.py, results/
├─ config.py, main.py, requirements.txt, .env.example
└─ tests/
```

</div>

### افزودن دامنه‌ی جدید

فقط یک ورودی در <span dir="ltr">`DOMAIN_SLOTS`</span> (فایل <span dir="ltr">`schemas/state.py`</span>)، به‌علاوه‌ی کلیدواژه‌ها در <span dir="ltr">`router/policy.py`</span> و قوانین استخراج در <span dir="ltr">`orchestrator/state.py`</span>. هیچ تغییری در <span dir="ltr">`interview.py`</span> لازم نیست — طراحی از ابتدا چنددامنه‌ای است و <span dir="ltr">`eye`</span> صرفاً تنها دامنه‌ای است که مشخصات بالینی‌اش داده شده.

## ارزیابی

<div dir="ltr">

```bash
python evaluation/runner.py --fake                       # بدون Core_LLM و بدون GPU
python evaluation/runner.py --models gemma-4-12b,gemma-4-31b --prompts v1,v2
```

</div>

برای هر ترکیب {مدل} × {نسخه‌ی پرامپت} یک فایل در <span dir="ltr">`evaluation/results/`</span> نوشته می‌شود، به‌علاوه‌ی یک <span dir="ltr">`results.json`</span> تجمیعی. محتویات <span dir="ltr">`results/`</span> در <span dir="ltr">`.gitignore`</span> است.

**متریک‌های عینی (پیاده‌سازی‌شده):** <span dir="ltr">`latency`</span>، <span dir="ltr">`tokens`</span>، <span dir="ltr">`throughput`</span>، <span dir="ltr">`failure_rate`</span>، <span dir="ltr">`format_validity`</span>، <span dir="ltr">`domain_accuracy`</span>، <span dir="ltr">`urgency_accuracy`</span>.

**متریک‌های پیاده‌سازی‌نشده (عمداً همیشه <span dir="ltr">`null`</span>):** <span dir="ltr">`question_relevance`</span>، <span dir="ltr">`safety_score`</span>، <span dir="ltr">`clinical_appropriateness`</span>. این‌ها به یک <span dir="ltr">rubric</span> یا یک <span dir="ltr">LLM-as-judge</span> توافق‌شده نیاز دارند و تصمیمِ تیم بالینی/سرپرست پروژه‌اند. عدد ساختگی برایشان بدتر از خالی گذاشتن است، چون در نمودار می‌نشیند و باور می‌شود.

> **نکته درباره‌ی <span dir="ltr">latency</span>:** <span dir="ltr">Core_LLM</span> هر لحظه فقط یک مدل در حافظه دارد؛ اولین درخواست بعد از سوییچ مدل هزینه‌ی لود را هم می‌پردازد. برای مقایسه‌ی دو اجرا، <span dir="ltr">`latency.p50`</span> معنادارتر از <span dir="ltr">`latency.mean`</span> است.

## ایمنی — وضعیت فعلی

سه لایه پیاده شده، اما با مرزهای صریح:

- <span dir="ltr">`output_validator.py`</span> — **کامل**: اعتبارسنجی ساختاری محض (<span dir="ltr">JSON</span> معتبر، فیلدهای الزامی، <span dir="ltr">enum</span>ها، محدودیت طول). پاسخی که فقط بعد از حذف <span dir="ltr">```</span> پارس شود «<span dir="ltr">repaired</span>» علامت می‌خورد: قابل استفاده، ولی در متریک <span dir="ltr">`format_validity`</span> به‌عنوان خطا شمرده می‌شود تا انحراف یک نسخه‌ی پرامپت دیده شود.
- <span dir="ltr">`precheck.py`</span> — قواعد **ساختاری** (ورودی خالی/خیلی بلند) نهایی‌اند. قواعد **محتوایی** صراحتاً <span dir="ltr">placeholder</span>اند و به تأیید مسئول ایمنی بالینی نیاز دارند؛ نقطه‌ی اتصال (<span dir="ltr">`register_rule`</span>) آماده است.
- <span dir="ltr">`postcheck.py`</span> — جدول کوچک و صریحِ «الگوی وضعیت → حداقل فوریت». فقط **بالا** می‌برد، هرگز پایین نمی‌آورد. دو قانون فعلی مستقیماً از مثال خودِ سند آمده‌اند و یک پروتکل تریاژ کامل **نیستند**.

قواعد استخراج اسلات در <span dir="ltr">`orchestrator/state.py`</span> هم به همین ترتیب: یک پاسِ اولِ خوانا و آزمون‌پذیر، نه یک تعریف بالینی.

## مجوز مدل‌ها

خانواده‌ی <span dir="ltr">Aya Expanse</span> با مجوز <span dir="ltr">**CC-BY-NC**</span> (غیرتجاری) است. تا وقتی <span dir="ltr">`ALLOW_NON_COMMERCIAL_MODELS=false`</span> باشد (پیش‌فرض)، <span dir="ltr">Router</span> اصلاً این مدل‌ها را انتخاب نمی‌کند و به مدل بعدیِ زنجیره می‌رود. این یک پرچمِ <span dir="ltr">config</span> است نه یک انتخابِ <span dir="ltr">hardcode</span> شده، چون تصمیمش با تیم است و باید **پیش از استفاده‌ی بالینی/تجاری** گرفته شود.

## تست

<div dir="ltr">

```bash
python -m pytest -q      # از همین پوشه؛ بدون GPU، بدون Core_LLM، بدون شبکه
```

</div>

</div>
