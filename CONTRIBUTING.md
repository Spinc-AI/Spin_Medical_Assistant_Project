<div dir="rtl">

# راهنمای مشارکت (<span dir="ltr">Contributing</span>)

شیوه‌ی کار ما روی پروژه‌ی دستیار پزشکی اسپین. پیش از اولین کامیت، این فایل را
بخوانید. راهنمای کامل و دستور‌به‌دستور <span dir="ltr">Git</span> در
[Module_Speech_To_Text/README.md](Module_Speech_To_Text/README.md) قرار دارد؛ این
فایل، کتابچه‌ی کوتاهِ قواعدِ سراسریِ مخزن است.

## ماژول‌ها

- **هر قابلیت جدید = یک پوشه‌ی <span dir="ltr">`Module_<Name>`</span> در ریشه‌ی مخزن.**
  هم‌سبک با نمونه‌ی موجود (<span dir="ltr">`Module_Speech_To_Text`</span>).
- **هر ماژول خودکفاست:** پوشه‌ی <span dir="ltr">`app/`</span> (یا سورس) خودش،
  <span dir="ltr">`requirements.txt`</span>، <span dir="ltr">`.gitignore`</span> و
  <span dir="ltr">`README.md`</span> مخصوص خودش.
- **ماژول‌ها فقط از طریق <span dir="ltr">API</span> با هم ارتباط می‌گیرند.** هرگز به
  درون ماژول دیگر دسترسی مستقیم (<span dir="ltr">import</span>) نداشته باشید. اگر به
  چیزی از جزء دیگری نیاز دارید، <span dir="ltr">endpoint</span> آن را صدا بزنید.
  قرارداد <span dir="ltr">API</span> را _پیش از_ نوشتن کدِ وابسته به آن، توافق کنید.

## شاخه‌بندی (<span dir="ltr">Branching</span>)

> **به‌روزرسانی:** پروژه دیگر یک شاخه‌ی جدا به‌ازای هر ماژول ندارد. همه‌ی
> ماژول‌ها (<span dir="ltr">STT</span>، <span dir="ltr">Core_LLM</span>،
> <span dir="ltr">Orchestrator</span>، <span dir="ltr">demo_app</span>، …) اکنون
> کنار هم، مستقیماً روی <span dir="ltr">`main`</span> هستند — <span dir="ltr">`main`</span>
> دیگر محافظت‌شده نیست و کار روزمره مستقیماً روی آن کامیت/پوش می‌شود. شاخه‌های
> قدیمیِ تک‌ماژولی (<span dir="ltr">`STT`</span>، <span dir="ltr">`Core_LLM`</span>،
> <span dir="ltr">`Orchestrator`</span>، <span dir="ltr">`demo_app`</span>) هنوز
> روی <span dir="ltr">hamgit</span> موجودند اما منسوخ شده‌اند — کار جدید را روی
> آن‌ها ادامه ندهید.

- برای کار روی یک قابلیت بزرگ یا پرریسک، همچنان می‌توانید یک شاخه‌ی موقت بسازید
  و در پایان با <span dir="ltr">`main`</span> ادغام کنید:

<div dir="ltr">

```bash
git checkout main && git pull --rebase
git checkout -b <feature_name>
```

</div>

## ایشوها (<span dir="ltr">Issues</span>)

ایشوها همان تسک‌ها هستند: کارهایی که سرپرستِ پروژه تعریف و به اعضای تیم تخصیص
می‌دهد. هر کس روی تسک‌های واگذارشده‌ی خودش کار می‌کند.

## کامیت‌ها

هر کامیت پیامی کوتاه و روشن داشته باشد که بگوید چه تغییری داده‌اید. تغییرات
کوچک/ایزوله را می‌توان مستقیم روی <span dir="ltr">`main`</span> کامیت کرد؛ برای
تغییرات بزرگ یا پرریسک، از یک شاخه‌ی موقت (بالا) استفاده کنید.

## هرگز کامیت نکنید

- محیط‌های مجازی (<span dir="ltr">`.venv/`</span>، <span dir="ltr">`venv/`</span>)
- فایل‌های راز/محیطی (<span dir="ltr">`.env`</span>، کلیدها) — به‌جایش
  <span dir="ltr">`.env.example`</span> را کامیت کنید
- وزن‌های مدل (<span dir="ltr">`*.pt`</span>، <span dir="ltr">`*.safetensors`</span>،
  <span dir="ltr">`*.bin`</span>، …)
- دیتاست‌ها یا فایل‌های صوتی حجیم، فراتر از نمونه‌های کوچکِ منتخب که به‌عنوان
  <span dir="ltr">fixture</span> نگه داشته می‌شوند
- **هیچ داده‌ی واقعی بیمار (<span dir="ltr">PHI</span>).**

<span dir="ltr">`.gitignore`</span>ِ هر ماژول موارد رایج را پوشش می‌دهد — همان‌جا
گسترشش دهید.

## زبان مستندات

فارسی (<span dir="ltr">`README.md`</span>، <span dir="ltr">`CONTRIBUTING.md`</span>)

</div>
