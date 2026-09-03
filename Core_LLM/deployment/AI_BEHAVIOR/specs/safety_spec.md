# Safety Spec
جریان: Pre-check -> LLM -> Post-check
Pre-check ورودی را با red_flags.yaml مقایسه می‌کند و در صورت تطابق،
مستقیم به مسیر emergency می‌رود.