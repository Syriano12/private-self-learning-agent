# Private Self-Learning Agent

نواة وكيل ذكاء اصطناعي محلية ومحمولة، مصممة للعمل على Termux/Android مع قابلية النقل إلى Linux/VPS. الإصدار الحالي يركز على Web Research & Information Gathering، ويقدم CLI وFastAPI دون واجهة رسومية.

## الحالة الحالية

الإصدار `0.1.0` يتضمن طبقة LLM محايدة مع Gemini REST، ومخططاً منظماً مدفوعاً بـLLM مع تحقق حتمي للخطة، وسجل أدوات، وبحثاً متعدد المصادر، واستخراج نص، وتخزين SQLite منفصلاً للمهام والتجارب والمعرفة، واسترجاع تجارب تشغيلية حتمي محدود السياق، ومحرك Reflection يستخرج أنماطاً منظمة مدعومة بالأدلة، وتحققاً من الأدلة، وتسجيل الفشل وإعادة المحاولة. عند غياب `GEMINI_API_KEY` يُستخدم مسار توافق محدود ولا يُعد تخطيطاً ديناميكياً مدفوعاً بـLLM.

## التثبيت

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

في Termux يمكن تثبيت Python من مستودع Termux ثم استخدام البيئة الافتراضية. لا توجد حاجة إلى Docker أو systemd في هذا الإصدار.

## التشغيل

```bash
private-agent run "ابحث عن 5 مشاريع مفتوحة المصدر يمكن استخدامها لبناء AI Agent محلي، وقارن بينها من حيث الترخيص واللغة والمتطلبات والذاكرة وقابلية التشغيل على Android/Termux"
private-agent serve --host 127.0.0.1 --port 8000
```

واجهة API: `GET /health`، و`GET /tools`، و`POST /tasks/run` مع جسم JSON مثل `{"goal":"..."}`.

## متغيرات البيئة

`AGENT_DB_PATH` يحدد قاعدة البيانات، و`AGENT_HTTP_TIMEOUT` مهلة HTTP، و`AGENT_MAX_ATTEMPTS` عدد محاولات التعافي. يستخدم المخطط `AGENT_LLM_PROVIDER` و`GEMINI_MODEL` و`AGENT_LLM_TIMEOUT` وإعدادات retry/backoff المقابلة. يجب توفير `GEMINI_API_KEY` محلياً فقط، ولا تُحفظ الأسرار داخل المصدر.

## حدود الإصدار

لم تُضف بعد واجهة WhatsApp أو المتصفح التفاعلي أو تنفيذ أوامر نظام عامة أو تشغيل كود غير موثوق أو وظائف Bug Bounty. ستُضاف فقط بعد تثبيت بوابات الأمان والموافقة والاختبارات المستقلة.

## الاختبارات

```bash
pytest -q
```

تغطي الاختبارات الاستجابة المنظمة، أخطاء JSON، 429 و5xx، إخفاء مفتاح API، التحقق من الخطط، التنفيذ الديناميكي، Observation/Verification، diagnosis/recovery/replanning، Experience Retrieval، Reflection patterns/confidence، وحماية الأسرار. هذه الطبقات ليست Self-Learning؛ لا توجد Automatic Strategy Updates في هذا الإصدار.
