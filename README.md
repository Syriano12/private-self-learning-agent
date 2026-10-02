# Private Self-Learning Agent

نواة وكيل ذكاء اصطناعي محلية ومحمولة، مصممة للعمل على Termux/Android مع قابلية النقل إلى Linux/VPS. الإصدار الحالي يركز على Web Research & Information Gathering، ويقدم CLI وFastAPI دون واجهة رسومية.

## الحالة الحالية

الإصدار `0.1.0` يتضمن طبقة LLM محايدة مع Gemini REST، ومخططاً منظماً مدفوعاً بـLLM مع تحقق حتمي للخطة، وسجل أدوات، وبحثاً متعدد المصادر، واستخراج نص، وتخزين SQLite منفصلاً للمهام والتجارب والمعرفة، واسترجاع تجارب تشغيلية حتمي محدود السياق، ومحرك Reflection يستخرج أنماطاً منظمة مدعومة بالأدلة، ومحرك LearningEngine يحول الـInsights إلى استراتيجيات منظمة قابلة للاسترجاع، وتحققاً من الأدلة، وتسجيل الفشل وإعادة المحاولة. عند غياب `GEMINI_API_KEY` يُستخدم مسار توافق محدود؛ ويمكن لهذا المسار استخدام الاستراتيجيات المتعلمة حتمياً مع بقاء الخطة خاضعة للتحقق والصلاحيات.

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

## Phase 7 — Real Self-Learning

تم إثبات التدفق السلوكي التالي:

```text
Experience → ReflectionInsight → LearnedStrategy → Future Planner Decision
```

`LearnedStrategy` هي بيانات منظمة فقط، وتحتوي على الشرط، والفعل المفضل، والفعل المتجنب، والأدلة، ومعرفات الخبرات والـInsights، والثقة، والحالة. تحفظ الاستراتيجيات في SQLite عبر abstraction مستقلة هي `LearningMemory`، ولا تنفذ أدوات أو Python أو Shell ولا تغير الصلاحيات. الاستراتيجيات منخفضة الثقة أو المتعارضة لا تدخل التخطيط النشط.

## Phase 8 — Safe Skill Learning وZero-Trust Skill Lifecycle

تمت إضافة دورة حياة المهارات التالية:

```text
Candidate
→ Contract Validation
→ AST Static Analysis
→ Bounded Mutation Testing
→ Capability-Aware Sandbox
→ Runtime Verification
→ Cross-Verification
→ Approval / Quarantine
→ Versioned Registry
→ Runtime Monitoring
→ Degradation Detection
→ Rollback
→ Experience
→ Reflection / Learning
```

المهارات المتعلمة تعامل ككود غير موثوق. لا يوجد ادعاء بإثبات رياضي للصحة؛ حالة `APPROVED` تعني فقط أن الفحوص المحددة في السياسة نجحت. Mutation testing يقيس قدرة الاختبارات على قتل mutations المحددة ولا يثبت غياب جميع الأخطاء.

في بيئة Sandbox الحالية تتوفر حدود العملية، timeout، resource limits، وتنظيف البيئة، بينما لا تتوفر عزل قوي أو قيود مستقلة على filesystem/network. لذلك تُوضع المهارات التي تعلن network أو filesystem capability في `QUARANTINED` بدلاً من تشغيلها بثقة زائفة.

## الاختبارات

```bash
pytest -q
```

تغطي الاختبارات الاستجابة المنظمة، أخطاء JSON، 429 و5xx، إخفاء مفتاح API، التحقق من الخطط، التنفيذ الديناميكي، Observation/Verification، diagnosis/recovery/replanning، Experience Retrieval، Reflection patterns/confidence، LearningEngine، التعلم التزايدي، الأدلة المتعارضة والسلبية، Phase 8 AST/contract/mutation/sandbox/verification/approval/versioning/rollback، الاستمرارية عبر إعادة التشغيل، والإثباتات السلوكية Candidate A/B/C وحماية الأسرار.
