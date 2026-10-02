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

## Phase 9 — Security, Human Approval & Permission Policy Hardening

تمت إضافة طبقة تحكم أمنية حتمية حول التنفيذ، مع إبقاء الـLLM والـPlanner في دور الاقتراح فقط:

```text
Goal
→ Planner / LLM Proposal
→ Policy Evaluation
→ Risk Classification
→ Approval Decision
→ Execution-Boundary Policy Check
→ Executor
→ Observation
→ Verification
→ Audit
```

### Policy model

يستخدم `PolicyEngine` قدرات صريحة قابلة للتوسعة، منها:

```text
READ_WORKSPACE
WRITE_WORKSPACE
NETWORK_ACCESS
EXTERNAL_API
DATABASE_READ
DATABASE_WRITE
SENSITIVE_DATA_ACCESS
SYSTEM_COMMAND
LONG_RUNNING_TASK
SKILL_ACTIVATION
SKILL_UPDATE
```

ويصنف الخطر deterministic إلى:

```text
LOW → MEDIUM → HIGH → CRITICAL
```

ويُرجع قراراً منظماً فقط:

```text
ALLOW | DENY | REQUIRE_APPROVAL | QUARANTINE
```

لا يمكن لنص LLM مثل `permission approved` أن يغير القرار، ولا تمنح Phase 8 `APPROVED` للمهارة صلاحيات invocation تلقائياً.

يُنشئ التطبيق `SecurityController` افتراضياً. وبما أن `WebResearchTool` ينفذ HTTP خارجياً، فهو يعلن `NETWORK_ACCESS` ويُصنف `HIGH`؛ لذلك يعيد التشغيل طلب موافقة بدلاً من تنفيذ الشبكة تلقائياً. هذا السلوك مقصود، ويمكن لطبقة تكامل موثوقة استدعاء `ApprovalGate.approve()` بعد عرض payload الكامل للمستخدم.

### Human approval model

`ApprovalGate` يفرض موافقة صريحة مرتبطة بالضبط بـ:

- `task_id`
- `action_id`
- `tool_or_skill`
- capabilities
- input fingerprint
- risk level
- policy version

الموافقات تمر بالحالات:

```text
PENDING → APPROVED
PENDING → DENIED
PENDING → CANCELLED
PENDING / APPROVED → EXPIRED
```

ولا تُعتبر الموافقة القديمة صالحة إذا تغيرت هوية الإجراء أو capabilities أو policy version. لا يتم تفسير الصمت أو timeout أو موافقة مهمة أخرى على أنها موافقة.

### Execution boundary and audit

يُعاد فحص السياسة عند الحد التنفيذي الفعلي داخل `GenericExecutor`، وليس في Planner فقط. كما يتم إعادة فحص استدعاء المهارة داخل `SkillLifecycleManager`، وتخضع أدوات recovery/replanning لنفس البوابة.

يحفظ SQLite:

```text
security_policies
approval_requests
security_decisions
audit_events
```

ويتضمن سجل التدقيق قرارات السياسة، طلبات الموافقة، المنع، التنفيذ، التفعيل، quarantine، rollback، وتغيير policy version، مع تمرير metadata عبر sanitization قبل الحفظ.

لا يثبت ذلك mathematically proven security أو perfect sandbox؛ العزل القوي للشبكة والملفات غير متاح في Sandbox الحالية، ولذلك تبقى القدرات التي لا يمكن عزلها تحت `QUARANTINE` أو `REQUIRE_APPROVAL` وفق السياسة.

## الاختبارات

```bash
pytest -q
```

تغطي الاختبارات الاستجابة المنظمة، أخطاء JSON، 429 و5xx، إخفاء مفتاح API، التحقق من الخطط، التنفيذ الديناميكي، Observation/Verification، diagnosis/recovery/replanning، Experience Retrieval، Reflection patterns/confidence، LearningEngine، التعلم التزايدي، الأدلة المتعارضة والسلبية، Phase 8 AST/contract/mutation/sandbox/verification/approval/versioning/rollback، Phase 9 capability/risk/policy/approval binding/expiry/denial/cancellation/execution-boundary/audit/sanitization/planner-bypass/recovery-security/learning-security/Phase 8 integration، الاستمرارية عبر إعادة التشغيل، والإثباتات السلوكية Candidate A/B/C وحماية الأسرار.
