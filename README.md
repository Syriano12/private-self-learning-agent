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

## Phase 10 — Durable Task State, Checkpoints & Safe Resume

تمت إضافة persistence للحالة التشغيلية عبر `TaskState` وواجهات `Store` transactional. دورة المهمة أصبحت:

```text
Goal
→ TASK_CREATED
→ PLANNING
→ PLAN_CREATED
→ READY
→ BEFORE_EXECUTION
→ EXECUTING
→ AFTER_EXECUTION
→ OBSERVING
→ AFTER_OBSERVATION
→ VERIFYING
→ AFTER_VERIFICATION
→ READY / NEXT ACTION
→ TASK_COMPLETED أو TASK_FAILED
```

حالات الانقطاع غير الآمن لا تُفسر على أنها نجاح:

```text
EXECUTING
→ EXECUTION_UNKNOWN
→ explicit observation / verification
→ READY أو BLOCKED
```

إذا كان من الممكن أن يكون side effect خارجياً قد حدث، فإن `resume_task(task_id)` يحظر التنفيذ افتراضياً. لا يُسمح بإعادة المحاولة إلا بعد تقديم قرار صريح موثق بأن الإجراء لم يحدث، ولا يُسمح بتحويل الحالة إلى `VERIFIED` إلا مع observation وverification evidence صالحين.

### Durable state model

تتضمن الحالة المحفوظة:

- task identity والهدف والحالة الحالية.
- current phase وcurrent step.
- plan version وplan fingerprint.
- state version وstate fingerprint.
- completed/pending/failed action identities.
- observations وverification results.
- recovery/replanning state.
- approval IDs وsecurity decisions.
- created/updated/checkpoint timestamps.
- آخر checkpoint ناجح.

يستخدم كل action هوية deterministic مشتقة من:

```text
plan_version + step_id + tool + canonical input
```

ولا يُعاد تنفيذ action مسجل كـ`VERIFIED` بعد restart. هذا يحمي الأدوات غير idempotent مثل network calls والكتابات وتغييرات filesystem/database على مستوى state machine، مع بقاء idempotency الفعلية لكل خدمة خارجية مسؤولية عقد الأداة.

### Atomic persistence

يحفظ SQLite الحالة في جداول:

```text
task_state_schema
task_states
task_checkpoints
task_action_records
```

يتم حفظ state snapshot وcheckpoint sequence وaction records وtask summary داخل transaction واحدة. لا يتم اعتبار action مكتملاً قبل حفظ observation والـverification بنتيجة `VERIFIED`. ولا تصبح المهمة `COMPLETED` قبل حفظ verification النهائية.

`state_version=1` معروف فقط حالياً. أي state version غير معروف، أو state JSON تالف، أو fingerprint غير مطابق، أو transition غير مسموح، يفشل مغلقاً ولا يُخمّن أو يُصلح تلقائياً. جدول `task_state_schema` يثبت إصدار مخطط persistence الحالي لاستراتيجية migrations مستقبلية صريحة.

### Resume and Phase 9

الاستئناف لا يتجاوز الأمن. قبل كل action مستأنف يتم:

1. تحميل policy الحالية من SQLite.
2. إعادة تقييم capabilities وrisk.
3. إعادة فحص approval binding وinput fingerprint.
4. إعادة فحص policy version وexpiration.
5. إعادة تمرير action عبر execution-boundary `SecurityController`.

تبقى approval `PENDING` pending بعد restart. الموافقة `APPROVED` لا تبقى صالحة إذا انتهت أو تغيرت هوية action أو capabilities أو policy version أو fingerprint. عند invalidation يتم تسجيل `APPROVAL_INVALIDATED` وإعادة الحظر/طلب موافقة جديدة.

### Audit events

تتكامل checkpoints مع AuditTrail وتشمل الأحداث:

```text
TASK_CREATED
CHECKPOINT_SAVED
TASK_RESUMED
RESUME_BLOCKED
EXECUTION_UNKNOWN
RECOVERY_REQUIRED
APPROVAL_REVALIDATED
APPROVAL_INVALIDATED
TASK_COMPLETED
TASK_FAILED
```

لا تُحفظ الأسرار أو raw sensitive payloads في سجل التدقيق؛ تمر البيانات عبر sanitization القائمة.

هذه المرحلة لا تنفذ Phase 12 public resume API، ولا Phase 13 UI، ولا Phase 14 WhatsApp integration. `resume_task` واجهة داخلية deterministic فقط.

## Phase 11 — Structured Observability & Monitoring

تمت إضافة طبقة Observability مستقلة لا تنفذ tools ولا تمنح approvals ولا تقرر permissions:

```text
Agent Runtime
→ StructuredEvent
→ SQLite Event Store
→ Metrics
→ Task Timeline
→ Diagnostics / Queries
```

### Event schema and ordering

كل `StructuredEvent` يحتوي على:

```text
event_id
event_type
timestamp
task_id
action_id
step_id
correlation_id
component
severity
schema_version
metadata
sequence
```

يُستخدم vocabulary مغلق deterministic بدلاً من free-form log strings. ويستخدم SQLite `sequence INTEGER PRIMARY KEY AUTOINCREMENT` لترتيب الأحداث؛ لذلك لا يعتمد timeline على تساوي wall-clock timestamps أو على insertion order غير الموثق.

الأحداث تشمل task/plan/action/tool/observation/verification/recovery/replan/approval/security/skill/checkpoint/memory/reflection/learning lifecycle، مثل:

```text
TASK_CREATED → PLAN_CREATED → ACTION_STARTED → TOOL_STARTED
→ TOOL_COMPLETED → OBSERVATION_COMPLETED → VERIFICATION_PASSED
→ CHECKPOINT_SAVED → TASK_COMPLETED
```

ويستمر نفس `task_id` و`correlation_id` بعد restart/resume.

### Metrics and timing

يوفر `ObservabilityMonitor` واجهات داخلية:

```python
get_task_timeline(task_id)
get_recent_events(limit)
get_task_metrics(task_id)
get_system_metrics()
get_failures(task_id=None)
get_security_events(task_id=None)
get_tool_events(task_id=None)
```

تشمل metrics counts للمهام/actions/verification/recovery/replan/approvals/tool failures/unknown execution، إضافة إلى planning/action/tool/observation/verification/recovery/task durations. تُقاس durations باستخدام `time.monotonic()`، ولا يُستنتج منها أي ادعاء عن intelligence؛ هي قياسات تشغيلية وتشخيصية فقط.

### Persistence and degraded observability

تُحفظ الأحداث في نفس SQLite عبر:

```text
observability_events
```

مع uniqueness على `event_id` وtransactional sequence. إعادة كتابة نفس event ID بالمحتوى نفسه idempotent، أما conflict فيُرفض. لا توجد قاعدة بيانات telemetry ثانية.

السلوك الافتراضي عند فشل كتابة observability هو `degraded`:

- تُسجل العملية في عداد `degraded_writes` واسم الخطأ فقط في الذاكرة.
- لا تتغير policy أو approval أو execution.
- لا يُعتبر action ناجحاً بسبب غياب telemetry.
- لا تُفسد task state ولا تُتجاوز security boundary.

يوجد `fail_mode="raise"` للاختبارات أو البيئات التي تريد جعل فشل telemetry مرئياً صراحة، لكنه ليس الوضع الافتراضي.

### Sensitive-data protection

يمر metadata عبر redaction deterministic قبل persistence. تُحجب مفاتيح مثل:

```text
api_key, password, token, authorization, credentials
```

وتُختصر raw `input`/`output`/`payload` إلى النوع وSHA-256، بينما تُنقح Bearer/API-key patterns وتُختصر النصوص الكبيرة. لا يتم تسجيل complete tool inputs/outputs أو authorization headers في observability.

### Security and Phase 8 integration

SecurityController يبقى السلطة الوحيدة. أحداث:

```text
SECURITY_ALLOW
SECURITY_DENY
SECURITY_REQUIRE_APPROVAL
SECURITY_QUARANTINE
APPROVAL_REQUESTED
APPROVAL_APPROVED
APPROVAL_DENIED
APPROVAL_EXPIRED
APPROVAL_INVALIDATED
```

هي visibility facts فقط؛ لا يفسرها monitor كموافقة ولا يطلق execution منها. كما تُظهر Store أحداث skill candidate/activation/quarantine/rollback/registry دون تعديل isolation أو approval requirements في Phase 8.

تظل security/audit records منفصلة دلالياً عن telemetry العادية، حتى مع استخدام SQLite نفسه، ولا تطبق Phase 11 retention deletion قد يحذف evidence أمنية.

هذه المرحلة لا تنفذ Web UI أو PWA أو WhatsApp أو strong sandbox أو billing أو multi-tenant architecture أو LLM provider جديد.

## Phase 12 — Stable Agent API

تمت إضافة طبقة API رقيقة في `private_agent.api` فوق الـAgent Core الموجود. المسار التشغيلي هو:

```text
HTTP Request
→ request id + typed validation
→ Bearer authentication
→ single-owner authorization
→ AgentAPIService
→ Orchestrator / Store / SecurityController / ObservabilityMonitor
```

لا تنشئ routes Planner أو Executor أو Security أو persistence أو telemetry جديدة، ولا تنقل business logic إلى HTTP handlers.

### Authentication and authorization

تتطلب كل مسارات `/api/v1/...` و`POST /tasks/run`:

```http
Authorization: Bearer <AGENT_API_TOKEN>
```

يُقرأ `AGENT_API_TOKEN` من environment/configuration فقط. لا يوجد token افتراضي، ولا bypass لقيم مثل `True` أو `admin` أو `secret` أو `test`. الطلب المفقود أو malformed أو غير الصحيح يعيد `401`، وغياب الإعداد يجعل readiness `503`.

المصادقة منفصلة عن authorization. Phase 12 هو single-owner فقط؛ لا يضيف fake multi-tenancy. بعد المصادقة يستطيع المالك قراءة موارده المحلية، لكن `SecurityController` يظل السلطة الوحيدة للـcapabilities/risk/approval/quarantine، ويُعاد فحص security عند execution boundary كما في Phase 9.

### Endpoints

| Method | Endpoint | الوظيفة |
|---|---|---|
| GET | `/health` | liveness رخيص وغير مصادق عليه |
| GET | `/ready` | فحص auth configuration وSQLite والجداول المطلوبة؛ يعيد `503` عند الفشل |
| POST | `/api/v1/tasks` | تشغيل هدف عبر `Orchestrator` وإرجاع persisted task summary |
| GET | `/api/v1/tasks/{task_id}` | قراءة typed safe task state |
| POST | `/api/v1/tasks/{task_id}/resume` | استدعاء `Orchestrator.resume_task` فقط |
| GET | `/api/v1/tasks/{task_id}/timeline` | timeline مرتب بـSQLite sequence |
| GET | `/api/v1/tasks/{task_id}/metrics` | metrics للمهمة |
| GET | `/api/v1/approvals` | pending approvals افتراضياً؛ `pending_only=false` لعرض الحالات الأخرى |
| GET | `/api/v1/approvals/{approval_id}` | قراءة approval binding بصورة آمنة |
| POST | `/api/v1/approvals/{approval_id}/approve` | موافقة على request الموجود فقط |
| POST | `/api/v1/approvals/{approval_id}/deny` | رفض request الموجود فقط |
| GET | `/api/v1/metrics` | system metrics |
| GET | `/api/v1/failures` | failure events، مع `task_id` اختياري |
| GET | `/api/v1/security-events` | policy/approval/security visibility، مع `task_id` اختياري |
| GET | `/api/v1/tool-events` | tool lifecycle events، مع `task_id` اختياري |
| GET | `/api/v1/tools` | وصف typed للـtools المتاحة |

لا يوجد endpoint cancel في هذه المرحلة: Core لا يقدم cancellation آمنًا قابلاً للحفظ والاستئناف، ولذلك لا يتم fake support له. ولا يوجد endpoint لتفعيل skill أو تجاوز quarantine.

إنشاء task synchronous لأن الـCore الحالي يقدم `Orchestrator.run` متزامناً ولا يقدم queue/worker API. الاستجابة لا تفترض completion: status وphase مأخوذان من TaskState المحفوظ فعلياً؛ high-risk task يعود `WAITING_APPROVAL` بدلاً من تنفيذ الشبكة تلقائياً.

### Typed responses and redaction

الاستجابات تستخدم Pydantic models ولا تعيد SQLite rows أو `TaskState` الخام. Task state يعرض status/phase/step/plan version/action summaries/recovery/verification/checkpoint/approval state والتواريخ، لكنه لا يعرض plan inputs أو tool outputs أو authorization headers أو secrets.

كل خطأ له الشكل:

```json
{
  "error_code": "task_not_found",
  "message": "Task was not found",
  "request_id": "...",
  "task_id": "...",
  "approval_id": "",
  "api_version": "v1",
  "schema_version": 1
}
```

تُستخدم `400` للمعرّفات/headers غير الصالحة، `401` للمصادقة، `404` للموارد غير الموجودة، `409` لتعارض الحالة/Idempotency/approval، `422` لفشل schema validation، `500` لخطأ داخلي برسالة عامة، و`503` لعدم الجاهزية. لا تُعاد stack traces أو environment variables أو raw exceptions.

### Resume and approvals

`POST /api/v1/tasks/{task_id}/resume` يمرر الطلب إلى `resume_task` الموجود. لا يوجد retry بديل في API. حالة `EXECUTION_UNKNOWN` تبقى محظورة حتى observation/verification الصريحين وفق Phase 10، مع security/approval revalidation عند التنفيذ.

Approval endpoints لا تقبل تعديل `task_id` أو `action_id` أو capabilities أو risk أو fingerprint أو policy version أو expiry. جسم approve/deny لا يحتوي binding fields؛ القرار يطبق على الطلب المحفوظ فقط، ثم يبقى execution-boundary check إلزامياً.

### Idempotency and concurrency

- `POST /api/v1/tasks` يدعم `Idempotency-Key` اختيارياً، ويخزن fingerprint وtask ID في جدول `api_idempotency_keys`. إعادة نفس key ونفس goal لا تنفذ مرة ثانية؛ إعادة استخدامه مع goal مختلف تعيد `409`.
- approve/deny لا يعيدان تغيير approval غير `PENDING`، والإعادة الآمنة تعيد `409` بدلاً من تنفيذ شيء إضافي.
- resume محمي بـlock مشترك داخل `AgentAPIService`؛ طلبات resume/approval المتزامنة تُسلسل ولا تضيف retry عام. Phase 12 يستهدف process API واحداً في single-owner deployment؛ SQLite يبقى مصدر persistence عند restart.
- كل response يحمل `X-Request-ID`، ويُستخدم نفس identifier كـPhase 11 `correlation_id` في API request events. لا تُحفظ credentials في event metadata.

### API documentation and limits

FastAPI يوفر OpenAPI/Swagger على `/docs` و`/openapi.json`. هذا API contract منفصل عن internal commit/phase versions، ويعرض `api_version` و`schema_version` في الاستجابات.

هذه المرحلة لا تضيف async job queue، cancellation، public UI/PWA، WhatsApp، payment/billing، multi-tenancy، commercial licensing، new LLM provider، scanners، exploitation، أو strong sandbox.

## الاختبارات

```bash
pytest -q
```

تغطي الاختبارات الاستجابة المنظمة، أخطاء JSON، 429 و5xx، إخفاء مفتاح API، التحقق من الخطط، التنفيذ الديناميكي، Observation/Verification، diagnosis/recovery/replanning، Experience Retrieval، Reflection patterns/confidence، LearningEngine، التعلم التزايدي، الأدلة المتعارضة والسلبية، Phase 8 AST/contract/mutation/sandbox/verification/approval/versioning/rollback، Phase 9 capability/risk/policy/approval binding/expiry/denial/cancellation/execution-boundary/audit/sanitization/planner-bypass/recovery-security/learning-security/Phase 8 integration، Phase 10 task creation/checkpoints/normal resume/crash-like interruption/EXECUTION_UNKNOWN/duplicate prevention/idempotency/stale-corruption-version failures/approval persistence and revalidation/policy and capability invalidation/action fingerprint/recovery resume/atomic snapshots/audit lifecycle، Phase 11 structured schema/validation/ordering/correlation/timeline/metrics/duration/security-approval/tool-verification/recovery-memory-reflection-learning-skill/redaction/persistence/resume continuity/degraded storage/duplicate protection/authority invariants، Phase 12 authentication/validation/typed task state/idempotency/resume unknown/approval binding/expiry/deny/security boundary/timeline ordering/metrics/request correlation/redaction/restart/concurrency/readiness/error model، الاستمرارية عبر إعادة التشغيل، والإثباتات السلوكية Candidate A/B/C وحماية الأسرار.
