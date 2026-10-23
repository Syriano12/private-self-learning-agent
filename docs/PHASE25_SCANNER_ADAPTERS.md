# Phase 25 — Controlled scanner adapters

## الهدف

تضيف هذه المرحلة حدًا تشغيليًا ضيقًا لربط ProjectDiscovery `httpx` و`nuclei` بطابور Agent Core وVulnora. ليست هذه المرحلة ماسحًا عامًا، ولا تجعل نتائج scanner findings مؤكدة تلقائيًا.

## التدفق

```text
Vulnora /api/v1/bug-bounty/tasks
  → typed scanner + explicit scope + target identity/resource
  → durable task queue
  → SecurityController / HIGH + NETWORK_ACCESS
  → owner approval
  → GenericExecutor
  → registered HttpxReconTool or NucleiScanTool
  → SafeSubprocessRunner (argv, no shell, bounded timeout/output)
  → normalized observations
  → Vulnora typed scan response
```

المسار deterministic القديم يبقى كما هو. اختيار `scanner=httpx` أو `scanner=nuclei` يغيّر الأداة المتخصصة فقط؛ لا يُسمح بتمرير حقول scanner إلى `bugbounty_analysis` القديم.

## الضوابط المنفذة

- الأدوات مسجلة في `ToolRegistry` وتعلن `NETWORK_ACCESS` و`HIGH`، لذلك لا يصل التنفيذ الفعلي دون إعادة تفويض وموافقة owner المرتبطة بالـtask/action/tool/input fingerprint/policy version.
- `SafeSubprocessRunner` يستخدم `shell=False`، `stdin=DEVNULL`، بيئة محدودة، timeout، سقفًا للخرج، وإيقافًا لمجموعة العملية على POSIX. لا يعيد محاولة process غير المحسوم.
- `ProjectDiscoveryProbe` يتحقق من executable قابل للتشغيل، version ثلاثي الأجزاء، ووجود flags المطلوبة. وجود اسم binary وحده لا يكفي.
- active adapters تقبل حاليًا IP literal IPv4 فقط. Hostname/DNS يفشل مغلقًا بـ`UNAVAILABLE` لأن transport يثبت socket address في طبقة HTTP غير متاح بعد.
- `ScopeDefinition` يفرض تطابق host/path/method/identity/resource/port ومدة الصلاحية. alternate ports تُرفض.
- `httpx` يخرج observations bounded فقط. redirect أو resolved address خارج النطاق يحول النتيجة إلى `INCOMPLETE`.
- `nuclei` يتطلب manifest JSON موثوقًا، digest SHA-256 معروفًا من configuration، وSHA-256 مستقلًا لكل template داخل manifest، مع paths غير قابلة للعبور وtags غير مدمرة. النتائج تُطبع `HYPOTHESIS`/observation فقط.
- API وVulnora يعيدان نتيجة scanner typed، ويعيدان فحص candidate findings قبل العرض. لا يوجد report confirmation لناتج scanner.

## الإعداد

```bash
export AGENT_HTTPX_BINARY=/absolute/path/to/httpx
export AGENT_NUCLEI_BINARY=/absolute/path/to/nuclei
export AGENT_NUCLEI_TEMPLATES_DIR=/absolute/path/to/trusted/templates
export AGENT_NUCLEI_MANIFEST_SHA256=<trusted-manifest-sha256>
```

إعدادات Nuclei الثلاثة الأخيرة مطلوبة فقط عند استخدام Nuclei. بعد Phase 26 أصبحت provenance الموقعة و`*_SIGNER_FINGERPRINT` وexternal sandbox المتحقق منها مطلوبة أيضًا؛ في البيئة الحالية تعود النتيجة `UNAVAILABLE`/`scanner_sandbox_unavailable` بدل fallback إلى host execution أو fixture غير معلن.

## ما تم اختباره

- fake ProjectDiscovery executables controlled داخل الاختبارات فقط.
- version/flag verification، missing runtime، malformed/empty JSONL، output cap، timeout، no-shell argument handling، DNS pinning boundary، alternate-port scope denial، redirect containment، trusted template tags/digest، وNuclei hypothesis normalization.
- executor integration: عدم التنفيذ قبل approval، التنفيذ بعد approval الصحيح، ورفض إعادة استخدام approval مع target مختلف.
- API E2E: POST → pending approval → approve → queue/worker → scanner quarantine → `FAILED` آمن، دون اتصال شبكة أو هدف خارجي. لا يُعد هذا دليلًا على تشغيل ProjectDiscovery الحقيقي.

## الحدود المتبقية

Phase 25 لا توفر DNS-pinned HTTP transport للـhostnames، ولا browser/WebSocket/GraphQL/business-logic engine، ولا credential/test-account management، ولا sandbox OCI حقيقيًا. Phase 26 شدّدت DNS mixed-answer checks وأضافت signed provenance gate، لكنها أبقت الأدوات المسجلة محجوبة لأن sandbox غير متاح. لا تُستخدم الأدوات إلا على أهداف تملك تفويضًا مكتوبًا لاختبارها، ولا يُستنتج من `COMPLETED` أن الهدف آمن أو أن scanner اكتشف كل الثغرات.
