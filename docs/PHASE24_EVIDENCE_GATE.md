# Phase 24 — Evidence Gate and Conservative Findings

## الهدف

تمنع هذه المرحلة إعلان finding من فئة IDOR/BOLA على أنه `CONFIRMED` أو `REPORT_READY` ما لم يمر عبر تحقق دلالي قابل لإعادة الفحص. نجاح request وحده، أو تطابق status، أو heuristic الملكية لا يكفي.

## الضمانات المنفذة

- `Finding.transition()` يستدعي `validate_confirmed_evidence()` قبل حالات `CONFIRMED` و`REPORT_READY` و`HUMAN_REVIEWED`.
- التحقق يربط finding بالنطاق، هوية المالك وهوية الاختبار، resource وendpoint، URL، method، وقرار authorization الصريح `DENY`.
- يجب وجود baseline protected content وcandidate observed protected content ومقارنة response منظمة؛ الوصول إلى نفس المورد المحمي قد ينتج body متطابقًا، ولذلك لا تعتبر مساواة body وحدها سببًا للرفض أو القبول.
- evidence تحمل canonical digest. تغيير evidence بعد validation يبطل confirmation قبل التقرير أو persistence.
- `Finding.from_dict()` لا يثق بحالة persisted report؛ السجل المقروء الذي يفتقد validation أو يملك digest غير مطابق يُعاد إلى `HYPOTHESIS` مع `REJECTED` validation.
- `BugBountyRepository` يحفظ `tests_json` كـper-case status ledger مع migration additive آمنة للـdatabases القديمة.
- حدود persistence لا تعتبر `authorization_decision` credential؛ تحفظ فقط enum المسموح `ALLOW`/`DENY`/`UNKNOWN` بينما تستمر في إخفاء Bearer tokens والحقول السرية.
- يعيد API sanitizer فحص findings قبل إرجاعها، ويعيد حساب metrics من السجل المنظف، ويحجب التقرير النصي إذا خُفّض finding persisted إلى `HYPOTHESIS`.
- واجهة Vulnora تفكك finding/evidence/validation typed، وتخفض أي confirmation غير validated إلى `HYPOTHESIS` بدل عرضه كإثبات.
- ترفض الواجهة غياب `tests` ledger، وتحجب report عند وجود finding validation مرفوضة، وتعرض حالات الاختبار من backend فقط.

## ما لم يُنفذ

هذه البوابة، ضمن نطاق Phase 24، لا تثبت صحة نظام خارجي عام ولا تضيف HTTP scanner أو browser automation أو exploit execution. أضيفت لاحقًا في Phase 25 adapters منفصلة لـ`httpx`/`nuclei`، لكن مخرجاتها تبقى observations/hypotheses ولا تتجاوز هذه البوابة. أي `CONFIRMED` صالح هنا يعني أن contract evidence اجتاز الشروط، وليس اكتشافًا موثوقًا لـzero-day أو تصريحًا باختبار هدف خارجي.

## الاختبارات

اختبارات Phase 24 تغطي missing/unknown/ALLOW policy، identity/resource/request mismatch، asset/endpoint binding، status contradiction، غياب protected content، mutation بعد validation، validation metadata forgery، persisted forged `REPORT_READY`، metrics reconciliation، SQLite round-trip، policy decision redaction boundary، وحفظ per-case ledger. التحقق الكامل الأخير: backend `586 passed`، frontend `23 passed`، TypeScript typecheck، production build، `compileall`، و`git diff --check`.

## التشغيل الآمن

لا تستخدم active testing إلا على نطاق مكتوب ومصرح به. لا تغير scope أو identities أو evidence لإجبار confirmation؛ عند نقص الدليل يجب أن تبقى النتيجة hypothesis/inconclusive أو error.
