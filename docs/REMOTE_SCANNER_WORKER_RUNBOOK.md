# Vulnora Remote Scanner Worker Runbook

## الحالة الحالية

هذا الملف يصف الحد الأدنى المطلوب لمضيف Linux مستقبلي فقط. لا ينشئ خادمًا، ولا يثبت حزمًا، ولا يفعّل scanner. يبقى Android/Termux control plane محليًا، وتبقى الماسحات النشطة `UNAVAILABLE` إلى أن ينجح إثبات runtime الكامل.

الحالة الحالية للكود لا تحتوي بروتوكول remote-worker آمنًا ومكتملًا يسمح لعامل بعيد بالوصول المباشر إلى SQLite أو تنفيذ jobs عبر API عامة. لذلك لا يجوز تشغيل Worker بعيد بوضع مشاركة ملف SQLite أو عبر SSH command relay أو endpoint غير موثق. يلزم أولًا تنفيذ بروتوكول jobs مستقل، مصادق عليه، lease-bound، ومراجع أمنيًا.

## لماذا لا يكفي Termux

`proot`، وPython subprocess، و`unshare` وحدها ليست sandbox. لا تضمن هذه المسارات filesystem isolation أو network egress policy أو process cleanup أو منع الوصول إلى مفاتيح Backend وقاعدة البيانات. عند اكتشاف `TERMUX_VERSION` أو مسار Termux، يعرض preflight الآن check باسم `host_platform` وحالة `FAIL` (`android_termux_scanner_runtime_unverified`)؛ لا يوجد مسار تثبيت أو fallback يتجاوز ذلك.

## مواصفات المضيف الدنيا

يجب أن يكون المضيف Linux حقيقيًا ومدعومًا، وليس Android/Termux:

- معمارية `x86_64` أو `aarch64` مع Linux kernel يدعم user namespaces وcgroups v2 وseccomp.
- حساب خدمة غير root مع rootless Podman مثبت ومتحقق الإصدار.
- `newuidmap` و`newgidmap` مع نطاقات صحيحة في `/etc/subuid` و`/etc/subgid`.
- `slirp4netns` أو `pasta`، مع network namespace قابلة للاختبار وسياسة egress صريحة لكل job.
- cgroup controllers فعالة على الأقل لـ`cpu` و`memory` و`pids`.
- seccomp profile قابل للإثبات داخل runtime، filesystem root للـscanner read-only، mounts محددة allowlist، و`/tmp` مؤقت محدود.
- قدرة موثوقة على قتل process group وchild processes عند timeout أو cancellation.
- ذاكرة عملية لا تقل عن 4 GiB، و2 vCPU، ومساحة حرة لا تقل عن 20 GiB للمضيف التجريبي؛ هذه أرقام تشغيلية دنيا مقترحة وليست إثباتًا للعزل.
- لا تُتاح للمهمة ملفات SQLite أو `.env` أو Owner cookies أو مفاتيح Gemini أو socket الخاص بـPodman.
- image scanner immutable ومثبتة بـdigest، وcommand manifest ومصدر binary موثقان بتوقيع/تحقق خارج مسار المستخدم.

توفر هذه العناصر لا يكفي وحده. يجب أن تنجح اختبارات القبول الفعلية، وإلا تبقى النتيجة `SANDBOX_UNAVAILABLE`.

## التسلسل القابل لإعادة الإنتاج على مضيف مدعوم

1. أنشئ مستخدم خدمة غير root مخصصًا للعامل، ولا تشغله من حساب Owner أو من جلسة تملك أسرار التطبيق.
2. ثبّت Podman وrootless helpers وnetwork helper من مستودع Linux موثوق، ثم تحقق من الإصدارات والتواقيع وفق سياسة المضيف. لا تستخدم `proot` كبديل.
3. شغّل preflight من النسخة المثبتة:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 python -m private_agent.cli runtime-preflight --json
   ```

   يجب أن تكون checks الخاصة بالمضيف `PASS` أو `REQUIRED_ACCEPTANCE` حسب العقد. أي `FAIL` يوقف المسار.

4. جهّز image قبول محلية مثبتة بـdigest وmanifest command ثابت. لا تسحب image أو binary غير مثبت الهوية أثناء job.
5. شغّل harness القبول الموجود:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 python scripts/phase28_runtime_acceptance.py --image <digest-pinned-image>
   ```

   لا تعتبر نجاح بدء container إثباتًا كافيًا. يجب أن تثبت النتائج filesystem restrictions وnetwork policy وresource limits وprocess cleanup وimage/manifest identity.

6. افحص أن كل نتيجة قبول مرتبطة بـruntime identity وimage digest وcommand manifest digest. أي تغيير في أحدها يبطل الربط ويعيد الحالة إلى unavailable.
7. لا تُصدر `RuntimeActivationProof` من env variable أو boolean أو اسم runtime أو response يرسله العميل. الإصدار يجب أن يأتي من adapter موثوق بعد نجاح جميع checks.
8. بعد تنفيذ بروتوكول remote jobs مستقل ومراجع، يجب أن يمر كل job عبر target allowlist وscope وapproval وlease fencing، مع سجل sanitized للـtool version والـruntime identity والـdigests والنتيجة.

## بروتوكول remote worker المطلوب قبل أي تفعيل

هذا البروتوكول غير منفذ بعد، ولذلك لا تستخدم هذه الوثيقة كأنها endpoint جاهز:

- قناة مصادق عليها بين الهاتف وworker، مع هوية worker قابلة للدوران وعدم استخدام Bearer legacy كبديل عن جلسة Owner.
- job envelope موقّع/مربوط بـ`task_id` و`stage_id` و`lease_id` وscope digest وruntime proof digest.
- العامل لا يستقبل raw Owner credentials أو Gemini key، ولا يقرأ SQLite الهاتف مباشرة.
- كل job يحدد tool identity وimage digest وcommand manifest digest وbudgets وegress policy.
- heartbeat وlease fencing و`EXECUTION_UNKNOWN` عند فقدان الاتصال؛ لا retry تلقائي لعملية غير معلومة.
- النتائج bounded وsanitized، ولا تتحول observation إلى finding confirmed دون Evidence Gate ومراجعة Owner.
- إلغاء job يقتل process group داخل runtime، وتثبت acceptance أن child processes لا تبقى بعد الإلغاء.

## معيار الإيقاف

يجب إبقاء `httpx` و`nuclei` وعمليات active target في `UNAVAILABLE` إذا تحقق أي مما يلي:

- الهاتف هو Android/Termux.
- Podman أو rootless helpers أو network policy غير متاحة.
- image أو command manifest غير مثبتَي الهوية.
- filesystem/network/process/resource acceptance لم تنجح.
- لا توجد قناة remote-worker مصادق عليها وlease-bound.
- لا يوجد scope مكتوب ومؤكد وقرار Backend/Owner مناسب.

## ما يمكن تشغيله الآن على الهاتف

يمكن تشغيل Dashboard وFastAPI وSQLite وWorker المحلي والمراحل deterministic/local evidence وpublic passive allowlist وفق عقودها الحالية. لا يعني ذلك أن الهاتف يملك sandbox scanner، ولا أن public policy retrieval يمثل authorization، ولا أن نجاح اختبارات fixture يثبت فحص HackerOne حيًا.
