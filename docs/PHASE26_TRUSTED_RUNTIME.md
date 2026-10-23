# Phase 26 — Trusted Runtime, DNS-Pinned Transport & Isolated Execution

## النتيجة

هذه المرحلة **PARTIAL/BLOCKED** للتشغيل النشط الخارجي. أضيفت بوابات تمنع الادعاء بأن binary أو sandbox موثوقان، لكن بيئة التنفيذ الحالية لا تحتوي runtime عزل OS موثوقًا يمكنه تشغيل scanner بأمان.

## ما نُفّذ

- `trusted_tools.py` يعرّف `TrustedToolInstallSpec` ثابتًا ومقيدًا لـProjectDiscovery `httpx` و`nuclei`.
- يجب أن يطابق release URL القناة الرسمية وإصدارًا مثبتًا واسم asset، وأن تكون ملفات archive/checksum/signature/keyring وbinary ملفات regular مملوكة للمستخدم وغير قابلة للكتابة لغيره.
- لا يكفي version string أو SHA-256 يقدمه caller. التحقق يتطلب `gpgv` النظامي، checksum file موقّعًا، و`VALIDSIG` مطابقًا لـ`*_SIGNER_FINGERPRINT` موفرًا من مصدر موثوق خارج الوكيل، ثم يطابق checksum الـarchive ويطابق محتوى binary عضو archive.
- لا يحدث download تلقائي ولا تشغيل binary أثناء التحقق. الإعداد البيئي الناقص يعيد `None`، وأي فشل يعيد `UNAVAILABLE`.
- `build_orchestrator()` يمرر metadata provenance الكاملة، لكنه يبقي `external_sandbox_available=False` عمدًا؛ أدوات `httpx` و`nuclei` المسجلة تعيد `scanner_sandbox_unavailable` قبل أي parsing أو subprocess.
- تم تشديد DNS transport في `research.py`: كل الإجابات IPv4/IPv6 تُفحص، mixed allowed/blocked تُرفض، ويعاد فحص العنوان عند فتح الاتصال؛ لا يوجد fallback عادي للـhostname.

## المصادر الرسمية المقيدة

- httpx install: <https://docs.projectdiscovery.io/opensource/httpx/install>
- httpx repository: <https://github.com/projectdiscovery/httpx>
- httpx release metadata: <https://api.github.com/repos/projectdiscovery/httpx/releases/tags/v1.12.0>
- nuclei repository: <https://github.com/projectdiscovery/nuclei>
- nuclei release metadata: <https://api.github.com/repos/projectdiscovery/nuclei/releases/tags/v3.11.1>

الـrelease assets الموثقة حاليًا هي `httpx_1.12.0_linux_amd64.zip` و`nuclei_3.11.1_linux_amd64.zip`. لا تعتبر هذه المراجع وحدها إثباتًا كافيًا؛ يجب حفظ checksum signature وfingerprint مستقلين والتحقق منهما محليًا.

## سبب الحجب الحالي

فحص البيئة وجد أن `podman`, `docker`, `bwrap`, و`firejail` غير متاحة. لذلك لا توجد حاليًا boundary OS-enforced تحقق CPU/memory/process/filesystem/network limits وتنظيف process tree. `SafeSubprocessRunner` ليس sandbox، ولا يُستخدم لتفعيل المسح النشط من API.

Fake executables في الاختبارات تثبت عقود parsing والtimeouts وauthorization فقط؛ لا تثبت تشغيل ProjectDiscovery الحقيقي ولا العزل الحقيقي. لا يوجد في هذه المرحلة اتصال بهدف إنتاجي أو external target.

## التفعيل المستقبلي الآمن

لا يكفي إضافة env flag. يجب أولًا توفير adapter داخلي يولد proof غير قابل للإنشاء من caller بعد:

1. تحقق signer fingerprint من مصدر ProjectDiscovery موثوق مستقلًا.
2. تحقق archive وbinary وtemplate manifest قبل التنفيذ.
3. تنفيذ scanner داخل runtime OCI rootless موثق، مع image digest وmanifest digest وسياسة network/filesystem/resources.
4. ربط DNS-pinned transport بحد التنفيذ الفعلي، أو إبقاء hostname targets `UNAVAILABLE`.
5. اختبار cancellation/process-tree cleanup، executable replacement، secret isolation، redirects، IPv4/IPv6، وEvidence Gate باستخدام تطبيق محلي فقط.

تغيير `external_sandbox_available` أو وضع signer fingerprint عشوائي في البيئة ليس تفعيلًا صالحًا.

## التحديث والإبطال

عند إصدار جديد: حدّث الإصدار وasset وrelease URL وarchive checksum وsignature وsigner fingerprint بعد مراجعة مستقلة، ثم أعد focused/full suites. عند سحب إصدار أو key: احذف spec/keyring المعنيين أو اعزل binary؛ لا تستخدم fallback إلى `latest` أو إلى host execution.
