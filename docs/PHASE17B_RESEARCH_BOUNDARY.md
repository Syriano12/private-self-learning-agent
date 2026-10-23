# Phase 17B — Bounded Research Network Boundary

## ما تم تحسينه

`WebResearchTool` يمر الآن عبر حد HTTP محدود قبل أي طلب:

- HTTP وHTTPS فقط.
- رفض userinfo داخل URL.
- رفض localhost وmetadata hostnames وIP literals الخاصة أو loopback أو link-local أو multicast أو unspecified أو reserved.
- حل DNS والتحقق من جميع العناوين قبل الاتصال؛ فشل الحل يُعامل كفشل مغلق.
- transport الاتصال يعيد حل العنوان مباشرة قبل فتح socket، ويتصل بعنوان IP مثبت مع إبقاء hostname لـTLS/Host؛ إذا تبدل DNS إلى عنوان محظور يُرفض الاتصال.
- redirects يدوية ومحدودة إلى ثلاث قفزات، مع إعادة التحقق من كل host بعد كل redirect.
- response body تُقرأ streaming مع سقف bytes قبل تحليل HTML.
- query limit وfetch text limit وtimeout لها حدود صريحة.
- نتائج البحث العامة تعرض origin فقط ولا تعيد fetch URL الخام؛ رسائل الخطأ لا تعيد credentials أو query/path الخام، والنصوص القادمة من query/title/snippet/body تمر عبر redaction محدود للقيم السرية المعروفة.
- اختبارات MockTransport لا تستخدم شبكة حقيقية ولا أهدافًا خارجية.

## ما لا يعنيه هذا

هذه الضوابط لا تجعل البحث الخارجي مسموحًا تلقائيًا. `web_research` ما زالت capability عالية الخطورة (`NETWORK_ACCESS`) ويجب أن تمر عبر SecurityController وapproval binding عند استدعائها من Agent Core.

لا تزال هناك حاجة إلى:

- test accounts وscope owner authorization عند active target testing؛
- egress allowlist خارجية عند تشغيل live program؛
- TLS/deployment/rate-limit controls؛
- typed source ingestion يربط content hash بالمحتوى الفعلي؛
- benchmark مستقل يميز retrieval success عن factual verification.

لا يتم أثناء هذه المرحلة تشغيل DuckDuckGo أو أي هدف خارجي.
