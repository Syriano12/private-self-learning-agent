# Phase 17A — Controlled API and Worker Control Plane

## الهدف

إكمال المسار التشغيلي المحلي من API إلى SQLite queue إلى BackgroundWorker ثم إلى الحالة النهائية، مع إبقاء SecurityController وApprovalGate وlease fencing هي حدود السلطة الوحيدة.

## ما تم تنفيذه

- `create_app(..., worker=...)` يدعم دورة حياة worker مرتبطة بـFastAPI lifespan.
- عند بدء التطبيق يبدأ worker مرة واحدة فقط.
- عند إيقاف التطبيق يطلب worker توقفًا تعاونيًا محدودًا زمنيًا؛ لا يحول مهمة قيد التنفيذ إلى نجاح مصطنع.
- `/ready` يعرض حالة worker (`running`, `stopped`, أو `fatal`) عندما يكون worker مضمّنًا في التطبيق.
- `private-agent serve --with-worker` يشغل API والعامل في عملية واحدة.
- موافقة API لا تعدل binding fields. عند تشغيل التطبيق مع worker مضمّن، تعيد إدراج المهمة تلقائيًا فقط بعد تسجيل الموافقة، مع إعادة التحقق المعتاد عند claim والتنفيذ.
- worker يعيد conciliation لمهام `WAITING_APPROVAL` التي أصبحت كل موافقاتها معتمدة قبل claim جديد. هذا انتقال queue-only ولا ينفذ action.
- حالات `EXECUTION_UNKNOWN` لا يعاد تشغيلها تلقائيًا.

## الاستخدام المحلي

```bash
export AGENT_REQUIRE_OWNER_AUTH=1
export AGENT_OWNER_EMAIL=ashbm1995@gmail.com
export AGENT_DB_PATH=/tmp/vulnora-agent.sqlite3
PYTHONPATH=src python3 -m private_agent.cli auth init-owner
private-agent serve --with-worker --host 127.0.0.1 --port 8000
```

ثم:

```bash
# افتح Vulnora على نفس الجهاز وسجّل الدخول بجلسة Cookie/CSRF.
# لا تُرسل كلمة المرور أو جلسة HttpOnly في URL أو السجلات.
```

إرجاع `QUEUED` يعني أن المهمة حُفظت durable فقط. يجب قراءة المهمة لاحقًا عبر `GET /api/v1/tasks/{task_id}`.

## حدود الضمان

هذه المرحلة لا تضيف:

- external target connector؛
- browser automation؛
- sandbox runtime فعليًا؛
- public hosting أو TLS termination؛
- multi-user authorization؛
- automatic retry بعد side effect غير محسوم.

تشغيل التطبيق على `0.0.0.0` أو خلف reverse proxy يحتاج عقد نشر مستقلًا يشمل TLS، rate limiting، secret rotation، وقيود وصول. لا يعتبر `--with-worker` وحده إعدادًا production-ready.

## دليل التحقق

اختبارات Phase 17A:

- API approval → automatic queue requeue عند وجود worker مضمّن.
- FastAPI lifespan start/stop.
- readiness worker state.
- end-to-end task execution عبر API وSQLite وworker وverification.

الأدلة الفعلية الحالية تُحفظ في نتائج الاختبار، ولا تتضمن تشغيل هدف خارجي أو كود غير موثوق.
