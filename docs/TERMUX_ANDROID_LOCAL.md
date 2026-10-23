# Vulnora محليًا على Android عبر Termux

هذا الدليل يشغّل **control plane محليًا**: واجهة Vulnora في متصفح الهاتف، وBackend FastAPI، وWorker، وSQLite على الجهاز نفسه. لا يستخدم Oracle Cloud أو خدمة مدفوعة، ولا يفعّل httpx أو Nuclei.

## النتيجة الحالية

- Backend مبني على Python 3.11+ وFastAPI/Uvicorn وSQLite القياسي. لا يعتمد تشغيل API/Worker على systemd أو Docker أو Podman.
- `hashlib.scrypt`، SQLite، `threading`، وواجهات HTTP المستخدمة في المشروع متاحة مبدئيًا على Python/Termux، لكن التشغيل الفعلي على هاتف Android يجب تأكيده على الجهاز نفسه.
- Frontend مبني على Vite/React وNode.js. أضيف `pnpm dev:termux` و`pnpm preview:termux`، وكلاهما يربط على `127.0.0.1` فقط.
- `VITE_AGENT_API_BASE_URL` يجب أن يشير إلى `http://127.0.0.1:8123/api/v1` في تشغيل منفصل للواجهة والBackend.
- لا تضع `GEMINI_API_KEY` ولا أي token في Git أو في متغير `VITE_*`. عند غياب مفتاح Gemini يبقى planner المحلي deterministic؛ لا توجد بيانات تُرسل إلى مزود خارجي بسبب هذا الدليل.
- scanner runtime يبقى محجوبًا؛ غياب Podman/rootless sandbox موثوق على Termux لا يسمح بأي fallback إلى تشغيل host.

## قبل النقل إلى الهاتف

المستودعان الحاليان في بيئة التطوير يحتويان تعديلات محلية كثيرة غير منشورة. لا يعكس `git clone` من GitHub هذه التعديلات غير المرفوعة. لتشغيل **الحالة الحالية** انقل الملفات إلى الهاتف عبر USB/نسخ محلي موثوق أو انشر نسخة مراجعة بالطريقة التي يوافق عليها صاحب المشروع. لا تضع ملفات `.env` أو SQLite أو كلمات مرور في النقل العام.

## الخطوات على Termux

نفّذ كل أمر بعد نجاح الأمر السابق.

### 1. تثبيت Termux من مصدر واحد موثوق

ثبّت Termux وجميع إضافاته الاختيارية من المصدر نفسه فقط، مثل F-Droid أو GitHub الرسمي. لا تخلط APKs موقعة من مصادر مختلفة.

**الناتج المتوقع:** يفتح تطبيق Termux إلى shell يعمل تحت حساب Android غير root.

### 2. تحديث مستودعات Termux

```sh
pkg update
```

**الناتج المتوقع:** تحديث فهارس الحزم دون أخطاء repository.

### 3. ترقية حزم Termux الأساسية

```sh
pkg upgrade
```

**الناتج المتوقع:** اكتمال الترقية وعودة prompt. إذا طلب Termux تأكيدًا، وافق فقط على الحزم من مستودع Termux الرسمي.

### 4. تثبيت المتطلبات الأساسية

```sh
pkg install git python nodejs curl
```

**الناتج المتوقع:** توفر أوامر `git` و`python` و`python3` و`node` و`npm`. لا تحتاج خدمة systemd أو Docker أو Podman لتشغيل control plane المحلي.

### 5. التحقق من المعمارية والإصدارات

```sh
uname -m
```

**الناتج المتوقع:** غالبًا `aarch64` على هاتف ARM64. إذا ظهر غير ذلك، لا تفترض توافق الثنائيات قبل اختبارها.

```sh
python --version
```

**الناتج المتوقع:** Python 3.11 أو أحدث.

```sh
node --version
```

**الناتج المتوقع:** إصدار Node يدعم متطلبات lockfile؛ Node 18+ هو الحد العملي الآمن، وNode 22 مناسب.

```sh
npm --version
```

**الناتج المتوقع:** رقم إصدار npm يعمل.

### 6. تثبيت pnpm المطابق للمشروع

```sh
npm install --global pnpm@11.0.0
```

**الناتج المتوقع:** تثبيت pnpm 11 دون أخطاء.

```sh
pnpm --version
```

**الناتج المتوقع:** `11.0.0` أو إصدار 11 متوافق.

### 7. إنشاء مجلد بيانات خاص

```sh
mkdir -p "$HOME/vulnora-data"
```

**الناتج المتوقع:** لا توجد رسالة خطأ. ستبقى SQLite خارج checkout وداخل home الخاص بـTermux.

### 8. الانتقال إلى Backend

```sh
cd "$HOME/private-self-learning-agent"
```

**الناتج المتوقع:** يصبح prompt داخل checkout Backend. إذا نُقل المشروع إلى مسار مختلف، استخدم مساره الفعلي بدل هذا المسار.

### 9. إنشاء بيئة Python اختيارية ومعزولة

```sh
python -m venv "$HOME/vulnora-venv"
```

**الناتج المتوقع:** إنشاء `$HOME/vulnora-venv`. إذا كان venv غير متاح في حزمة Python على جهازك، استخدم Python Termux مباشرة بعد التحقق، ولا تخلط تثبيتات pip بين مشاريع مختلفة.

### 10. تفعيل البيئة

```sh
. "$HOME/vulnora-venv/bin/activate"
```

**الناتج المتوقع:** يظهر اسم البيئة في prompt، مثل `(vulnora-venv)`.

### 11. تثبيت Backend من metadata الحالية

```sh
python -m pip install --upgrade pip
```

**الناتج المتوقع:** اكتمال تحديث pip داخل البيئة فقط.

```sh
python -m pip install -e ".[test]"
```

**الناتج المتوقع:** تثبيت FastAPI وUvicorn وhttpx وhttpcore وBeautifulSoup4 وpytest. هذه الاعتماديات لا تتطلب Docker أو systemd. إذا فشل بناء wheel لمكتبة على هاتف محدد، سجّل الخطأ ولا تستبدل الحزمة بمكتبة غير مراجعة.

### 12. ضبط إعدادات loopback المحلية

```sh
export AGENT_DB_PATH="$HOME/vulnora-data/agent.sqlite3"
```

**الناتج المتوقع:** لا توجد رسالة؛ ستُحفظ المهام في SQLite داخل `$HOME/vulnora-data`.

```sh
export AGENT_REQUIRE_OWNER_AUTH=1
```

**الناتج المتوقع:** لا توجد رسالة؛ جلسة المالك مطلوبة.

```sh
export AGENT_OWNER_EMAIL="ashbm1995@gmail.com"
```

**الناتج المتوقع:** لا توجد رسالة؛ هذا يحدد البريد المسموح لتهيئة المالك، ولا ينشئ الحساب وحده.

```sh
export AGENT_COOKIE_SECURE=0
```

**الناتج المتوقع:** لا توجد رسالة؛ هذا مسموح فقط لأن الاتصال loopback عبر HTTP محلي. لا تستخدمه على عنوان شبكة أو Internet.

```sh
export AGENT_COOKIE_SAMESITE=lax
```

**الناتج المتوقع:** لا توجد رسالة؛ إعداد Cookie مناسب لتشغيل loopback المحلي.

```sh
export AGENT_ALLOWED_ORIGINS="http://127.0.0.1:3000,http://localhost:3000"
```

**الناتج المتوقع:** لا توجد رسالة؛ كلا العنوانين loopback فقط، ولا يسمح هذا بعنوان Wi-Fi أو عنوان عام.

```sh
export AGENT_WORKER_ID="termux-local-worker"
```

**الناتج المتوقع:** لا توجد رسالة؛ هوية worker محلية ثابتة لهذه الجلسة.

```sh
unset GEMINI_API_KEY
```

**الناتج المتوقع:** لا توجد رسالة. لن يستخدم التشغيل المحلي مزود LLM خارجيًا بسبب مفتاح مخفي أو وهمي.

### 13. تهيئة المالك مرة واحدة

```sh
PYTHONPATH=src python -m private_agent.cli auth init-owner --database "$AGENT_DB_PATH"
```

**الناتج المتوقع:** يطلب كلمة المرور مرتين، ثم يطبع JSON بحالة `initialized`. أدخل كلمة مرور جديدة مباشرة في prompt فقط؛ لا تضعها في الأمر أو ملف أو سجل أو محادثة. إذا كان الحساب مهيأً مسبقًا، سيرفض الأمر ولا ينشئ مالكًا ثانيًا.

### 14. تشغيل Backend وWorker على loopback

```sh
PYTHONPATH=src python -m private_agent.cli serve --with-worker --host 127.0.0.1 --port 8123
```

**الناتج المتوقع:** يبدأ Uvicorn ويستمع على `127.0.0.1:8123`، ويعمل worker ضمن دورة حياة API. يجب ألا يظهر `0.0.0.0` أو عنوان Wi-Fi في سطر الاستماع.

اترك جلسة Termux هذه مفتوحة أثناء الاستخدام. أوقفها بـ`Ctrl-C`؛ لا تقتل العملية أثناء مهمة حساسة إلا مع فهم أن المهمة قد تصبح `EXECUTION_UNKNOWN` وفق قواعد النظام.

### 15. فتح جلسة Termux ثانية

```sh
cd "$HOME/agentctrl"
```

**الناتج المتوقع:** يصبح prompt داخل checkout Frontend.

### 16. تثبيت Frontend من lockfile

```sh
pnpm install --frozen-lockfile
```

**الناتج المتوقع:** اكتمال التثبيت دون تعديل `pnpm-lock.yaml`. يحتوي lockfile الحالي على حزم ARM64 الخاصة بـesbuild وRollup، لكن يجب تأكيد نجاح التثبيت على هاتفك نفسه.

### 17. تشغيل الواجهة على loopback فقط

```sh
VITE_AGENT_API_BASE_URL=http://127.0.0.1:8123/api/v1 pnpm dev:termux
```

**الناتج المتوقع:** يبدأ Vite على `http://127.0.0.1:3000/` فقط.

### 18. فتح Vulnora في متصفح الهاتف

افتح في متصفح الهاتف:

```text
http://127.0.0.1:3000/
```

**الناتج المتوقع:** تظهر شاشة تسجيل الدخول. سجّل بالبريد والعبارة التي عيّنتها تفاعليًا، ثم تحقق من عرض الجلسة والمهام.

### 19. فحص جاهزية Backend من جلسة Termux ثالثة

```sh
curl http://127.0.0.1:8123/health
```

**الناتج المتوقع:** استجابة JSON ناجحة للصحة.

```sh
curl http://127.0.0.1:8123/ready
```

**الناتج المتوقع:** `ready` فقط إذا كانت SQLite والschema والمصادقة والworker جاهزة. لا تعتبر `/health` وحدها دليلًا على جاهزية worker.

### 20. إبقاء العملية مستيقظة أثناء التشغيل الطويل

```sh
termux-wake-lock
```

**الناتج المتوقع:** لا توجد رسالة أو يظهر إشعار Termux بوجود wakelock. هذا يقلل sleep العميق لكنه لا يضمن أن Android أو الشركة المصنعة لن توقف العملية.

```sh
termux-wake-unlock
```

**الناتج المتوقع:** تحرير wakelock عند الانتهاء.

## ما لم يتم إثباته هنا

تم التحقق في Sandbox من Backend الكامل، واختبارات Frontend والتكامل المحلي، وbuild وtypecheck وcompileall. لم يتم تشغيل المشروع على هاتف Android فعلي، لذلك لم يُثبت بعد:

- نجاح تثبيت كل wheel أو حزمة Python على جهازك.
- نجاح Vite/esbuild/Rollup على معمارية هاتفك الفعلية.
- استمرار worker بعد إطفاء الشاشة أو انتقال Termux للخلفية.
- سلوك Android/OEM عند الضغط أو انخفاض الذاكرة.
- الأداء والحرارة والبطارية والتخزين المتاح.
- نجاح نقل التعديلات المحلية غير المنشورة إلى الهاتف.

Android يفرض قيودًا على خدمات الخلفية، وTermux الرسمي يحذر من عدم الاستقرار على Android 12+ ومن قتل العمليات الوهمية أو العمليات التي تستهلك CPU بكثرة. لذلك هذا تشغيل محلي أثناء بقاء الهاتف وTermux متاحين، وليس خدمة 24/7 مضمونة.

## حدود الأمان

- Backend وFrontend كلاهما loopback؛ لا تستخدم `--host 0.0.0.0` على الهاتف.
- لا تفتح منفذ 8123 أو 3000 للـWi-Fi أو Internet.
- لا تستخدم `AGENT_COOKIE_SECURE=0` خارج loopback.
- لا تضع كلمة المرور أو session cookie أو CSRF token في ملفات أو logs.
- لا تشغّل `runtime-preflight` باعتباره إثباتًا لوجود scanner sandbox، ولا تحاول تثبيت أو تشغيل Podman/rootless runtime على Termux كمسار تجاوز.
- لا تشغّل httpx أو Nuclei أو أي executable غير موثوق. Public scanner adapters يجب أن تبقى `UNAVAILABLE` عند غياب external sandbox verified.
- SQLite والبيانات تبقى داخل مساحة Termux الخاصة، وليست متاحة تلقائيًا لتطبيقات Android الأخرى.

## أول خطوة للمستخدم

**ثبّت Termux من مصدر رسمي واحد، وافتحه، ثم نفّذ الأمر `uname -m` وأرسل الناتج قبل نقل المشروع أو تثبيت الاعتماديات.** هذا هو أول تحقق من أن الهاتف ARM64 وأننا لا نخلط بين توافق Sandbox وتوافق الجهاز الفعلي.

## مصادر التشغيل والقيود

- Termux الرسمي: https://github.com/termux/termux-app
- Termux docs: https://termux.dev/en/docs/
- Android Background Execution Limits: https://developer.android.com/about/versions/oreo/background
- Android background work restrictions: https://developer.android.com/develop/background-work/background-tasks/bg-work-restrictions
