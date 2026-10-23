# Private Agent Control — Plain Local Android App

هذا مجلد تطبيق Expo/React Native محلي يتصل فقط بـPrivate Agent API الموجود في مجلد المشروع الأب. لا يحتوي على Agent Core أو SQLite أو executor أو أدوات مباشرة.

## الحالة الحالية

تم إنشاء source scaffold وعميل API typed وواجهات Dashboard/Tasks/Approvals/Observability/Tools/Settings. لم تُثبت dependencies في البيئة الحالية، ولا توجد Expo CLI أو Android SDK/ADB هنا؛ لذلك لم يُدّعَ بناء APK أو اختبار جهاز بعد.

## التشغيل بعد توفير toolchain

```bash
cd mobile
pnpm install
EXPO_PUBLIC_AGENT_API_BASE_URL=http://10.0.2.2:8000/api/v1 pnpm start
```

بعد أول تثبيت ناجح يجب حفظ `pnpm-lock.yaml` ومراجعة التغييرات قبل استخدام `--frozen-lockfile` في CI.

لـAndroid Emulator استخدم `10.0.2.2` للوصول إلى API على host loopback. على جهاز حقيقي استخدم عنوان LAN للآلة التي تشغّل API، مع إبقاء API محميًا وعدم فتحه للعامة. أدخل Bearer token من شاشة Settings؛ لا يُحفظ في source أو logs، ويُخزن في SecureStore عندما تتوفر مكتبة Expo.

## حدود الأمن

التطبيق لا يوافق تلقائيًا على أي طلب، ولا يعدل approval binding، ولا يصل إلى أهداف خارجية. أزرار Approve وDeny صريحة ومصادق عليها. لا يجوز تشغيل API على `0.0.0.0` في شبكة غير موثوقة دون TLS وrate limiting وowner access control.

## بوابة التسليم المتبقية

يلزم لاحقًا تثبيت dependencies من lockfile، تشغيل typecheck/tests، تشغيل Expo Web/Expo Go، ثم بناء Android فعلي عبر Android SDK أو EAS/Expo build. لا يعتبر scaffold الحالي APK أو تطبيقًا نهائيًا.
