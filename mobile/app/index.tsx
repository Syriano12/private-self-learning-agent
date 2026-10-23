import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AgentApiClient, Approval, Event, Metrics, Readiness, TaskState, TaskSummary, Tool, isTerminalTask, validateApiBaseUrl } from '@/lib/api';
import { useAgentSession } from '@/lib/session';
import { Button, Card, colors, Field, Label, StateMessage, statusColor } from '@/components/ui';

type Tab = 'dashboard' | 'tasks' | 'approvals' | 'tools' | 'settings';

const tabs: Array<{ id: Tab; label: string }> = [
  { id: 'dashboard', label: 'الرئيسية' },
  { id: 'tasks', label: 'المهام' },
  { id: 'approvals', label: 'الموافقات' },
  { id: 'tools', label: 'الأدوات' },
  { id: 'settings', label: 'الإعدادات' },
];

function Badge({ value }: { value: string }) {
  return <View style={[styles.badge, { borderColor: statusColor(value) }]}><Text style={[styles.badgeText, { color: statusColor(value) }]}>{value}</Text></View>;
}

function SectionTitle({ title, detail }: { title: string; detail?: string }) {
  return <View style={styles.sectionHeading}><Text style={styles.sectionTitle}>{title}</Text>{detail ? <Text style={styles.sectionDetail}>{detail}</Text> : null}</View>;
}

function MetricStrip({ metrics }: { metrics: Metrics | null }) {
  const values = metrics ? [
    ['المهام', metrics.task_count],
    ['مكتملة', metrics.completed_tasks],
    ['فشل', metrics.failed_tasks],
    ['غير محسوم', metrics.execution_unknown],
  ] : [];
  return <View style={styles.metricRow}>{values.map(([label, value]) => <Card key={String(label)} style={styles.metricCard}><Text style={styles.metricValue}>{String(value)}</Text><Text style={styles.metricLabel}>{label}</Text></Card>)}</View>;
}

function TaskCard({ task, onPress }: { task: TaskSummary; onPress: () => void }) {
  return <Pressable accessibilityRole="button" accessibilityLabel={`فتح المهمة ${task.goal}`} onPress={onPress} style={({ pressed }) => [pressed && styles.pressed]}><Card><View style={styles.rowBetween}><Text style={styles.taskGoal} numberOfLines={2}>{task.goal}</Text><Badge value={task.status} /></View><Text style={styles.muted}>{task.task_id}</Text><View style={styles.rowBetween}><Text style={styles.muted}>الطابور: {task.queue_status}</Text><Text style={styles.muted}>التحقق: {task.verification_status}</Text></View></Card></Pressable>;
}

function TaskDetail({ task, timeline, onClose, onResume, busy }: { task: TaskState; timeline: Event[]; onClose: () => void; onResume: () => Promise<void>; busy: boolean }) {
  const resumable = task.status === 'FAILED' || task.status === 'BLOCKED';
  return <View style={styles.detailPanel}><View style={styles.rowBetween}><SectionTitle title="تفاصيل المهمة" /><Button label="إغلاق" variant="quiet" onPress={onClose} /></View><Card><View style={styles.rowBetween}><Text style={styles.taskGoal}>{task.goal}</Text><Badge value={task.status} /></View><Text style={styles.muted}>الخطوة الحالية: {task.current_step || '—'}</Text><Text style={styles.muted}>حالة التحقق: {task.verification_status}</Text><Text style={styles.muted}>آخر checkpoint: {task.checkpoint.checkpoint_type || '—'}</Text>{resumable ? <Button label={busy ? 'جارٍ التحقق…' : 'استئناف آمن'} onPress={() => void onResume()} disabled={busy} /> : null}{task.status === 'EXECUTION_UNKNOWN' ? <Text style={styles.notice}>هذه المهمة غير محسومة؛ لا يعيد التطبيق تنفيذها تلقائيًا.</Text> : null}</Card><SectionTitle title="Timeline" detail={`${timeline.length} حدث`} />{timeline.length === 0 ? <StateMessage title="لا توجد أحداث بعد" /> : timeline.slice(-30).reverse().map((event) => <Card key={event.event_id}><View style={styles.rowBetween}><Text style={styles.eventType}>{event.event_type}</Text><Text style={styles.muted}>{event.severity}</Text></View><Text style={styles.muted}>{event.timestamp}</Text><Text style={styles.muted}>{event.component}</Text></Card>)}</View>;
}

function Dashboard({ readiness, metrics, onRefresh }: { readiness: Readiness | null; metrics: Metrics | null; onRefresh: () => Promise<void> }) {
  return <ScrollView contentContainerStyle={styles.content} refreshControl={<RefreshControl onRefresh={onRefresh} refreshing={false} tintColor={colors.accent} />}><SectionTitle title="Private Agent Control" detail="مراقبة آمنة للوكيل المحلي" /><Card><View style={styles.rowBetween}><Text style={styles.cardTitle}>جاهزية النظام</Text><Badge value={readiness?.status ?? 'غير متصل'} /></View>{readiness ? Object.entries(readiness.checks).map(([key, value]) => <View key={key} style={styles.rowBetween}><Text style={styles.muted}>{key}</Text><Text style={{ color: statusColor(value), fontWeight: '700' }}>{value}</Text></View>) : <Text style={styles.muted}>أدخل إعدادات الاتصال ثم حدّث الصفحة.</Text>}</Card><MetricStrip metrics={metrics} /><Card><Text style={styles.cardTitle}>حدود آمنة</Text><Text style={styles.muted}>التطبيق لا ينفذ أدوات مباشرة ولا يوافق تلقائيًا. كل التنفيذ يمر عبر API وSecurityController والـworker المحلي.</Text></Card></ScrollView>;
}

function Tasks({ tasks, selectedTask, timeline, goal, setGoal, onCreate, onSelect, onClose, onResume, busy }: { tasks: TaskSummary[]; selectedTask: TaskState | null; timeline: Event[]; goal: string; setGoal: (value: string) => void; onCreate: () => Promise<void>; onSelect: (id: string) => Promise<void>; onClose: () => void; onResume: () => Promise<void>; busy: boolean }) {
  if (selectedTask) return <ScrollView contentContainerStyle={styles.content}><TaskDetail task={selectedTask} timeline={timeline} onClose={onClose} onResume={onResume} busy={busy} /></ScrollView>;
  return <ScrollView contentContainerStyle={styles.content}><SectionTitle title="المهام" detail="الإنشاء يعيد QUEUED؛ لا يعني اكتمال المهمة" /><Card><Field label="هدف المهمة" value={goal} onChangeText={setGoal} placeholder="مثال: حلل مستودعًا محليًا مصرحًا به" autoCapitalize="sentences" /><Button label={busy ? 'جارٍ الإنشاء…' : 'إنشاء مهمة'} onPress={onCreate} disabled={busy || !goal.trim()} /></Card>{tasks.length === 0 ? <StateMessage title="لا توجد مهام" detail="أنشئ مهمة محلية مصرحًا بها للبدء." /> : tasks.map((task) => <TaskCard key={task.task_id} task={task} onPress={() => void onSelect(task.task_id)} />)}</ScrollView>;
}

function Approvals({ approvals, onDecision, busy }: { approvals: Approval[]; onDecision: (approval: Approval, decision: 'approve' | 'deny') => Promise<void>; busy: boolean }) {
  return <ScrollView contentContainerStyle={styles.content}><SectionTitle title="مركز الموافقات" detail="راجع binding والسبب قبل اتخاذ القرار" />{approvals.length === 0 ? <StateMessage title="لا توجد موافقات معلقة" detail="لن تتم أي موافقة تلقائيًا." /> : approvals.map((approval) => <Card key={approval.approval_id}><View style={styles.rowBetween}><Text style={styles.cardTitle}>{approval.tool_or_skill}</Text><Badge value={approval.status} /></View><Text style={styles.muted}>الخطر: {approval.risk_level} · policy: {approval.policy_version}</Text><Text style={styles.muted}>السبب: {approval.reason}</Text><Text style={styles.muted}>الأثر المخطط: {approval.planned_effect}</Text><View style={styles.buttonRow}><Button label="رفض" variant="danger" disabled={busy} onPress={() => void onDecision(approval, 'deny')} /><Button label="موافقة" disabled={busy} onPress={() => void onDecision(approval, 'approve')} /></View></Card>)}</ScrollView>;
}

function Tools({ tools }: { tools: Tool[] }) {
  return <ScrollView contentContainerStyle={styles.content}><SectionTitle title="الأدوات" detail="قراءة فقط" />{tools.length === 0 ? <StateMessage title="لا توجد أدوات متاحة" /> : tools.map((tool) => <Card key={tool.name}><View style={styles.rowBetween}><Text style={styles.cardTitle}>{tool.name}</Text><Badge value={tool.risk_level} /></View><Text style={styles.muted}>{tool.description}</Text><Text style={styles.muted}>permission: {tool.permission_level}</Text><Text style={styles.muted}>capabilities: {tool.required_capabilities.join(', ') || 'none'}</Text></Card>)}</ScrollView>;
}

function Settings({ baseUrl, token, setBaseUrl, setToken, onSave, onClear, busy, message }: { baseUrl: string; token: string; setBaseUrl: (value: string) => void; setToken: (value: string) => void; onSave: () => Promise<void>; onClear: () => Promise<void>; busy: boolean; message: string }) {
  return <ScrollView contentContainerStyle={styles.content}><SectionTitle title="إعدادات الاتصال" detail="لا تُرسل القيم إلا عند طلب API" /><Card><Field label="API base URL" value={baseUrl} onChangeText={setBaseUrl} placeholder="http://10.0.2.2:8000/api/v1" autoCapitalize="none" /><Field label="Bearer token" value={token} onChangeText={setToken} placeholder="يُخزن في SecureStore" secureTextEntry /><Button label={busy ? 'جارٍ الحفظ…' : 'حفظ واختبار'} onPress={onSave} disabled={busy} /><Button label="مسح الرمز" variant="quiet" onPress={() => void onClear()} />{message ? <Text style={styles.notice}>{message}</Text> : null}</Card><Card><Text style={styles.cardTitle}>تنبيه</Text><Text style={styles.muted}>استخدم HTTPS خارج الجهاز المحلي. لا تضع token داخل source أو screenshots أو logs.</Text></Card></ScrollView>;
}

export default function Index() {
  const session = useAgentSession();
  const [tab, setTab] = useState<Tab>('dashboard');
  const [baseUrl, setBaseUrl] = useState(session.apiBaseUrl);
  const [token, setToken] = useState(session.token);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [selectedTask, setSelectedTask] = useState<TaskState | null>(null);
  const [timeline, setTimeline] = useState<Event[]>([]);
  const [goal, setGoal] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [settingsMessage, setSettingsMessage] = useState('');

  const client = useMemo(() => {
    if (!session.apiBaseUrl.trim() || !session.token.trim()) return null;
    try { return new AgentApiClient({ apiBaseUrl: session.apiBaseUrl, token: session.token }); } catch { return null; }
  }, [session.apiBaseUrl, session.token]);

  const refresh = useCallback(async () => {
    if (!client) return;
    setError('');
    const results = await Promise.allSettled([client.readiness(), client.metrics(), client.listTasks(), client.listApprovals(), client.tools()]);
    const [readyResult, metricResult, taskResult, approvalResult, toolResult] = results;
    if (readyResult.status === 'fulfilled') setReadiness(readyResult.value);
    if (metricResult.status === 'fulfilled') setMetrics(metricResult.value);
    if (taskResult.status === 'fulfilled') setTasks(taskResult.value.items);
    if (approvalResult.status === 'fulfilled') setApprovals(approvalResult.value.filter((item) => item.status === 'PENDING'));
    if (toolResult.status === 'fulfilled') setTools(toolResult.value);
    const failed = results.find((item) => item.status === 'rejected');
    if (failed && failed.status === 'rejected') setError(failed.reason instanceof Error ? failed.reason.message : 'تعذر تحديث البيانات.');
  }, [client]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!client) return undefined;
    const timer = setInterval(() => void refresh(), 5000);
    return () => clearInterval(timer);
  }, [client, refresh]);
  useEffect(() => {
    setBaseUrl(session.apiBaseUrl);
    setToken(session.token);
  }, [session.apiBaseUrl, session.token]);

  const createTask = async () => {
    if (!client) return;
    setBusy(true); setError('');
    try { await client.createTask(goal); setGoal(''); setTab('tasks'); await refresh(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'تعذر إنشاء المهمة.'); } finally { setBusy(false); }
  };
  const selectTask = async (taskId: string) => {
    if (!client) return;
    setBusy(true); setError('');
    try { const [task, events] = await Promise.all([client.getTask(taskId), client.getTimeline(taskId)]); setSelectedTask(task); setTimeline(events); } catch (cause) { setError(cause instanceof Error ? cause.message : 'تعذر قراءة المهمة.'); } finally { setBusy(false); }
  };
  const resumeSelectedTask = async () => {
    if (!client || !selectedTask) return;
    setBusy(true); setError('');
    try { const task = await client.resumeTask(selectedTask.task_id); setSelectedTask(task); setTimeline(await client.getTimeline(task.task_id)); await refresh(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'تعذر استئناف المهمة.'); } finally { setBusy(false); }
  };
  const decide = async (approval: Approval, decision: 'approve' | 'deny') => {
    if (!client) return;
    setBusy(true); setError('');
    try { if (decision === 'approve') await client.approve(approval.approval_id); else await client.deny(approval.approval_id); await refresh(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'تعذر تسجيل القرار.'); } finally { setBusy(false); }
  };
  const saveSettings = async () => {
    setBusy(true); setSettingsMessage(''); setError('');
    try { validateApiBaseUrl(baseUrl); if (!token.trim()) throw new Error('token_required'); await session.save({ apiBaseUrl: baseUrl.trim(), token: token.trim() }); setSettingsMessage('تم حفظ الإعدادات محليًا.'); } catch (cause) { setSettingsMessage(cause instanceof Error ? cause.message : 'تعذر حفظ الإعدادات.'); } finally { setBusy(false); }
  };
  const clearToken = async () => { await session.clear(); setToken(''); setSettingsMessage('تم مسح الرمز من SecureStore.'); };

  if (!session.ready) return <SafeAreaView style={styles.safe}><StateMessage title="جارٍ تجهيز الجلسة…" loading /></SafeAreaView>;
  if (!session.apiBaseUrl || !session.token) return <SafeAreaView style={styles.safe}><Settings baseUrl={baseUrl} token={token} setBaseUrl={setBaseUrl} setToken={setToken} onSave={saveSettings} onClear={clearToken} busy={busy} message={settingsMessage || 'ابدأ بإدخال عنوان API ورمز المالك.'} /></SafeAreaView>;

  return <SafeAreaView style={styles.safe}><View style={styles.appHeader}><View><Text style={styles.kicker}>PRIVATE AGENT</Text><Text style={styles.headerTitle}>{tab === 'dashboard' ? 'مركز التحكم' : tabs.find((item) => item.id === tab)?.label}</Text></View><Badge value={readiness?.status ?? 'offline'} /></View>{error ? <Pressable accessibilityRole="alert" onPress={() => setError('')} style={styles.error}><Text style={styles.errorText}>{error}</Text></Pressable> : null}<View style={styles.main}>{tab === 'dashboard' ? <Dashboard readiness={readiness} metrics={metrics} onRefresh={refresh} /> : tab === 'tasks' ? <Tasks tasks={tasks} selectedTask={selectedTask} timeline={timeline} goal={goal} setGoal={setGoal} onCreate={createTask} onSelect={selectTask} onClose={() => setSelectedTask(null)} onResume={resumeSelectedTask} busy={busy} /> : tab === 'approvals' ? <Approvals approvals={approvals} onDecision={decide} busy={busy} /> : tab === 'tools' ? <Tools tools={tools} /> : <Settings baseUrl={baseUrl} token={token} setBaseUrl={setBaseUrl} setToken={setToken} onSave={saveSettings} onClear={clearToken} busy={busy} message={settingsMessage} />}</View><View style={styles.tabBar}>{tabs.map((item) => <Pressable key={item.id} accessibilityRole="tab" accessibilityState={{ selected: tab === item.id }} onPress={() => { setTab(item.id); if (item.id !== 'tasks') setSelectedTask(null); }} style={[styles.tab, tab === item.id && styles.activeTab]}><Text style={[styles.tabText, tab === item.id && styles.activeTabText]}>{item.label}</Text></Pressable>)}</View></SafeAreaView>;
}

const styles = StyleSheet.create({
  safe: { backgroundColor: colors.bg, flex: 1 },
  main: { flex: 1 },
  content: { gap: 14, padding: 18, paddingBottom: 32 },
  appHeader: { alignItems: 'center', borderBottomColor: colors.border, borderBottomWidth: 1, flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 18, paddingVertical: 15 },
  kicker: { color: colors.accent, fontSize: 10, fontWeight: '900', letterSpacing: 2 },
  headerTitle: { color: colors.text, fontSize: 24, fontWeight: '900', marginTop: 3 },
  sectionHeading: { gap: 3 },
  sectionTitle: { color: colors.text, fontSize: 20, fontWeight: '900' },
  sectionDetail: { color: colors.muted, fontSize: 13 },
  cardTitle: { color: colors.text, fontSize: 16, fontWeight: '800' },
  taskGoal: { color: colors.text, flex: 1, fontSize: 16, fontWeight: '800', lineHeight: 22 },
  eventType: { color: colors.accent, fontSize: 14, fontWeight: '800' },
  muted: { color: colors.muted, fontSize: 13, lineHeight: 19 },
  badge: { alignSelf: 'flex-start', borderRadius: 999, borderWidth: 1, paddingHorizontal: 9, paddingVertical: 5 },
  badgeText: { fontSize: 11, fontWeight: '900' },
  rowBetween: { alignItems: 'center', flexDirection: 'row', gap: 12, justifyContent: 'space-between' },
  metricRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  metricCard: { flexGrow: 1, minWidth: '45%', padding: 13 },
  metricValue: { color: colors.accent, fontSize: 25, fontWeight: '900' },
  metricLabel: { color: colors.muted, fontSize: 12, marginTop: 2 },
  detailPanel: { gap: 14 },
  buttonRow: { flexDirection: 'row', gap: 10, marginTop: 5 },
  notice: { color: colors.accent, fontSize: 13, lineHeight: 19 },
  error: { backgroundColor: '#3A2025', borderBottomColor: colors.danger, borderBottomWidth: 1, paddingHorizontal: 18, paddingVertical: 11 },
  errorText: { color: colors.danger, fontSize: 13 },
  pressed: { opacity: 0.72 },
  tabBar: { backgroundColor: colors.panel, borderTopColor: colors.border, borderTopWidth: 1, flexDirection: 'row', paddingBottom: 7, paddingHorizontal: 5, paddingTop: 6 },
  tab: { alignItems: 'center', flex: 1, minHeight: 46, justifyContent: 'center', paddingHorizontal: 4 },
  activeTab: { borderTopColor: colors.accent, borderTopWidth: 2 },
  tabText: { color: colors.muted, fontSize: 11, fontWeight: '700' },
  activeTabText: { color: colors.accent },
});
