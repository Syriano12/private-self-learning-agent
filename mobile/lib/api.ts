export type ApiVersioned = { api_version?: string; schema_version?: number };

export type ApiErrorPayload = ApiVersioned & {
  error_code?: string;
  message?: string;
  request_id?: string;
  task_id?: string;
  approval_id?: string;
};

export type Health = ApiVersioned & { status: string; service: string };
export type Readiness = ApiVersioned & {
  status: string;
  checks: Record<string, string>;
  service: string;
  request_id: string;
};
export type TaskCreate = ApiVersioned & {
  task_id: string;
  status: string;
  created_at: string;
  current_phase: string;
  request_id: string;
  idempotent_replay: boolean;
};
export type Checkpoint = {
  last_checkpoint_at: string;
  last_successful_checkpoint: string;
  checkpoint_type: string;
  sequence: number;
};
export type ActionSummary = {
  action_id: string;
  status: string;
  step_id: string;
  tool_name: string;
  execution_status: string;
  input_fingerprint: string;
  updated_at: string;
};
export type Approval = ApiVersioned & {
  approval_id: string;
  task_id: string;
  action_id: string;
  tool_or_skill: string;
  requested_capabilities: string[];
  risk_level: string;
  reason: string;
  planned_effect: string;
  input_fingerprint: string;
  policy_version: number;
  requested_at: string;
  expires_at: string;
  status: string;
  decided_at: string;
  actor: string;
};
export type TaskState = ApiVersioned & {
  task_id: string;
  goal: string;
  status: string;
  current_phase: string;
  current_step: string;
  plan_version: number;
  completed_actions: string[];
  pending_actions: string[];
  failed_actions: string[];
  actions: ActionSummary[];
  recovery_state: Record<string, unknown>;
  verification_status: string;
  checkpoint: Checkpoint;
  approvals: Approval[];
  created_at: string;
  updated_at: string;
  request_id: string;
};
export type TaskSummary = ApiVersioned & {
  task_id: string;
  goal: string;
  status: string;
  queue_status: string;
  current_phase: string;
  current_step: string;
  plan_version: number;
  verification_status: string;
  attempt_count: number;
  created_at: string;
  updated_at: string;
  last_checkpoint_at: string;
  available_at: string;
  last_error_code: string;
  waiting_for_approval: boolean;
  execution_unknown: boolean;
};
export type TaskList = ApiVersioned & { items: TaskSummary[]; next_cursor: string | null; request_id: string };
export type Event = ApiVersioned & {
  event_id: string;
  event_type: string;
  timestamp: string;
  task_id: string;
  action_id: string;
  step_id: string;
  correlation_id: string;
  component: string;
  severity: string;
  schema_version: number;
  sequence: number;
  metadata: Record<string, unknown>;
};
export type Metrics = ApiVersioned & {
  task_id?: string | null;
  task_count: number;
  completed_tasks: number;
  failed_tasks: number;
  blocked_tasks: number;
  action_count: number;
  successful_actions: number;
  failed_actions: number;
  blocked_actions: number;
  verification_success: number;
  verification_failure: number;
  recovery_count: number;
  replan_count: number;
  approval_requests: number;
  approval_approvals: number;
  approval_denials: number;
  approval_expirations: number;
  tool_failures: number;
  execution_unknown: number;
  observability_event_count: number;
  [key: string]: unknown;
};
export type Tool = {
  name: string;
  description: string;
  permission_level: string;
  risk_level: string;
  required_capabilities: string[];
  network_required: boolean;
};

const MAX_BASE_URL_LENGTH = 2048;
const MAX_GOAL_LENGTH = 16_384;
const TERMINAL_TASK_STATES = new Set(['COMPLETED', 'FAILED', 'BLOCKED', 'EXECUTION_UNKNOWN']);

function asObject(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? value as Record<string, unknown> : {};
}

function randomId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && 'randomUUID' in cryptoApi) return cryptoApi.randomUUID();
  return `mobile-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function trimSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

export function validateApiBaseUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_BASE_URL_LENGTH) throw new Error('api_base_url_required');
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error('api_base_url_invalid');
  }
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error('api_base_url_invalid');
  }
  return trimSlash(trimmed);
}

export function isTerminalTask(task: Pick<TaskState, 'status'>): boolean {
  return TERMINAL_TASK_STATES.has(task.status);
}

export class MobileApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string;
  readonly taskId: string;
  readonly approvalId: string;

  constructor(status: number, payload: ApiErrorPayload = {}) {
    super(payload.message || 'تعذر إكمال طلب الوكيل.');
    this.name = 'MobileApiError';
    this.status = status;
    this.code = payload.error_code || 'request_failed';
    this.requestId = payload.request_id || '';
    this.taskId = payload.task_id || '';
    this.approvalId = payload.approval_id || '';
  }
}

export type ApiClientOptions = {
  apiBaseUrl: string;
  token: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
};

export class AgentApiClient {
  private readonly apiBaseUrl: string;
  private readonly healthBaseUrl: string;
  private readonly token: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ApiClientOptions) {
    this.apiBaseUrl = validateApiBaseUrl(options.apiBaseUrl);
    this.healthBaseUrl = trimSlash(this.apiBaseUrl.replace(/\/api\/v1$/, ''));
    this.token = options.token.trim();
    this.timeoutMs = options.timeoutMs ?? 12_000;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  private async request<T>(path: string, init: RequestInit = {}, allowUnauthenticated = false): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers: Record<string, string> = {
      Accept: 'application/json',
      'X-Request-ID': randomId(),
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...(allowUnauthenticated ? {} : { Authorization: `Bearer ${this.token}` }),
      ...(init.headers as Record<string, string> | undefined),
    };
    try {
      const response = await this.fetchImpl(path, { ...init, headers, signal: controller.signal });
      const contentType = response.headers.get('content-type') || '';
      const payload = contentType.includes('json') ? await response.json().catch(() => ({})) : {};
      if (!response.ok) throw new MobileApiError(response.status, asObject(payload) as ApiErrorPayload);
      return payload as T;
    } catch (error) {
      if (error instanceof MobileApiError) throw error;
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new MobileApiError(408, { error_code: 'request_timeout', message: 'انتهت مهلة الاتصال.' });
      }
      throw new MobileApiError(0, { error_code: 'network_error', message: 'تعذر الوصول إلى API الوكيل.' });
    } finally {
      clearTimeout(timer);
    }
  }

  health() { return this.request<Health>(`${this.healthBaseUrl}/health`, {}, true); }
  readiness() { return this.request<Readiness>(`${this.healthBaseUrl}/ready`, {}, true); }
  listTasks(limit = 20) { return this.request<TaskList>(`${this.apiBaseUrl}/tasks?limit=${Math.max(1, Math.min(limit, 100))}`); }
  createTask(goal: string) {
    const cleaned = goal.trim();
    if (!cleaned || cleaned.length > MAX_GOAL_LENGTH) throw new MobileApiError(422, { error_code: 'goal_invalid', message: 'أدخل هدفًا صالحًا.' });
    return this.request<TaskCreate>(`${this.apiBaseUrl}/tasks`, {
      method: 'POST',
      headers: { 'Idempotency-Key': randomId() },
      body: JSON.stringify({ goal: cleaned }),
    });
  }
  getTask(taskId: string) { return this.request<TaskState>(`${this.apiBaseUrl}/tasks/${encodeURIComponent(taskId)}`); }
  resumeTask(taskId: string, approvalIds: Record<string, string> = {}) {
    return this.request<TaskState>(`${this.apiBaseUrl}/tasks/${encodeURIComponent(taskId)}/resume`, {
      method: 'POST',
      body: JSON.stringify({ approval_ids: approvalIds }),
    });
  }
  getTimeline(taskId: string) { return this.request<Event[]>(`${this.apiBaseUrl}/tasks/${encodeURIComponent(taskId)}/timeline`); }
  getTaskMetrics(taskId: string) { return this.request<Metrics>(`${this.apiBaseUrl}/tasks/${encodeURIComponent(taskId)}/metrics`); }
  listApprovals(pendingOnly = true) { return this.request<Approval[]>(`${this.apiBaseUrl}/approvals?pending_only=${String(pendingOnly)}`); }
  approve(approvalId: string) { return this.request<Approval>(`${this.apiBaseUrl}/approvals/${encodeURIComponent(approvalId)}/approve`, { method: 'POST', body: '{}' }); }
  deny(approvalId: string) { return this.request<Approval>(`${this.apiBaseUrl}/approvals/${encodeURIComponent(approvalId)}/deny`, { method: 'POST', body: '{}' }); }
  metrics() { return this.request<Metrics>(`${this.apiBaseUrl}/metrics`); }
  tools() { return this.request<Tool[]>(`${this.apiBaseUrl}/tools`); }
  failures(taskId?: string) { return this.events('/failures', taskId); }
  securityEvents(taskId?: string) { return this.events('/security-events', taskId); }
  toolEvents(taskId?: string) { return this.events('/tool-events', taskId); }

  private events(path: string, taskId?: string) {
    const query = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
    return this.request<Event[]>(`${this.apiBaseUrl}${path}${query}`);
  }
}
