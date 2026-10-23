import { AgentApiClient, MobileApiError, validateApiBaseUrl } from '../lib/api';

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
  } as Response;
}

describe('AgentApiClient', () => {
  it('rejects invalid or credential-bearing base URLs', () => {
    expect(() => validateApiBaseUrl('')).toThrow('api_base_url_required');
    expect(() => validateApiBaseUrl('file:///tmp/agent')).toThrow('api_base_url_invalid');
    expect(() => validateApiBaseUrl('https://user:pass@example.test/api/v1')).toThrow('api_base_url_invalid');
    expect(validateApiBaseUrl('http://10.0.2.2:8000/api/v1/')).toBe('http://10.0.2.2:8000/api/v1');
  });

  it('sends Bearer and request headers without exposing the token in the URL', async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    const client = new AgentApiClient({
      apiBaseUrl: 'http://127.0.0.1:8123/api/v1',
      token: 'unit-test-token',
      fetchImpl: async (input, init) => {
        calls.push({ url: String(input), init });
        return response({ task_id: 'task-1', status: 'QUEUED', created_at: '', current_phase: '', request_id: 'req-1', idempotent_replay: false });
      },
    });

    await client.createTask('inspect local fixture');
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('http://127.0.0.1:8123/api/v1/tasks');
    expect(calls[0]?.url).not.toContain('unit-test-token');
    expect((calls[0]?.init?.headers as Record<string, string>).Authorization).toBe('Bearer unit-test-token');
    expect((calls[0]?.init?.headers as Record<string, string>)['Idempotency-Key']).toBeTruthy();
  });

  it('converts API errors into a typed error without echoing credentials', async () => {
    const client = new AgentApiClient({
      apiBaseUrl: 'http://127.0.0.1:8123/api/v1',
      token: 'unit-test-token',
      fetchImpl: async () => response({ error_code: 'not_ready', message: 'not ready', request_id: 'req-2' }, 503),
    });

    await expect(client.metrics()).rejects.toMatchObject({ status: 503, code: 'not_ready', requestId: 'req-2' });
    try {
      await client.metrics();
    } catch (error) {
      expect(error).toBeInstanceOf(MobileApiError);
      expect(String(error)).not.toContain('unit-test-token');
    }
  });

  it('bounds task list requests to the API contract', async () => {
    let requested = '';
    const client = new AgentApiClient({
      apiBaseUrl: 'http://127.0.0.1:8123/api/v1',
      token: 'unit-test-token',
      fetchImpl: async (input) => {
        requested = String(input);
        return response({ items: [], next_cursor: null, request_id: 'req-3' });
      },
    });
    await client.listTasks(10000);
    expect(requested).toBe('http://127.0.0.1:8123/api/v1/tasks?limit=100');
  });
});
