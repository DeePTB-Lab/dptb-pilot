import { afterEach, describe, expect, it, vi } from 'vitest';
import { DptbPilotApiError, DptbPilotClient, encodeWorkspacePath, normalizeWorkspacePath } from './client';


describe('DptbPilotClient errors', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('updates execution configuration through the dedicated endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ message: 'ok', execution_mode: 'Remote', remote_enabled: true }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    ));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');

    await client.setExecutionConfig({
      session_id: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef',
      execution_mode: 'Remote',
      selected_machine_id: 'bohrium-1',
      remote_machine: {
        id: 'bohrium-1',
        name: 'Bohrium',
        type: 'Bohrium',
        config: {
          username: 'user@example.com',
          password: 'secret',
          project_id: '123',
        },
      },
    });

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:50003/api/execution-config');
    expect(request.method).toBe('PUT');
    expect(JSON.parse(request.body as string)).toMatchObject({
      execution_mode: 'Remote',
      selected_machine_id: 'bohrium-1',
    });
  });

  it('preserves a missing endpoint status for compatibility fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: 'Not Found' }),
      {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      },
    )));
    const client = new DptbPilotClient('http://localhost:50003');

    await expect(client.setExecutionConfig({
      session_id: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef',
      execution_mode: 'Local',
    })).rejects.toMatchObject({
      name: 'DptbPilotApiError',
      status: 404,
      message: 'Not Found',
    } satisfies Partial<DptbPilotApiError>);
  });

  it('reads the authoritative per-chat agent status', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: 'in_progress',
      tool_name: 'run_abacus',
      status_revision: 7,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');

    const result = await client.getAgentStatus('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef', 'chat-one');

    expect(result).toMatchObject({ status: 'in_progress', tool_name: 'run_abacus', status_revision: 7 });
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:50003/api/agent-status/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef?chat_id=chat-one');
  });

  it('loads and approves a pending workflow plan', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        workflow_plan: {
          title: 'Band workflow',
          summary: 'Run SCF and NSCF',
          task: 'calculate bands',
          steps: [],
        },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        message: 'Workflow plan approved',
        decision: 'approved',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');
    const sessionId = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef';
    const chatId = 'chat-one';

    const result = await client.getCurrentWorkflowPlan(sessionId, chatId);
    await client.decideWorkflowPlan(sessionId, 'approved', chatId);

    expect(result.workflow_plan).toMatchObject({ title: 'Band workflow' });
    expect(fetchMock.mock.calls[0][0]).toBe(`http://localhost:50003/api/workflow-plan/${sessionId}?chat_id=${chatId}`);
    expect(fetchMock.mock.calls[1][0]).toBe('http://localhost:50003/api/workflow-plan/decision');
    expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: 'POST' });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body as string)).toEqual({
      session_id: sessionId,
      chat_id: chatId,
      decision: 'approved',
    });
  });

  it('loads recorded task artifacts for a session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      tasks: [{ id: 'task-1', chat_id: 'chat-1', task: 'plot bands', completed_at: '2026-08-06T00:00:00Z', artifacts: [] }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');
    const sessionId = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef';

    const result = await client.getArtifacts(sessionId);

    expect(result.tasks[0].task).toBe('plot bands');
    expect(fetchMock.mock.calls[0][0]).toBe(`http://localhost:50003/api/artifacts/${sessionId}`);
  });

  it('stages a recorded remote artifact in the session workspace', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      artifact: { id: 'artifact-1', source_path: 'bohrium://store/result.tgz', workspace_path: 'artifacts/artifact-1/result.tgz', exists: true },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');
    const sessionId = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef';

    const result = await client.stageArtifact(sessionId, 'artifact-1');

    expect(result.artifact.workspace_path).toBe('artifacts/artifact-1/result.tgz');
    expect(fetchMock.mock.calls[0][0]).toBe(`http://localhost:50003/api/artifacts/${sessionId}/artifact-1/stage`);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST' });
  });

  it('uploads an 11 MB model as multipart binary data', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      uploaded_files: [{ name: 'model.pth', path: '/tmp/model.pth', size: 11 * 1024 * 1024 }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new DptbPilotClient('http://localhost:50003');
    const model = new Blob([new Uint8Array(11 * 1024 * 1024)]);

    await client.uploadFiles('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef', [{ name: 'model.pth', content: model }]);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(request.body).toBeInstanceOf(FormData);
    const uploaded = (request.body as FormData).get('files');
    expect(uploaded).toBeInstanceOf(Blob);
    expect((uploaded as Blob).size).toBe(11 * 1024 * 1024);
  });

  it('encodes nested artifact paths without losing directory separators', () => {
    expect(encodeWorkspacePath('results/band plot #1.png')).toBe('results/band%20plot%20%231.png');
    expect(encodeWorkspacePath('results\\dos?.csv')).toBe('results/dos%3F.csv');
    expect(encodeWorkspacePath('band_16_0_full/bandstructure_1788249247.npz/')).toBe('band_16_0_full/bandstructure_1788249247.npz');
    expect(normalizeWorkspacePath('band_16_0_full\\bandstructure_1788249247.npz\\')).toBe('band_16_0_full\\bandstructure_1788249247.npz');
  });
});
