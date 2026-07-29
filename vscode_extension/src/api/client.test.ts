import { afterEach, describe, expect, it, vi } from 'vitest';
import { DptbPilotApiError, DptbPilotClient } from './client';


describe('DptbPilotClient errors', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
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
});
