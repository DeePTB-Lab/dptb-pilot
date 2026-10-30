import { describe, expect, it } from 'vitest';
import type { ChatSession, WSMessage } from '../../shared/types';
import { applyRunStarted, applyStageCompleted, runtimeFromMessage } from './chatRuntime';

const session: ChatSession = {
  chat_id: 'chat-1',
  user_id: 'user',
  title: 'Chat',
  history: [],
  created_at: '2026-09-01T00:00:00Z',
  last_active: '2026-09-01T00:00:00Z',
  message_count: 0,
};

describe('chat runtime reducers', () => {
  it('adds a user message only when a queued prompt starts', () => {
    const event = { type: 'run_started', prompt_id: 'p1', run_id: 'r1', message: 'first' } as WSMessage;
    const once = applyRunStarted([session], 'chat-1', event);
    const twice = applyRunStarted(once, 'chat-1', event);
    expect(twice[0].history).toHaveLength(1);
    expect(twice[0].history[0]).toMatchObject({ role: 'user', content: 'first', message_id: 'p1' });
  });

  it('appends distinct stage messages and de-duplicates replays', () => {
    const first = applyStageCompleted([session], 'chat-1', {
      type: 'stage_completed', message_id: 'm1', run_id: 'r1', stage_id: 'tool-batch-1', sequence: 2, content: 'step one', tools: ['list_workspace_files'],
      tool_results: [{ tool: 'list_workspace_files', input: { pattern: '*.json' }, result: { files: ['a.json'] } }],
    });
    const second = applyStageCompleted(first, 'chat-1', {
      type: 'stage_completed', message_id: 'm2', run_id: 'r1', stage_id: 's2', sequence: 4, content: 'step two',
    });
    const replayed = applyStageCompleted(second, 'chat-1', {
      type: 'stage_completed', message_id: 'm1', run_id: 'r1', stage_id: 's1', sequence: 2, content: 'step one updated',
    });
    expect(replayed[0].history.map((item) => item.content)).toEqual(['step one updated', 'step two']);
    expect(replayed[0].history[0].tools).toEqual(['list_workspace_files']);
    expect(replayed[0].history[0].tool_results?.[0].input).toEqual({ pattern: '*.json' });
  });

  it('keeps FIFO queue state separate from the active run', () => {
    const queued = runtimeFromMessage(undefined, {
      type: 'prompt_queued', prompt_id: 'p1', message: 'first', position: 1,
    });
    const queuedTwice = runtimeFromMessage(queued, {
      type: 'prompt_queued', prompt_id: 'p2', message: 'second', position: 2,
    });
    const started = runtimeFromMessage(queuedTwice, {
      type: 'run_started', prompt_id: 'p1', run_id: 'r1', message: 'first',
    });
    expect(started.active_run?.run_id).toBe('r1');
    expect(started.queue.map((item) => item.prompt_id)).toEqual(['p2']);
  });
});
