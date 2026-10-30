import { describe, expect, it } from 'vitest';
import type { ChatMessage, ChatSession } from '../../shared/types';
import { applyAssistantSnapshot, applyAssistantSnapshotToSession } from './chatResponse';

describe('applyAssistantSnapshot', () => {
  it('replaces cumulative websocket snapshots instead of appending repeated content', () => {
    const initial: ChatMessage[] = [
      { role: 'user', content: 'Run the workflow' },
      { role: 'assistant', content: '' },
    ];

    const first = applyAssistantSnapshot(initial, 'Workflow approved.');
    const second = applyAssistantSnapshot(first, 'Workflow approved. Step one complete.');
    const final = applyAssistantSnapshot(second, 'Workflow approved. Step one complete. Done.');

    expect(final.at(-1)?.content).toBe('Workflow approved. Step one complete. Done.');
  });

  it('adds an assistant message when the history does not already end with one', () => {
    const history: ChatMessage[] = [{ role: 'user', content: 'Hello' }];

    expect(applyAssistantSnapshot(history, 'Hi')).toEqual([
      ...history,
      { role: 'assistant', content: 'Hi' },
    ]);
  });

  it('routes a delayed response only to the conversation that sent the request', () => {
    const sessions: ChatSession[] = [
      {
        chat_id: 'waiting-chat',
        user_id: 'user',
        title: 'Waiting chat',
        history: [
          { role: 'user', content: 'Run a calculation' },
          { role: 'assistant', content: '' },
        ],
        created_at: '2026-08-10T00:00:00.000Z',
        last_active: '2026-08-10T00:00:00.000Z',
        message_count: 2,
      },
      {
        chat_id: 'active-chat',
        user_id: 'user',
        title: 'Active chat',
        history: [{ role: 'user', content: 'A different conversation' }],
        created_at: '2026-08-10T00:00:00.000Z',
        last_active: '2026-08-10T00:00:00.000Z',
        message_count: 1,
      },
    ];

    const updated = applyAssistantSnapshotToSession(
      sessions,
      'waiting-chat',
      'Calculation complete.',
      {},
      '2026-08-10T00:01:00.000Z',
    );

    expect(updated[0].history.at(-1)?.content).toBe('Calculation complete.');
    expect(updated[1]).toBe(sessions[1]);
    expect(updated[1].history).toEqual([{ role: 'user', content: 'A different conversation' }]);
  });
});
