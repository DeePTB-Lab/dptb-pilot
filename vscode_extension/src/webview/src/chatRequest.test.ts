import { describe, expect, it } from 'vitest';
import type { ChatMessage } from '../../shared/types';
import { conversationHistorySnapshot } from './chatRequest';

describe('conversationHistorySnapshot', () => {
  it('sends only completed messages from the active conversation', () => {
    const history: ChatMessage[] = [
      { role: 'user', content: 'Use silicon' },
      { role: 'assistant', content: 'I will use silicon.' },
      { role: 'assistant', content: '' },
    ];

    expect(conversationHistorySnapshot(history)).toEqual(history.slice(0, 2));
  });

  it('returns a detached snapshot', () => {
    const history: ChatMessage[] = [{ role: 'user', content: 'Original' }];
    const snapshot = conversationHistorySnapshot(history);

    snapshot[0].content = 'Changed';
    expect(history[0].content).toBe('Original');
  });
});
