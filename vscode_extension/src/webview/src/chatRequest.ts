import type { ChatMessage } from '../../shared/types';

export function conversationHistorySnapshot(history: ChatMessage[]): ChatMessage[] {
  return history
    .filter((message) => message.content.trim().length > 0)
    .map((message) => ({ ...message }));
}
