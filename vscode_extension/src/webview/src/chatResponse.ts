import type { ChatMessage, ChatSession } from '../../shared/types';

export function applyAssistantSnapshot(
  history: ChatMessage[],
  content: string,
  patch: Partial<ChatMessage> = {},
): ChatMessage[] {
  const next = [...history];
  const last = next[next.length - 1];
  if (last?.role === 'assistant') {
    next[next.length - 1] = { ...last, content, ...patch };
  } else {
    next.push({ role: 'assistant', content, ...patch });
  }
  return next;
}

export function applyAssistantSnapshotToSession(
  sessions: ChatSession[],
  chatId: string,
  content: string,
  patch: Partial<ChatMessage> = {},
  updatedAt = new Date().toISOString(),
): ChatSession[] {
  return sessions.map((session) => {
    if (session.chat_id !== chatId) return session;
    const history = applyAssistantSnapshot(session.history, content, patch);
    return {
      ...session,
      history,
      last_active: updatedAt,
      message_count: history.length,
    };
  });
}
