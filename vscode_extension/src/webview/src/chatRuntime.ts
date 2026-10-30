import type {
  ChatMessage,
  ChatRuntimeSnapshot,
  ChatSession,
  QueuedPromptInfo,
  WSMessage,
} from '../../shared/types';

function updateSession(
  sessions: ChatSession[],
  chatId: string,
  updater: (history: ChatMessage[]) => ChatMessage[],
): ChatSession[] {
  return sessions.map((session) => {
    if (session.chat_id !== chatId) return session;
    const history = updater(session.history);
    return {
      ...session,
      history,
      last_active: new Date().toISOString(),
      message_count: history.length,
    };
  });
}

export function applyRunStarted(
  sessions: ChatSession[],
  chatId: string,
  message: WSMessage,
): ChatSession[] {
  const promptId = message.prompt_id || '';
  return updateSession(sessions, chatId, (history) => {
    if (promptId && history.some((item) => item.message_id === promptId)) return history;
    return [...history, {
      role: 'user',
      content: message.message || '',
      timestamp: message.started_at || new Date().toISOString(),
      message_id: promptId || undefined,
      prompt_id: promptId || undefined,
      run_id: message.run_id,
      status: 'running',
    }];
  });
}

export function applyStageCompleted(
  sessions: ChatSession[],
  chatId: string,
  message: WSMessage,
): ChatSession[] {
  const messageId = message.message_id || `${message.run_id || 'run'}:${message.sequence || 0}`;
  const nextMessage: ChatMessage = {
    role: 'assistant',
    content: message.content || '',
    timestamp: message.timestamp || new Date().toISOString(),
    message_id: messageId,
    run_id: message.run_id,
    stage_id: message.stage_id,
    sequence: message.sequence,
    status: message.status,
    tools: message.tools,
    tool_results: message.tool_results,
    usage_metadata: message.usage_metadata,
    usage_events: message.usage_events,
  };
  return updateSession(sessions, chatId, (history) => {
    const existingIndex = history.findIndex((item) => item.message_id === messageId);
    if (existingIndex >= 0) {
      const next = [...history];
      next[existingIndex] = {
        ...next[existingIndex],
        ...nextMessage,
        tools: nextMessage.tools ?? next[existingIndex].tools,
        tool_results: nextMessage.tool_results ?? next[existingIndex].tool_results,
      };
      return next;
    }
    return [...history, nextMessage];
  });
}

export function runtimeFromMessage(
  previous: ChatRuntimeSnapshot | undefined,
  message: WSMessage,
): ChatRuntimeSnapshot {
  const current: ChatRuntimeSnapshot = previous || { active_run: null, queue: [] };
  if (message.type === 'runtime_snapshot') {
    return {
      active_run: message.active_run || null,
      queue: message.queue || [],
      recent_runs: message.recent_runs || [],
    };
  }
  if (message.type === 'prompt_queued' && message.prompt_id) {
    const queued: QueuedPromptInfo = {
      prompt_id: message.prompt_id,
      message: message.message || '',
      workflow_mode: message.workflow_mode,
      protocol_version: message.protocol_version,
      enqueued_at: message.enqueued_at,
      position: message.position || current.queue.length + 1,
    };
    return {
      ...current,
      queue: [...current.queue.filter((item) => item.prompt_id !== queued.prompt_id), queued]
        .sort((left, right) => left.position - right.position),
    };
  }
  if (message.type === 'queued_prompt_removed' && message.prompt_id) {
    return {
      ...current,
      queue: current.queue
        .filter((item) => item.prompt_id !== message.prompt_id)
        .map((item, index) => ({ ...item, position: index + 1 })),
    };
  }
  if (message.type === 'run_started') {
    return {
      ...current,
      active_run: {
        run_id: message.run_id || '',
        prompt_id: message.prompt_id || '',
        message: message.message || '',
        status: 'running',
        sequence: message.sequence || 0,
        started_at: message.started_at,
      },
      queue: current.queue
        .filter((item) => item.prompt_id !== message.prompt_id)
        .map((item, index) => ({ ...item, position: index + 1 })),
    };
  }
  if (['run_completed', 'run_failed', 'run_cancelled'].includes(message.type)) {
    return { ...current, active_run: null };
  }
  return current;
}
