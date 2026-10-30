import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { ReactNode } from 'react';
import remarkGfm from 'remark-gfm';
import { DptbPilotApiError, DptbPilotClient } from '../../api/client';
import { DEFAULT_SERVER_URL, generateSessionId, isValidSessionId, normalizeBaseUrl, resolveServerResourceUrl } from '../../api/url';
import { applyAssistantSnapshotToSession } from './chatResponse';
import { applyRunStarted, applyStageCompleted, runtimeFromMessage } from './chatRuntime';
import { conversationHistorySnapshot } from './chatRequest';
import { installSelectionCopyBridge } from './clipboard';
import type {
  ArtifactInfo,
  ArtifactTask,
  AgentStatus,
  ChatMessage,
  ChatRuntimeSnapshot,
  ChatSession,
  ExecutionMode,
  ExecutionPreferences,
  FileInfo,
  HostToWebviewMessage,
  ModifyMode,
  BohriumConfig,
  RemoteMachine,
  SlurmConfig,
  ToolSchema,
  ToolResult,
  VisualizationPayload,
  WorkflowDecision,
  WorkflowMode,
  WorkflowPlan,
  WebviewToHostMessage,
  WSMessage
} from '../../shared/types';

declare global {
  interface Window {
    acquireVsCodeApi?: () => { postMessage: (message: WebviewToHostMessage) => void; getState: () => unknown; setState: (state: unknown) => void };
  }
}

const vscode = window.acquireVsCodeApi?.();
const defaultMatterViz = { commandId: '', uriTemplate: '', autoOpen: false };

const shortcutPrompts = [
  { title: 'Analyze Structure', description: 'Inspect an uploaded crystal structure file', prompt: 'Please analyze the uploaded crystal structure file.' },
  { title: 'Calculate Bands', description: 'Compute and plot a band structure', prompt: 'Please help me compute and plot a band structure.' },
  { title: 'Generate Config', description: 'Generate a DeePTB training config', prompt: 'Please generate a DeePTB training configuration file.' },
  { title: 'Search Materials', description: 'Search related materials data', prompt: 'Please search for relevant materials data.' }
];

export function App() {
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [activeServerUrl, setActiveServerUrl] = useState(DEFAULT_SERVER_URL);
  const [sessionId, setSessionId] = useState('');
  const [connected, setConnected] = useState(false);
  const [executionConfigSupported, setExecutionConfigSupported] = useState(true);
  const [chatEventV2Supported, setChatEventV2Supported] = useState(false);
  const [loading, setLoading] = useState(false);
  const [responding, setResponding] = useState(false);
  const [agentStatus, setAgentStatus] = useState<'idle' | 'waiting_for_reply' | 'in_progress'>('idle');
  const [agentProgress, setAgentProgress] = useState('');
  const [agentDetails, setAgentDetails] = useState<AgentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [runtimeByChat, setRuntimeByChat] = useState<Record<string, ChatRuntimeSnapshot>>({});
  const [socketRevision, setSocketRevision] = useState(0);
  const [currentChatId, setCurrentChatId] = useState<string>('');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [artifactTasks, setArtifactTasks] = useState<ArtifactTask[]>([]);
  const [stagingArtifactIds, setStagingArtifactIds] = useState<Set<string>>(() => new Set());
  const [input, setInput] = useState('');
  const [schema, setSchema] = useState<ToolSchema | null>(null);
  const [modifyMode, setModifyMode] = useState<ModifyMode>('individual');
  const [executionMode, setExecutionMode] = useState<ExecutionMode>(
    () => readLocal('dptb.executionMode', 'Local')
  );
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>(
    () => readLocal('dptb.workflowMode', 'auto')
  );
  const [parallelMode, setParallelMode] = useState<'sequential' | 'auto' | 'parallel'>(
    () => readLocal('dptb.parallelMode', 'auto')
  );
  const [maxParallelism, setMaxParallelism] = useState<number>(
    () => Number(readLocal('dptb.maxParallelism', 4)) || 4
  );
  const [schemaJson, setSchemaJson] = useState('');
  const [activePanel, setActivePanel] = useState<'workflow' | 'params' | 'files'>('params');
  const [workflowPlan, setWorkflowPlan] = useState<WorkflowPlan | null>(null);
  const [workflowSubmitting, setWorkflowSubmitting] = useState<WorkflowDecision | null>(null);
  const [preview, setPreview] = useState<{ file: FileInfo; content?: string; imageUrl?: string } | null>(null);
  const [remoteMachines, setRemoteMachines] = useState<RemoteMachine[]>(() => readLocal('dptb.remoteMachines', []));
  const [selectedMachineId, setSelectedMachineId] = useState<string>(() => readLocal('dptb.selectedMachineId', ''));
  const [executionPreferencesHydrated, setExecutionPreferencesHydrated] = useState(false);
  const [matterViz, setMatterViz] = useState(defaultMatterViz);

  const wsRef = useRef<WebSocket | null>(null);
  const wsByChatRef = useRef(new Map<string, WebSocket>());
  const agentStateByChatRef = useRef(new Map<string, AgentStatus>());
  const agentStatusOrderRef = useRef(new Map<string, { revision: number; updatedAt: number }>());
  const runtimeByChatRef = useRef<Record<string, ChatRuntimeSnapshot>>({});
  const requestedMachineSecretsRef = useRef(new Set<string>());
  const currentChatIdRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const client = useMemo(() => new DptbPilotClient(activeServerUrl), [activeServerUrl]);
  const currentSession = sessions.find((item) => item.chat_id === currentChatId) || null;
  const currentRuntime = runtimeByChat[currentChatId] || { active_run: null, queue: [] };

  useEffect(() => {
    const listener = (event: MessageEvent<HostToWebviewMessage>) => {
      const message = event.data;
      if (message.type === 'bootstrap') {
        const configuredUrl = message.payload.serverUrl || DEFAULT_SERVER_URL;
        setServerUrl(configuredUrl);
        try {
          setActiveServerUrl(normalizeBaseUrl(configuredUrl));
        } catch {
          setError('保存的服务器地址无效，请输入正确的 DPTB Pilot 后端地址。');
        }
        if (message.payload.sessionId) setSessionId(message.payload.sessionId);
        if (message.payload.executionPreferences) {
          setExecutionMode(message.payload.executionPreferences.executionMode);
          setWorkflowMode(message.payload.executionPreferences.workflowMode || 'auto');
          setParallelMode(message.payload.executionPreferences.parallelMode || 'auto');
          setMaxParallelism(message.payload.executionPreferences.maxParallelism || 4);
          setRemoteMachines(message.payload.executionPreferences.remoteMachines);
          setSelectedMachineId(message.payload.executionPreferences.selectedMachineId);
        }
        setExecutionPreferencesHydrated(true);
        setMatterViz({ ...defaultMatterViz, ...message.payload.matterViz });
      } else if (message.type === 'serverUrlChanged') {
        setServerUrl(message.payload.serverUrl);
        try {
          setActiveServerUrl(normalizeBaseUrl(message.payload.serverUrl));
        } catch {
          setError('服务器地址无效，请输入正确的 HTTP 或 HTTPS 地址。');
        }
        setConnected(false);
        setExecutionConfigSupported(true);
      } else if (message.type === 'sessionChanged') {
        setSessionId(message.payload.sessionId);
      } else if (message.type === 'filesChanged') {
        setNotice(message.payload.uploadedFiles.length === 1
          ? `Uploaded ${message.payload.uploadedFiles[0]}.`
          : `Uploaded ${message.payload.uploadedFiles.length} files.`);
        setActivePanel('files');
        void loadFiles();
      } else if (message.type === 'uploadFailed') {
        setError(message.payload.message);
      } else if (message.type === 'secret') {
        const machineId = message.payload.requestId;
        if (message.payload.value) {
          setRemoteMachines((old) => old.map((machine) => {
            if (machine.id !== machineId || machine.type !== 'Bohrium') {
              return machine;
            }
            return {
              ...machine,
              config: {
                ...(machine.config as BohriumConfig),
                password: message.payload.value,
              },
            };
          }));
        }
      } else if (message.type === 'openNewSession') {
        void createNewSession();
      }
    };

    window.addEventListener('message', listener);
    return () => window.removeEventListener('message', listener);
  }, [client, sessionId, sessions]);

  useEffect(() => {
    vscode?.postMessage({ type: 'ready' });
  }, []);

  useEffect(() => installSelectionCopyBridge((text) => {
    vscode?.postMessage({ type: 'copyText', payload: { text } });
  }), []);

  useEffect(() => {
    currentChatIdRef.current = currentChatId;
    const savedState = agentStateByChatRef.current.get(currentChatId);
    setAgentStatus(savedState?.status || 'idle');
    setAgentProgress(savedState?.progress || '');
    setAgentDetails(savedState || null);
    setResponding(Boolean(currentRuntime.active_run || (savedState && savedState.status !== 'idle')));
  }, [currentChatId, currentRuntime.active_run]);

  useEffect(() => {
    if (!executionPreferencesHydrated) return;
    const persistentMachines = remoteMachines.map((machine) => {
      if (machine.type !== 'Bohrium') return machine;
      const config = machine.config as BohriumConfig;
      if (config.password) {
        vscode?.postMessage({
          type: 'saveSecret',
          payload: {
            key: remoteMachineSecretKey(machine.id),
            value: config.password,
          },
        });
      }
      return { ...machine, config: { ...config, password: '' } };
    });
    localStorage.setItem('dptb.remoteMachines', JSON.stringify(persistentMachines));
    const preferences: ExecutionPreferences = {
      executionMode,
      remoteMachines: persistentMachines,
      selectedMachineId,
      workflowMode,
      parallelMode,
      maxParallelism,
    };
    vscode?.postMessage({ type: 'setExecutionPreferences', payload: preferences });
  }, [executionMode, executionPreferencesHydrated, remoteMachines, selectedMachineId, workflowMode, parallelMode, maxParallelism]);

  useEffect(() => {
    remoteMachines.forEach((machine) => {
      if (machine.type !== 'Bohrium') return;
      const config = machine.config as BohriumConfig;
      if (config.password || requestedMachineSecretsRef.current.has(machine.id)) {
        return;
      }
      requestedMachineSecretsRef.current.add(machine.id);
      vscode?.postMessage({
        type: 'getSecret',
        payload: {
          key: remoteMachineSecretKey(machine.id),
          requestId: machine.id,
        },
      });
    });
  }, [remoteMachines]);

  useEffect(() => {
    if (!executionPreferencesHydrated) return;
    localStorage.setItem('dptb.selectedMachineId', selectedMachineId);
  }, [executionPreferencesHydrated, selectedMachineId]);

  useEffect(() => {
    if (!executionPreferencesHydrated) return;
    localStorage.setItem('dptb.executionMode', executionMode);
  }, [executionMode, executionPreferencesHydrated]);

  useEffect(() => {
    if (!executionPreferencesHydrated) return;
    localStorage.setItem('dptb.workflowMode', workflowMode);
  }, [executionPreferencesHydrated, workflowMode]);

  useEffect(() => {
    if (!executionPreferencesHydrated) return;
    localStorage.setItem('dptb.parallelMode', parallelMode);
    localStorage.setItem('dptb.maxParallelism', String(Math.min(32, Math.max(1, maxParallelism))));
  }, [executionPreferencesHydrated, parallelMode, maxParallelism]);

  useEffect(() => {
    if (!connected || !executionConfigSupported || !sessionId || responding) return;
    const timeout = window.setTimeout(() => {
      void applyExecutionConfig().catch((err) => {
        if (isMissingExecutionConfigEndpoint(err)) {
          setExecutionConfigSupported(false);
          setNotice(executionConfigCompatibilityMessage());
          return;
        }
        setError(errorMessage(err));
      });
    }, 300);
    return () => window.clearTimeout(timeout);
  }, [
    connected,
    executionConfigSupported,
    executionMode,
    remoteMachines,
    responding,
    selectedMachineId,
    sessionId,
  ]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.history]);

  useEffect(() => {
    if (!connected || !sessionId || !currentChatId) return;
    const socket = wsByChatRef.current.get(currentChatId);
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      connectWebSocket(sessionId, client, currentChatId);
    }
  }, [client, connected, currentChatId, sessionId, socketRevision]);

  useEffect(() => {
    if (!connected || !sessionId) return;
    void refreshReviewState();
    const interval = window.setInterval(() => void refreshReviewState(), 1200);
    return () => window.clearInterval(interval);
  }, [connected, sessionId, currentChatId, client]);

  useEffect(() => {
    if (!connected || !sessionId || !currentChatId) return;
    void refreshAgentStatus();
    const interval = window.setInterval(() => void refreshAgentStatus(), 500);
    return () => window.clearInterval(interval);
  }, [connected, sessionId, currentChatId, client]);

  async function login(targetSessionId = sessionId) {
    const cleanSessionId = targetSessionId.trim();
    if (!isValidSessionId(cleanSessionId)) {
      setError('Session ID must be 32 alphanumeric characters.');
      return;
    }

    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      setConnected(false);
      wsByChatRef.current.forEach((socket) => socket.close());
      wsByChatRef.current.clear();
      agentStateByChatRef.current.clear();
      agentStatusOrderRef.current.clear();
      runtimeByChatRef.current = {};
      setRuntimeByChat({});
      setAgentStatus('idle');
      setAgentProgress('');
      setAgentDetails(null);
      setResponding(false);
      const cleanServerUrl = normalizeBaseUrl(serverUrl);
      const activeClient = new DptbPilotClient(cleanServerUrl);
      const health = await activeClient.getHealth();
      if (health.status !== 'ok') {
        throw new Error('该地址未返回有效的 DPTB Pilot 健康检查结果。');
      }
      setServerUrl(cleanServerUrl);
      setActiveServerUrl(cleanServerUrl);
      setSessionId(cleanSessionId);
      vscode?.postMessage({ type: 'setServerUrl', payload: { serverUrl: cleanServerUrl } });
      vscode?.postMessage({ type: 'setSessionId', payload: { sessionId: cleanSessionId } });
      const loginResponse = await activeClient.login({ session_id: cleanSessionId });
      setChatEventV2Supported(Boolean(loginResponse.capabilities?.includes('chat_event_v2')));
      let compatibilityWarning: string | null = null;
      try {
        await applyExecutionConfig(activeClient, cleanSessionId);
        setExecutionConfigSupported(true);
      } catch (err) {
        if (!isMissingExecutionConfigEndpoint(err)) throw err;
        setExecutionConfigSupported(false);
        compatibilityWarning = executionConfigCompatibilityMessage();
      }
      setConnected(true);
      await loadSessions(cleanSessionId, activeClient);
      await loadFiles(cleanSessionId, activeClient);
      await loadArtifacts(cleanSessionId, activeClient);
      setNotice(compatibilityWarning);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function applyExecutionConfig(
    activeClient = client,
    activeSessionId = sessionId,
  ) {
    const selectedMachine = remoteMachines.find(
      (item) => item.id === selectedMachineId
    ) || null;
    await activeClient.setExecutionConfig({
      session_id: activeSessionId,
      execution_mode: executionMode,
      selected_machine_id: selectedMachineId,
      remote_machine: selectedMachine,
      parallel_mode: parallelMode,
      max_parallelism: maxParallelism,
    });
  }

  async function loadSessions(userId = sessionId, activeClient = client) {
    const response = await activeClient.getUserSessions(userId);
    let loaded = response.sessions || [];
    if (!loaded.length) {
      const created = makeSession(userId);
      loaded = [created];
      await activeClient.saveUserSessions(userId, loaded);
    }
    setSessions(loaded);
    setCurrentChatId((old) => (loaded.some((item) => item.chat_id === old) ? old : loaded[0]?.chat_id || ''));
  }

  async function loadFiles(userId = sessionId, activeClient = client) {
    if (!userId) return;
    const response = await activeClient.getFiles(userId);
    setFiles(response.files || []);
  }

  async function loadArtifacts(userId = sessionId, activeClient = client) {
    if (!userId) return;
    try {
      const response = await activeClient.getArtifacts(userId);
      setArtifactTasks(response.tasks || []);
    } catch (err) {
      if (err instanceof DptbPilotApiError && err.status === 404) {
        setArtifactTasks([]);
        return;
      }
      throw err;
    }
  }

  function connectWebSocket(userId = sessionId, activeClient = client, requestedChatId = currentChatIdRef.current) {
    const chatId = requestedChatId;
    if (!chatId) return null;
    wsByChatRef.current.get(chatId)?.close();
    const ws = activeClient.createWebSocket(userId, chatId);
    wsByChatRef.current.set(chatId, ws);
    wsRef.current = ws;
    ws.onmessage = (event) => handleWSMessage(DptbPilotClient.parseWSMessage(event.data), chatId);
    ws.onerror = () => {
      if (currentChatIdRef.current === chatId) {
        setError('WebSocket connection failed. Please check that the backend is reachable.');
      }
    };
    ws.onclose = () => {
      if (wsByChatRef.current.get(chatId) === ws) wsByChatRef.current.delete(chatId);
      if (currentChatIdRef.current === chatId) {
        setAgentProgress('Connection lost. Reconnecting...');
      }
      window.setTimeout(() => setSocketRevision((value) => value + 1), 1000);
    };
    return ws;
  }

  function handleWSMessage(message: WSMessage, socketChatId: string) {
    const targetChatId = message.chat_id || socketChatId;
    const isActiveChat = targetChatId === currentChatIdRef.current;
    const runtimeEventTypes: WSMessage['type'][] = [
      'runtime_snapshot', 'prompt_queued', 'queued_prompt_removed', 'run_started',
      'stage_started', 'stage_completed', 'run_completed', 'run_failed', 'run_cancelled'
    ];
    if (runtimeEventTypes.includes(message.type)) {
      setRuntimeByChat((old) => {
        const next = {
          ...old,
          [targetChatId]: runtimeFromMessage(old[targetChatId], message),
        };
        runtimeByChatRef.current = next;
        return next;
      });
      if (message.type === 'runtime_snapshot' && message.active_run) {
        const startedEvent: WSMessage = {
          type: 'run_started',
          chat_id: targetChatId,
          run_id: message.active_run.run_id,
          prompt_id: message.active_run.prompt_id,
          message: message.active_run.message,
          started_at: message.active_run.started_at,
        };
        setSessions((old) => {
          let next = applyRunStarted(old, targetChatId, startedEvent);
          for (const stage of message.active_run?.stages || []) {
            next = applyStageCompleted(next, targetChatId, {
              type: 'stage_completed',
              chat_id: targetChatId,
              ...(stage as Partial<WSMessage>),
            });
          }
          return next;
        });
      } else if (message.type === 'run_started') {
        setSessions((old) => applyRunStarted(old, targetChatId, message));
      } else if (message.type === 'stage_completed') {
        setSessions((old) => applyStageCompleted(old, targetChatId, message));
      }

      if (isActiveChat) {
        if (message.type === 'run_started' || message.type === 'stage_started' || message.type === 'stage_completed') {
          setResponding(true);
          setAgentStatus('in_progress');
        } else if (message.type === 'runtime_snapshot') {
          setResponding(Boolean(message.active_run));
          setAgentStatus(message.active_run ? 'in_progress' : 'idle');
          if (!message.active_run) {
            setAgentProgress('');
            setAgentDetails(null);
          }
        } else if (message.type === 'run_completed' || message.type === 'run_failed' || message.type === 'run_cancelled') {
          setResponding(false);
          setAgentStatus('idle');
          setAgentProgress('');
          setAgentDetails(null);
          void loadFiles();
          void loadArtifacts();
        }
      }
      return;
    }
    if (message.type === 'agent_status') {
      applyAgentStatus(targetChatId, message);
      return;
    }
    if (message.type === 'final_response' || message.type === 'error') {
      applyAgentStatus(targetChatId, {
        status: 'idle',
        progress: '',
        phase: 'idle',
        updated_at: new Date().toISOString(),
      });
    } else if (message.type === 'streaming_response') {
      applyAgentStatus(targetChatId, {
        status: 'in_progress',
        progress: 'Agent is generating a response',
        phase: 'generating',
        updated_at: new Date().toISOString(),
      });
    }
    if (message.type === 'streaming_response') {
      updateAssistantResponse(targetChatId, message.content || '', false, message);
    } else if (message.type === 'final_response') {
      updateAssistantResponse(targetChatId, message.content || '', true, message);
      if (isActiveChat) {
        setResponding(false);
        setAgentStatus('idle');
        setAgentProgress('');
        setAgentDetails(null);
      }
      void saveSessions();
      void loadFiles();
      void loadArtifacts();
    } else if (message.type === 'tool_modify_required') {
      if (isActiveChat && message.schema) {
        setSchema(message.schema);
        setSchemaJson(JSON.stringify(message.schema, null, 2));
        setActivePanel('params');
      }
    } else if (message.type === 'workflow_plan_required') {
      if (isActiveChat) {
        setWorkflowPlan(message.workflow_plan || null);
        setActivePanel('workflow');
      }
    } else if (message.type === 'error') {
      if (isActiveChat) {
        setError(message.message || '后端返回错误');
        setResponding(false);
        setAgentStatus('idle');
        setAgentProgress('');
      }
    } else if (message.type === 'charge_result') {
      patchLastAssistant(targetChatId, { charge_result: message.charge_result });
    } else if (message.type === 'usage_update') {
      const usage = normalizeUsage(message.usage);
      if (usage) {
        const previous = agentStateByChatRef.current.get(targetChatId) || { status: 'in_progress' as const };
        const next = { ...previous, usage: addUsage(previous.usage, usage), last_event: 'usage_update', updated_at: new Date().toISOString() };
        agentStateByChatRef.current.set(targetChatId, next);
        if (isActiveChat) setAgentDetails(next);
      }
    }
  }

  function applyAgentStatus(
    targetChatId: string,
    message: WSMessage | AgentStatus | (Partial<AgentStatus> & { status?: string }),
  ) {
    const status = message.status === 'waiting_for_reply' || message.status === 'in_progress'
      ? message.status
      : 'idle';
    const revision = Number(message.status_revision) || 0;
    const updatedAt = message.updated_at ? Date.parse(message.updated_at) || 0 : 0;
    const previousOrder = agentStatusOrderRef.current.get(targetChatId);
    if (previousOrder && (
      revision > 0 && previousOrder.revision > revision && updatedAt <= previousOrder.updatedAt
      || revision === previousOrder.revision && revision > 0 && updatedAt < previousOrder.updatedAt
      || revision === 0 && previousOrder.revision > 0
      || revision === 0 && previousOrder.revision === 0 && updatedAt < previousOrder.updatedAt
    )) {
      return;
    }
    agentStatusOrderRef.current.set(targetChatId, { revision, updatedAt });
    const next: AgentStatus = {
      ...message,
      status,
      progress: message.progress || '',
      usage: message.usage as AgentStatus['usage'],
      chat_id: targetChatId,
    };
    agentStateByChatRef.current.set(targetChatId, next);
    if (targetChatId !== currentChatIdRef.current) return;
    setAgentStatus(next.status);
    setAgentProgress(next.progress || '');
    setAgentDetails(next.status === 'idle' ? null : next);
    setResponding(Boolean(runtimeByChatRef.current[targetChatId]?.active_run) || next.status !== 'idle');
  }

  async function sendMessage(text = input) {
    const content = text.trim();
    if (!content || !currentSession) return;
    let ws = wsByChatRef.current.get(currentSession.chat_id);
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      ws = connectWebSocket(sessionId, client, currentSession.chat_id) || undefined;
    }
    if (!ws) return;
    if (ws.readyState === WebSocket.CONNECTING) {
      await new Promise<void>((resolve, reject) => {
        const previousOpen = ws!.onopen;
        ws!.onopen = (event) => {
          previousOpen?.call(ws!, event);
          resolve();
        };
        ws!.onerror = () => reject(new Error('WebSocket connection failed'));
      });
    }
    if (ws.readyState !== WebSocket.OPEN) return;

    setInput('');
    if (chatEventV2Supported) {
      const promptId = createRemoteMachineId();
      ws.send(JSON.stringify({
        type: 'enqueue_prompt',
        protocol_version: 2,
        prompt_id: promptId,
        message: content,
        chat_id: currentSession.chat_id,
        workflow_mode: workflowMode,
      }));
      return;
    }

    const nextHistory: ChatMessage[] = [
      ...currentSession.history,
      { role: 'user', content, timestamp: new Date().toISOString() },
      { role: 'assistant', content: '' },
    ];
    updateCurrentSession({ history: nextHistory, last_active: new Date().toISOString(), message_count: nextHistory.length });
    setResponding(true);
    setAgentStatus('waiting_for_reply');
    ws.send(JSON.stringify({
      message: content,
      chat_id: currentSession.chat_id,
      workflow_mode: workflowMode,
      history: conversationHistorySnapshot(currentSession.history),
    }));
  }

  function removeQueuedPrompt(promptId: string) {
    const ws = wsByChatRef.current.get(currentChatId);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: 'remove_queued_prompt',
      protocol_version: 2,
      chat_id: currentChatId,
      prompt_id: promptId,
    }));
  }

  function updateAssistantResponse(targetChatId: string, content: string, final: boolean, message: WSMessage) {
    const usage = normalizeUsage(message.usage_total || message.usage || message.usage_metadata);
    const finalPatch = final ? {
      timestamp: new Date().toISOString(),
      usage_metadata: usage,
      usage_events: message.usage_events,
      charge_result: message.charge_result
    } : {};
    setSessions((old) => applyAssistantSnapshotToSession(old, targetChatId, content, finalPatch));
    if (final && usage) {
      setSessions((old) => old.map((session) => session.chat_id === targetChatId
        ? { ...session, token_usage: addUsage(session.token_usage, usage) }
        : session));
    }
  }

  function patchLastAssistant(targetChatId: string, patch: Partial<ChatMessage>) {
    setSessions((old) => old.map((session) => {
      if (session.chat_id !== targetChatId) return session;
      const history = [...session.history];
      for (let i = history.length - 1; i >= 0; i -= 1) {
        if (history[i].role === 'assistant') {
          history[i] = { ...history[i], ...patch };
          break;
        }
      }
      return { ...session, history };
    }));
  }

  function updateCurrentSession(patch: Partial<ChatSession>) {
    setSessions((old) => old.map((session) => (session.chat_id === currentChatId ? { ...session, ...patch } : session)));
  }

  async function createNewSession() {
    if (!sessionId) return;
    const created = makeSession(sessionId);
    const next = [created, ...sessions];
    setSessions(next);
    setCurrentChatId(created.chat_id);
    if (connected) await client.saveUserSessions(sessionId, next);
  }

  async function saveSessions(next = sessions) {
    if (connected && sessionId) await client.saveUserSessions(sessionId, next);
  }

  async function deleteSession(chatId: string) {
    const runtime = runtimeByChatRef.current[chatId];
    if (runtime?.active_run || runtime?.queue.length) {
      setError('Cancel the active run and remove queued prompts before deleting this chat.');
      return;
    }
    wsByChatRef.current.get(chatId)?.close();
    wsByChatRef.current.delete(chatId);
    agentStateByChatRef.current.delete(chatId);
    agentStatusOrderRef.current.delete(chatId);
    const nextRuntime = { ...runtimeByChatRef.current };
    delete nextRuntime[chatId];
    runtimeByChatRef.current = nextRuntime;
    setRuntimeByChat(nextRuntime);
    const next = sessions.filter((item) => item.chat_id !== chatId);
    setSessions(next);
    if (currentChatId === chatId) setCurrentChatId(next[0]?.chat_id || '');
    await saveSessions(next);
  }

  async function refreshSchema() {
    try {
      const response = await client.getCurrentSchema(sessionId);
      const next = response.schema && Object.keys(response.schema).length ? response.schema : null;
      setSchema((old) => {
        const oldText = old ? JSON.stringify(old) : '';
        const newText = next ? JSON.stringify(next) : '';
        if (oldText !== newText) setSchemaJson(next ? JSON.stringify(next, null, 2) : '');
        return next;
      });
    } catch {
      // Keep chat usable when schema polling fails transiently.
    }
  }

  async function refreshWorkflowPlan() {
    try {
      const response = await client.getCurrentWorkflowPlan(sessionId, currentChatId);
      const next = response.workflow_plan && Object.keys(response.workflow_plan).length
        ? response.workflow_plan as WorkflowPlan
        : null;
      setWorkflowPlan(next);
    } catch {
      // Keep the rest of the UI usable when workflow polling fails transiently.
    }
  }

  async function refreshReviewState() {
    await Promise.all([refreshSchema(), refreshWorkflowPlan()]);
  }

  async function refreshAgentStatus() {
    try {
      const status = await client.getAgentStatus(sessionId, currentChatId);
      applyAgentStatus(currentChatId, status);
    } catch {
      // Older backends may not expose the status endpoint; WebSocket updates
      // remain the primary source in that case.
    }
  }

  async function decideWorkflow(decision: WorkflowDecision) {
    if (!workflowPlan || workflowSubmitting) return;
    setWorkflowSubmitting(decision);
    setError(null);
    try {
      const response = await client.decideWorkflowPlan(sessionId, decision, currentChatId);
      if (decision === 'pilot_review') {
        setWorkflowPlan(response.workflow_plan || workflowPlan);
        setNotice('Pilot review completed. Review the report before approving or rejecting.');
      } else {
        setWorkflowPlan(null);
        if (decision === 'rejected') setNotice('Workflow rejected.');
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setWorkflowSubmitting(null);
    }
  }

  async function submitParams(formData?: FormData) {
    if (!schema) return;
    setError(null);

    const selectedMachine = remoteMachines.find((item) => item.id === selectedMachineId) || null;
    if (executionMode === 'Remote' && !selectedMachine) {
      setError('Select a remote machine before submitting parameters.');
      setActivePanel('params');
      return;
    }

    let modifiedSchema: ToolSchema;
    if (modifyMode === 'json') {
      modifiedSchema = JSON.parse(schemaJson) as ToolSchema;
    } else {
      modifiedSchema = JSON.parse(JSON.stringify(schema)) as ToolSchema;
      Object.entries(modifiedSchema.input_schema.properties || {}).forEach(([key, prop]) => {
        const remoteProp = selectedMachine ? generateRemoteProperty(key, prop, selectedMachine, executionMode) : null;
        if (remoteProp) {
          modifiedSchema.input_schema.properties[key] = remoteProp;
          return;
        }
        if (key.toLowerCase() === 'executor' || key.toLowerCase() === 'storage') return;
        prop.user_input = parseFormValue(String(formData?.get(key) ?? ''), prop.type);
      });
    }

    await client.modifyParameters({
      session_id: sessionId,
      chat_id: currentChatId,
      modified_schema: modifiedSchema,
      execution_mode: executionMode,
      selected_machine_id: selectedMachineId,
      remote_machine: selectedMachine
    });
    setSchema(null);
    setSchemaJson('');
  }

  async function terminateExecution() {
    const response = await client.terminateExecution({ session_id: sessionId, chat_id: currentChatId });
    setError(response.status === 'terminating' ? null : response.message);
  }

  async function previewFile(file: FileInfo) {
    const ext = extensionOf(file.name);
    if (isImage(ext)) {
      setPreview({ file, imageUrl: client.getDownloadUrl(sessionId, file.name) });
      return;
    }
    const content = await client.previewTextFile(sessionId, file.name);
    setPreview({ file, content });
  }

  async function deleteFile(file: FileInfo) {
    await client.deleteFile(sessionId, file.name);
    await loadFiles();
  }

  function downloadArtifact(artifact: ArtifactInfo) {
    if (!artifact.exists || !artifact.workspace_path) return;
    vscode?.postMessage({
      type: 'downloadArtifact',
      payload: {
        sessionId,
        workspacePath: artifact.workspace_path,
        suggestedName: artifact.file_name || artifact.workspace_path.split('/').pop() || 'artifact'
      }
    });
  }

  async function stageArtifact(artifact: ArtifactInfo) {
    setStagingArtifactIds((current) => new Set(current).add(artifact.id));
    setError(null);
    try {
      await client.stageArtifact(sessionId, artifact.id);
      await Promise.all([loadFiles(), loadArtifacts()]);
      setNotice(`Artifact ${artifact.file_name || artifact.source_path.split('/').pop() || artifact.id} downloaded to the workspace.`);
    } catch (err) {
      setError(`Failed to download artifact to the workspace: ${errorMessage(err)}`);
    } finally {
      setStagingArtifactIds((current) => {
        const next = new Set(current);
        next.delete(artifact.id);
        return next;
      });
    }
  }

  function addRemoteMachine(type: 'Bohrium' | 'Slurm', name: string) {
    const machine: RemoteMachine = type === 'Bohrium'
      ? { id: createRemoteMachineId(), name, type, config: { username: '', password: '', project_id: '', image_name: '', scass_type: '' } }
      : { id: createRemoteMachineId(), name, type, config: { remote_root: '', hostname: '', username: '', key_filename: '', number_node: '1', queue_name: '' } };
    setRemoteMachines((old) => [...old, machine]);
    setSelectedMachineId(machine.id);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="brand">
          <strong>DPTB Pilot</strong>
          <span>VS Code Agent UI</span>
        </header>

        <section className="connect-card">
          <label>Server URL</label>
          <input value={serverUrl} onChange={(event) => { setServerUrl(event.target.value); setConnected(false); }} placeholder={DEFAULT_SERVER_URL} />
          <label>Session ID</label>
          <div className="inline-row">
            <input value={sessionId} onChange={(event) => setSessionId(event.target.value.replace(/[^a-zA-Z0-9]/g, '').slice(0, 32))} placeholder="32-char session id" />
            <button onClick={() => setSessionId(generateSessionId())}>生成</button>
          </div>
          <button className="primary" disabled={loading} onClick={() => void login()}>{loading ? '连接中...' : connected ? '重新连接' : '连接'}</button>
          {error && <div className="error-box">{error}</div>}
          {notice && <div className="notice-box">{notice}</div>}
        </section>

        <section className="sessions-header">
          <strong>会话</strong>
          <button onClick={() => void createNewSession()} disabled={!connected}>新建</button>
        </section>
        <div className="session-list">
          {sessions.map((session) => (
            <button key={session.chat_id} className={session.chat_id === currentChatId ? 'selected session-item' : 'session-item'} onClick={() => setCurrentChatId(session.chat_id)}>
              <span>{session.title || 'Untitled'}</span>
              <small>{session.message_count || session.history.length} messages</small>
              {session.token_usage?.total_tokens ? <small>{formatNumber(session.token_usage.total_tokens)} tokens</small> : null}
              <span className="delete" onClick={(event) => { event.stopPropagation(); void deleteSession(session.chat_id); }}>x</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-main">
        <div className="message-list">
          {!currentSession?.history.length && (
            <div className="welcome">
              <h2>Choose an action to start</h2>
              <div className="shortcut-grid">
                {shortcutPrompts.map((item) => (
                  <button key={item.title} onClick={() => void sendMessage(item.prompt)} disabled={!connected}>
                    <strong>{item.title}</strong>
                    <span>{item.description}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {currentSession?.history.map((message, index) => <MessageBubble key={message.message_id || `${message.role}-${index}`} message={message} matterViz={matterViz} serverUrl={activeServerUrl} sessionId={sessionId} responding={!chatEventV2Supported && responding && index === currentSession.history.length - 1} />)}
          {chatEventV2Supported && currentRuntime.active_run ? (
            <MessageBubble
              key={`active-${currentRuntime.active_run.run_id}`}
              message={{ role: 'assistant', content: '', run_id: currentRuntime.active_run.run_id, status: 'running' }}
              matterViz={matterViz}
              serverUrl={activeServerUrl}
              sessionId={sessionId}
              responding
            />
          ) : null}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          {currentRuntime.queue.length ? (
            <div className="prompt-queue" aria-label="Queued prompts">
              {currentRuntime.queue.map((queued) => (
                <div className="prompt-queue-item" key={queued.prompt_id}>
                  <span className="queue-position">{queued.position}</span>
                  <span className="queue-message" title={queued.message}>{queued.message}</span>
                  <button type="button" title="Remove queued prompt" aria-label="Remove queued prompt" onClick={() => removeQueuedPrompt(queued.prompt_id)}>x</button>
                </div>
              ))}
            </div>
          ) : null}
          <div className={`agent-status agent-status-${agentStatus} ${agentDetails?.phase === 'remote_waiting' ? 'agent-status-remote' : ''}`} role="status">
            <strong>{agentDetails?.phase === 'remote_waiting' ? 'REMOTE TASK RUNNING' : agentStatus === 'idle' ? 'Agent idle' : agentStatus === 'waiting_for_reply' ? 'Agent waiting for reply' : 'Agent in progress'}</strong>
            {agentDetails?.phase ? <span> · phase: {agentDetails.phase}</span> : null}
            {agentDetails?.tool_name ? <span> · tool: {agentDetails.tool_name}</span> : null}
            {agentDetails?.method_name ? <span> · method: {agentDetails.method_name}</span> : null}
            {agentDetails?.execution_mode ? <span> · mode: {agentDetails.execution_mode}</span> : null}
            {agentDetails?.remote_task_id ? <span className="agent-task-id"> · task: {agentDetails.remote_task_id}</span> : null}
            {agentDetails?.bohrium_task_id ? <span className="agent-task-id"> · Bohrium: {agentDetails.bohrium_task_id}</span> : null}
            {agentDetails?.remote_task_status ? <span> · status: {agentDetails.remote_task_status}</span> : null}
            {agentDetails?.remote_tasks?.length ? <span> · jobs: {agentDetails.remote_tasks.length}</span> : null}
            {agentProgress ? <span> · {agentProgress}</span> : null}
            {agentDetails?.event_count ? <span> · events: {agentDetails.event_count}</span> : null}
            {agentDetails?.elapsed_ms ? <span> · {formatDuration(agentDetails.elapsed_ms)}</span> : null}
            {agentDetails?.usage ? <span> · tokens: {formatNumber(agentDetails.usage.total_tokens || 0)}</span> : null}
          </div>
          <button
            title="Select files to upload"
            disabled={!connected || !sessionId}
            onClick={() => vscode?.postMessage({ type: 'pickFiles', payload: { serverUrl: activeServerUrl, sessionId } })}
          >Upload</button>
          <button
            title="Upload current editor file"
            disabled={!connected || !sessionId}
            onClick={() => vscode?.postMessage({ type: 'uploadCurrentFile', payload: { serverUrl: activeServerUrl, sessionId } })}
          >Current</button>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              void sendMessage();
            }
          }} placeholder="Type a message. Enter sends, Shift+Enter inserts a new line." />
          <button className="primary" disabled={!connected || !input.trim() || (!chatEventV2Supported && agentStatus !== 'idle')} onClick={() => void sendMessage()}>{chatEventV2Supported && currentRuntime.active_run ? 'Queue' : agentStatus !== 'idle' && !chatEventV2Supported ? 'Responding...' : 'Send'}</button>
        </div>
      </main>

      <aside className="right-panel">
        <div className="tabs">
          <button className={activePanel === 'workflow' ? 'selected' : ''} onClick={() => setActivePanel('workflow')}>
            Workflow{workflowPlan ? <span className="tab-badge">1</span> : null}
          </button>
          <button className={activePanel === 'params' ? 'selected' : ''} onClick={() => setActivePanel('params')}>Params</button>
          <button className={activePanel === 'files' ? 'selected' : ''} onClick={() => setActivePanel('files')}>Files</button>
        </div>
        {activePanel === 'workflow' ? (
          <WorkflowPanel plan={workflowPlan} submitting={workflowSubmitting} decide={decideWorkflow} />
        ) : activePanel === 'params' ? (
          <ParameterPanel
            schema={schema}
            schemaJson={schemaJson}
            setSchemaJson={setSchemaJson}
            modifyMode={modifyMode}
            setModifyMode={setModifyMode}
            executionMode={executionMode}
            setExecutionMode={setExecutionMode}
            workflowMode={workflowMode}
            setWorkflowMode={setWorkflowMode}
            parallelMode={parallelMode}
            setParallelMode={setParallelMode}
            maxParallelism={maxParallelism}
            setMaxParallelism={setMaxParallelism}
            remoteMachines={remoteMachines}
            selectedMachineId={selectedMachineId}
            setSelectedMachineId={setSelectedMachineId}
            setRemoteMachines={setRemoteMachines}
            addRemoteMachine={addRemoteMachine}
            submitParams={submitParams}
            terminateExecution={terminateExecution}
          />
        ) : (
          <FilePanel
            files={files}
            artifactTasks={artifactTasks}
            previewFile={previewFile}
            deleteFile={deleteFile}
            downloadUrl={(file) => client.getDownloadUrl(sessionId, file.name)}
            artifactUrl={(artifact) => client.getDownloadUrl(sessionId, artifact.workspace_path || '')}
            downloadArtifact={downloadArtifact}
            stageArtifact={stageArtifact}
            stagingArtifactIds={stagingArtifactIds}
          />
        )}
      </aside>

      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <header>
              <strong>{preview.file.name}</strong>
              <button onClick={() => setPreview(null)}>Close</button>
            </header>
            {preview.imageUrl ? <img src={preview.imageUrl} alt={preview.file.name} /> : <pre>{preview.content}</pre>}
            <a href={client.getDownloadUrl(sessionId, preview.file.name)}>Download</a>
          </div>
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message, matterViz, serverUrl, sessionId, responding }: { message: ChatMessage; matterViz: typeof defaultMatterViz; serverUrl: string; sessionId: string; responding: boolean }) {
  const isUser = message.role === 'user';
  const isToolBatch = !isUser && Boolean(message.tools?.length || message.stage_id?.startsWith('tool-batch-'));
  return (
    <article className={isUser ? 'message user' : 'message assistant'}>
      <div className="avatar">{isUser ? 'U' : 'AI'}</div>
      <div className="message-content">
        {isUser ? <p>{message.content}</p> : isToolBatch ? (
          <CollapsedToolResult message={message} matterViz={matterViz} serverUrl={serverUrl} sessionId={sessionId} />
        ) : (
          <MarkdownWithVisualize content={message.content} matterViz={matterViz} serverUrl={serverUrl} />
        )}
        {responding && <span className="typing">...</span>}
        {!isUser && message.usage_metadata && (
          <small className="metadata">输入 token {formatNumber(message.usage_metadata.prompt_tokens || 0)} · 输出 token {formatNumber(message.usage_metadata.candidates_tokens || 0)} · 合计 {formatNumber(message.usage_metadata.total_tokens || 0)}</small>
        )}
        {!isUser && message.charge_result && (
          <small className="metadata">光子 {message.charge_result.photon_amount || 0} · RMB {(message.charge_result.rmb_amount || 0).toFixed(2)}</small>
        )}
      </div>
    </article>
  );
}

function CollapsedToolResult({ message, matterViz, serverUrl, sessionId }: { message: ChatMessage; matterViz: typeof defaultMatterViz; serverUrl: string; sessionId: string }) {
  const { content } = message;
  const lines = content.split(/\r?\n/);
  const summary = lines.find((line) => line.trim()) || 'Tool batch completed';
  const summaryIndex = lines.indexOf(summary);
  const details = lines.slice(summaryIndex + 1).join('\n').trim();
  const toolResults: ToolResult[] = message.tool_results?.length
    ? message.tool_results
    : [{ tool: message.tools?.join(', ') || 'tool', input: {}, result: details || content }];
  const client = new DptbPilotClient(serverUrl);
  return (
    <details className="tool-result-details">
      <summary>{summary}</summary>
      <div className="tool-result-body">
        {toolResults.map((toolResult, index) => (
          <section className="tool-result-item" key={`${toolResult.tool}-${index}`}>
            <div className="tool-result-heading">
              <code>{toolResult.tool}</code>
              {toolResult.status ? <small>{toolResult.status}</small> : null}
            </div>
            <ToolResultSection label="Input">
              <pre>{formatToolValue(toolResult.input ?? {})}</pre>
            </ToolResultSection>
            <ToolResultSection label="Log">
              {toolResult.logs?.length ? toolResult.logs.map((log, logIndex) => (
                <div className="tool-log" key={`${log.path}-${logIndex}`}>
                  {log.workspace_path ? (
                    <a href={client.getDownloadUrl(sessionId, log.workspace_path)} title={log.path}>{log.workspace_path}</a>
                  ) : <code>{log.path}</code>}
                  {log.content !== undefined ? <pre>{log.content}</pre> : null}
                  {log.truncated ? <small>Showing the last 64 KB. Open the file for the complete log.</small> : null}
                </div>
              )) : <small>No remote log file is available for this tool run.</small>}
            </ToolResultSection>
            <ToolResultSection label="Result">
              {typeof toolResult.result === 'string'
                ? <MarkdownWithVisualize content={toolResult.result} matterViz={matterViz} serverUrl={serverUrl} />
                : <pre>{formatToolValue(toolResult.result)}</pre>}
            </ToolResultSection>
          </section>
        ))}
      </div>
    </details>
  );
}

function ToolResultSection({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="tool-result-section">
      <summary>{label}</summary>
      <div className="tool-result-section-body">{children}</div>
    </details>
  );
}

function formatToolValue(value: unknown): string {
  if (value === undefined) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function MarkdownWithVisualize({ content, matterViz, serverUrl }: { content: string; matterViz: typeof defaultMatterViz; serverUrl: string }) {
  const visualizeRegex = /:::visualize\n([\s\S]*?)\n:::/g;
  const parts: Array<{ type: 'markdown' | 'visualize'; content: string }> = [];
  let cursor = 0;
  for (const match of content.matchAll(visualizeRegex)) {
    if (match.index && match.index > cursor) parts.push({ type: 'markdown', content: content.slice(cursor, match.index) });
    parts.push({ type: 'visualize', content: match[1] });
    cursor = (match.index || 0) + match[0].length;
  }
  if (cursor < content.length) parts.push({ type: 'markdown', content: content.slice(cursor) });

  if (!parts.length) return <Markdown content={content} serverUrl={serverUrl} />;
  return <>{parts.map((part, index) => part.type === 'markdown'
    ? <Markdown key={index} content={part.content} serverUrl={serverUrl} />
    : <VisualizationCard key={index} json={part.content} matterViz={matterViz} />)}</>;
}

function Markdown({ content, serverUrl }: { content: string; serverUrl: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        img({ src, alt, ...props }) {
          const resolvedSrc = typeof src === 'string' ? resolveServerResourceUrl(serverUrl, src) : src;
          return <img src={resolvedSrc} alt={alt || ''} {...props} />;
        },
        code({ className, children, ...props }) {
          if (className === 'language-mermaid') {
            return <MermaidDiagram source={String(children).replace(/\n$/, '')} />;
          }
          return <code className={className} {...props}>{children}</code>;
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function MermaidDiagram({ source }: { source: string }) {
  const diagram = parseWorkflowDiagram(source);
  if (!diagram.nodes.length) {
    return <div className="mermaid-error"><strong>Unsupported diagram</strong><pre>{source}</pre></div>;
  }

  const nodeWidth = 170;
  const nodeHeight = 72;
  const gap = 54;
  const padding = 18;
  const positions = new Map(diagram.nodes.map((node, index) => [node.id, {
    x: padding + index * (nodeWidth + gap),
    y: padding
  }]));
  const width = padding * 2 + diagram.nodes.length * nodeWidth + Math.max(diagram.nodes.length - 1, 0) * gap;
  const height = nodeHeight + padding * 2;

  return (
    <div className="mermaid-diagram" aria-label="Workflow diagram">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          <marker id="workflow-arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" className="workflow-arrowhead" />
          </marker>
        </defs>
        {diagram.edges.map(([from, to]) => {
          const start = positions.get(from);
          const end = positions.get(to);
          if (!start || !end) return null;
          return <line key={`${from}-${to}`} x1={start.x + nodeWidth} y1={start.y + nodeHeight / 2} x2={end.x - 7} y2={end.y + nodeHeight / 2} markerEnd="url(#workflow-arrowhead)" />;
        })}
        {diagram.nodes.map((node) => {
          const position = positions.get(node.id)!;
          const label = splitDiagramLabel(node.label);
          return (
            <g key={node.id}>
              <rect x={position.x} y={position.y} width={nodeWidth} height={nodeHeight} rx="5" />
              <text x={position.x + nodeWidth / 2} y={position.y + 27} textAnchor="middle">
                <tspan className="diagram-node-id">{label[0]}</tspan>
                {label[1] ? <tspan x={position.x + nodeWidth / 2} dy="20">{label[1]}</tspan> : null}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function parseWorkflowDiagram(source: string) {
  const nodes: Array<{ id: string; label: string }> = [];
  const edges: Array<[string, string]> = [];
  for (const line of source.split(/\r?\n/)) {
    const node = line.match(/^\s*([\w-]+)\s*\["([\s\S]+)"\]\s*$/);
    if (node) {
      nodes.push({ id: node[1], label: node[2] });
      continue;
    }
    const edge = line.match(/^\s*([\w-]+)\s*-->\s*([\w-]+)\s*$/);
    if (edge) edges.push([edge[1], edge[2]]);
  }
  return { nodes, edges };
}

function splitDiagramLabel(label: string): [string, string] {
  const separator = label.indexOf(':');
  if (separator < 0) return [label.slice(0, 24), label.slice(24, 48)];
  return [label.slice(0, separator), label.slice(separator + 1).trim().slice(0, 34)];
}

function VisualizationCard({ json, matterViz }: { json: string; matterViz: typeof defaultMatterViz }) {
  const autoOpenedRef = useRef(false);
  const parsed = parseVisualizationPayload(json);
  const payloadKey = parsed.payload ? JSON.stringify(parsed.payload) : '';
  const matterVizConfigured = Boolean(matterViz.commandId || matterViz.uriTemplate);

  function openMatterViz() {
    if (!parsed.payload) return;
    vscode?.postMessage({ type: 'openInMatterViz', payload: { visualization: parsed.payload } });
  }

  useEffect(() => {
    if (!parsed.payload || !matterViz.autoOpen || !matterVizConfigured || autoOpenedRef.current) return;
    autoOpenedRef.current = true;
    openMatterViz();
  }, [matterViz.autoOpen, matterVizConfigured, payloadKey]);

  if (!parsed.payload) {
    return <div className="visualize-box"><strong>Visualization payload</strong><pre>{parsed.error || json}</pre></div>;
  }

  const payload = parsed.payload;
  return (
    <div className="visualize-box">
      <header className="visualize-header">
        <div>
          <strong>{payload.title || visualizationTitle(payload.kind)}</strong>
          <small>{payload.source?.file || payload.source?.tool || payload.kind}</small>
        </div>
        <button type="button" onClick={openMatterViz}>MatterViz</button>
      </header>
      <VisualizationBody payload={payload} />
      {!matterVizConfigured && <small>Set dptbPilot.matterViz.commandId or dptbPilot.matterViz.uriTemplate to enable handoff.</small>}
    </div>
  );
}

function VisualizationBody({ payload }: { payload: VisualizationPayload }) {
  if (payload.kind === 'bz') return <BzPreview payload={payload} />;
  if (payload.kind === 'band' || payload.kind === 'dos' || payload.kind === 'curve') return <CurvePreview payload={payload} />;
  if (payload.kind === 'image') return <ImagePreview payload={payload} />;
  if (payload.kind === 'table') return <TablePreview payload={payload} />;
  if (payload.kind === 'structure') return <StructureSummary payload={payload} />;
  return <pre>{JSON.stringify(payload.data, null, 2)}</pre>;
}

function StructureSummary({ payload }: { payload: VisualizationPayload }) {
  const text = typeof payload.data === 'string' ? payload.data : JSON.stringify(payload.data, null, 2);
  const lineCount = text.split(/\r?\n/).filter(Boolean).length;
  return (
    <div className="visualize-summary">
      <span>Format: {String(payload.meta?.format || payload.source?.file?.split('.').pop() || 'structure')}</span>
      <span>Lines: {lineCount}</span>
      <pre>{text.slice(0, 1200)}</pre>
    </div>
  );
}

function BzPreview({ payload }: { payload: VisualizationPayload }) {
  const data = asRecord(payload.data);
  const vertices = Array.isArray(data.vertices) ? data.vertices as number[][] : [];
  const edges = Array.isArray(data.edges) ? data.edges as number[][] : [];
  const points = vertices
    .filter((vertex) => Array.isArray(vertex) && vertex.length >= 2)
    .map((vertex) => [Number(vertex[0]), Number(vertex[1])] as [number, number]);
  const project = makeProjector(points, 260, 180);

  return (
    <div className="visualize-preview">
      <svg viewBox="0 0 260 180" role="img" aria-label="Brillouin zone projection">
        {edges.map((edge, index) => {
          const start = vertices[edge[0]];
          const end = vertices[edge[1]];
          if (!start || !end) return null;
          const [x1, y1] = project(Number(start[0]), Number(start[1]));
          const [x2, y2] = project(Number(end[0]), Number(end[1]));
          return <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} />;
        })}
      </svg>
      <small>{vertices.length} vertices, {edges.length} edges</small>
    </div>
  );
}

function CurvePreview({ payload }: { payload: VisualizationPayload }) {
  const points = extractPoints(payload.data);
  if (!points.length) return <pre>{JSON.stringify(payload.data, null, 2)}</pre>;

  const project = makeProjector(points, 300, 180);
  const path = points.map(([x, y], index) => {
    const [px, py] = project(x, y);
    return `${index === 0 ? 'M' : 'L'} ${px} ${py}`;
  }).join(' ');

  return (
    <div className="visualize-preview">
      <svg viewBox="0 0 300 180" role="img" aria-label={`${payload.kind} preview`}>
        <path d={path} />
      </svg>
      <small>{points.length} points</small>
    </div>
  );
}

function ImagePreview({ payload }: { payload: VisualizationPayload }) {
  const data = asRecord(payload.data);
  const src = typeof data.url === 'string' ? data.url : typeof data.src === 'string' ? data.src : typeof payload.data === 'string' ? payload.data : '';
  return src ? <img className="visualize-image" src={src} alt={payload.title || 'DPTB visualization'} /> : <pre>{JSON.stringify(payload.data, null, 2)}</pre>;
}

function TablePreview({ payload }: { payload: VisualizationPayload }) {
  const data = asRecord(payload.data);
  const rows = Array.isArray(data.rows) ? data.rows as Record<string, unknown>[] : Array.isArray(payload.data) ? payload.data as Record<string, unknown>[] : [];
  const columns = Array.isArray(data.columns) ? data.columns.map(String) : Object.keys(rows[0] || {});
  if (!rows.length || !columns.length) return <pre>{JSON.stringify(payload.data, null, 2)}</pre>;

  return (
    <div className="visualize-table-wrap">
      <table className="visualize-table">
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>{rows.slice(0, 12).map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? '')}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function parseVisualizationPayload(json: string): { payload?: VisualizationPayload; error?: string } {
  try {
    const parsed = JSON.parse(json);
    if (parsed && typeof parsed === 'object' && typeof parsed.kind === 'string') {
      return { payload: { kind: parsed.kind, title: parsed.title, source: parsed.source, data: parsed.data, meta: parsed.meta } as VisualizationPayload };
    }
    if (parsed && typeof parsed === 'object' && typeof parsed.format === 'string') {
      const format = parsed.format.toLowerCase();
      const kind = format === 'bz' ? 'bz' : 'structure';
      return { payload: { kind, title: kind === 'bz' ? 'Brillouin Zone' : 'Structure', data: parsed.data, meta: { format } } };
    }
    return { error: JSON.stringify(parsed, null, 2) };
  } catch (error) {
    return { error: errorMessage(error) };
  }
}

function visualizationTitle(kind: VisualizationPayload['kind']) {
  return {
    structure: 'Structure',
    bz: 'Brillouin Zone',
    band: 'Band Structure',
    dos: 'Density of States',
    curve: 'Curve',
    image: 'Image',
    table: 'Table',
    text: 'Text'
  }[kind];
}

function extractPoints(data: unknown): [number, number][] {
  const record = asRecord(data);
  const source = Array.isArray(data) ? data : Array.isArray(record.points) ? record.points : Array.isArray(record.series) ? record.series : [];
  return source.map((item: unknown, index: number) => {
    if (Array.isArray(item)) return [Number(item[0]), Number(item[1])] as [number, number];
    const point = asRecord(item);
    return [Number(point.x ?? index), Number(point.y ?? point.value)] as [number, number];
  }).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
}

function makeProjector(points: [number, number][], width: number, height: number) {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);
  const pad = 16;

  return (x: number, y: number): [number, number] => [
    pad + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad * 2),
    height - pad - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad * 2)
  ];
}

function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === 'object' ? value as Record<string, any> : {};
}

function WorkflowPanel({
  plan,
  submitting,
  decide
}: {
  plan: WorkflowPlan | null;
  submitting: WorkflowDecision | null;
  decide: (decision: WorkflowDecision) => Promise<void>;
}) {
  if (!plan) {
    return (
      <div className="panel-content">
        <div className="empty-state">
          <strong>No workflow waiting for review</strong>
          <p>Complex tasks will appear here before Pilot starts execution.</p>
        </div>
      </div>
    );
  }

  const reviewGroups: Array<[string, string[] | undefined]> = plan.review ? [
    ['Scientific logic', plan.review.scientific_logic],
    ['Execution logic', plan.review.execution_logic],
    ['Blocking issues', plan.review.blocking_issues],
    ['Warnings', plan.review.warnings],
    ['Recommendations', plan.review.recommendations]
  ] : [];

  return (
    <div className="panel-content workflow-panel">
      <header className="workflow-heading">
        <small>Waiting for review</small>
        <h2>{plan.title || 'Workflow plan'}</h2>
        <p>{plan.summary}</p>
      </header>

      <div className="workflow-graph" role="img" aria-label="Workflow dependency graph">
        {(plan.steps || []).map((step, index) => (
          <div className="workflow-node-wrap" key={step.id}>
            <div className="workflow-node">
              <strong>{step.id}</strong>
              <span>{step.title}</span>
            </div>
            {index < plan.steps.length - 1 ? <span className="workflow-arrow" aria-hidden="true">-&gt;</span> : null}
          </div>
        ))}
      </div>

      <section className="workflow-steps">
        {(plan.steps || []).map((step) => (
          <details key={step.id} open>
            <summary><strong>{step.id}</strong><span>{step.title}</span></summary>
            <p>{step.purpose}</p>
            {step.depends_on?.length ? <small>Depends on: {step.depends_on.join(', ')}</small> : null}
            {step.tools?.length ? (
              <div className="workflow-tools">
                {step.tools.map((tool) => <code key={tool}>{tool}</code>)}
              </div>
            ) : null}
          </details>
        ))}
      </section>

      {plan.risks?.length ? <WorkflowList title="Risks / checks" items={plan.risks} tone="warning" /> : null}
      {plan.questions?.length ? <WorkflowList title="Open questions" items={plan.questions} tone="warning" /> : null}

      {plan.review ? (
        <section className={`workflow-review verdict-${plan.review.verdict}`}>
          <div className="workflow-review-title">
            <strong>Pilot review</strong>
            <span>{plan.review.verdict.replace('_', ' ')}</span>
          </div>
          <p>{plan.review.summary}</p>
          {reviewGroups.map(([title, items]) => items?.length ? <WorkflowList key={title} title={title} items={items} /> : null)}
        </section>
      ) : null}

      <div className="workflow-actions">
        <button type="button" disabled={Boolean(submitting)} onClick={() => void decide('pilot_review')}>
          {submitting === 'pilot_review' ? 'Reviewing...' : 'Pilot Review'}
        </button>
        <div className="inline-row">
          <button className="primary" type="button" disabled={Boolean(submitting)} onClick={() => void decide('approved')}>
            {submitting === 'approved' ? 'Approving...' : 'Approve'}
          </button>
          <button className="danger" type="button" disabled={Boolean(submitting)} onClick={() => void decide('rejected')}>
            {submitting === 'rejected' ? 'Rejecting...' : 'Reject'}
          </button>
        </div>
      </div>
    </div>
  );
}

function WorkflowList({ title, items, tone }: { title: string; items: string[]; tone?: 'warning' }) {
  return (
    <section className={`workflow-list ${tone || ''}`}>
      <strong>{title}</strong>
      <ul>{items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}</ul>
    </section>
  );
}

function ParameterPanel(props: {
  schema: ToolSchema | null;
  schemaJson: string;
  setSchemaJson: (value: string) => void;
  modifyMode: ModifyMode;
  setModifyMode: (value: ModifyMode) => void;
  executionMode: ExecutionMode;
  setExecutionMode: (value: ExecutionMode) => void;
  workflowMode: WorkflowMode;
  setWorkflowMode: (value: WorkflowMode) => void;
  parallelMode: 'sequential' | 'auto' | 'parallel';
  setParallelMode: (value: 'sequential' | 'auto' | 'parallel') => void;
  maxParallelism: number;
  setMaxParallelism: (value: number) => void;
  remoteMachines: RemoteMachine[];
  selectedMachineId: string;
  setSelectedMachineId: (value: string) => void;
  setRemoteMachines: (value: RemoteMachine[] | ((old: RemoteMachine[]) => RemoteMachine[])) => void;
  addRemoteMachine: (type: 'Bohrium' | 'Slurm', name: string) => void;
  submitParams: (formData?: FormData) => Promise<void>;
  terminateExecution: () => Promise<void>;
}) {
  const selectedMachine = props.remoteMachines.find((machine) => machine.id === props.selectedMachineId) || null;
  const [pendingMachineType, setPendingMachineType] = useState<'Bohrium' | 'Slurm' | null>(null);
  const [pendingMachineName, setPendingMachineName] = useState('');

  function beginAddRemoteMachine(type: 'Bohrium' | 'Slurm') {
    setPendingMachineType(type);
    setPendingMachineName('');
  }

  function confirmAddRemoteMachine() {
    const name = pendingMachineName.trim();
    if (!pendingMachineType || !name) return;
    props.addRemoteMachine(pendingMachineType, name);
    setPendingMachineType(null);
    setPendingMachineName('');
  }

  return (
    <div className="panel-content">
      <label>Execution mode</label>
      <select value={props.executionMode} onChange={(event) => props.setExecutionMode(event.target.value as ExecutionMode)}>
        <option value="Local">Local</option>
        <option value="Remote">Remote</option>
      </select>

      <label>Workflow planning</label>
      <select value={props.workflowMode} onChange={(event) => props.setWorkflowMode(event.target.value as WorkflowMode)}>
        <option value="always">Always</option>
        <option value="auto">Automatic</option>
        <option value="disabled">Disabled</option>
      </select>

      <label>Remote submission</label>
      <select value={props.parallelMode} onChange={(event) => props.setParallelMode(event.target.value as 'sequential' | 'auto' | 'parallel')}>
        <option value="auto">Automatic parallelism</option>
        <option value="parallel">Parallel submission</option>
        <option value="sequential">Sequential submission</option>
      </select>
      <label>Maximum parallel jobs</label>
      <input
        type="number"
        min={1}
        max={32}
        value={props.maxParallelism}
        onChange={(event) => props.setMaxParallelism(Math.min(32, Math.max(1, Number(event.target.value) || 1)))}
      />

      {props.executionMode === 'Remote' && (
        <section className="remote-machines">
          <label>Remote machine
            <select value={props.selectedMachineId} onChange={(event) => props.setSelectedMachineId(event.target.value)}>
              <option value="">Select a remote machine</option>
              {props.remoteMachines.map((machine) => <option key={machine.id} value={machine.id}>{machine.name} ({machine.type})</option>)}
            </select>
          </label>
          <div className="inline-row remote-actions">
            <button type="button" onClick={() => beginAddRemoteMachine('Bohrium')}>Add Bohrium</button>
            <button type="button" onClick={() => beginAddRemoteMachine('Slurm')}>Add Slurm</button>
          </div>
          {pendingMachineType && (
            <div className="remote-machine-name-editor">
              <label>{pendingMachineType} machine name
                <input
                  autoFocus
                  value={pendingMachineName}
                  onChange={(event) => setPendingMachineName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      confirmAddRemoteMachine();
                    } else if (event.key === 'Escape') {
                      setPendingMachineType(null);
                    }
                  }}
                  placeholder={`My ${pendingMachineType} machine`}
                />
              </label>
              <div className="inline-row">
                <button className="primary" type="button" disabled={!pendingMachineName.trim()} onClick={confirmAddRemoteMachine}>Add</button>
                <button type="button" onClick={() => setPendingMachineType(null)}>Cancel</button>
              </div>
            </div>
          )}
          {selectedMachine && <small className="selected-machine">Selected: {selectedMachine.name} ({selectedMachine.type})</small>}
          {props.remoteMachines.map((machine) => (
            <details key={machine.id} open={machine.id === props.selectedMachineId}>
              <summary>{machine.name} ({machine.type})</summary>
              <textarea value={JSON.stringify(machine.config, null, 2)} onChange={(event) => {
                try {
                  const config = JSON.parse(event.target.value);
                  props.setRemoteMachines((old) => old.map((item) => item.id === machine.id ? { ...item, config } : item));
                } catch {
                  // ignore invalid JSON while editing
                }
              }} />
            </details>
          ))}
        </section>
      )}

      <div className="inline-row">
        <button className={props.modifyMode === 'individual' ? 'selected' : ''} onClick={() => props.setModifyMode('individual')}>Form</button>
        <button className={props.modifyMode === 'json' ? 'selected' : ''} onClick={() => props.setModifyMode('json')}>JSON</button>
      </div>

      {!props.schema ? <div className="empty-state">Waiting for an agent schema.</div> : (
        <form onSubmit={(event) => { event.preventDefault(); void props.submitParams(new FormData(event.currentTarget)); }}>
          <h3>{props.schema.name}</h3>
          {props.schema.description && <p>{props.schema.description}</p>}
          {props.modifyMode === 'json' ? (
            <textarea className="json-editor" value={props.schemaJson} onChange={(event) => props.setSchemaJson(event.target.value)} />
          ) : Object.entries(props.schema.input_schema.properties || {}).map(([key, prop]) => {
            if (key.toLowerCase() === 'executor' || key.toLowerCase() === 'storage') return null;
            return <label key={key}>{prop.title || key}<small>{prop.type}</small><input name={key} defaultValue={String(prop.agent_input ?? prop.default ?? '')} /></label>;
          })}
          <div className="inline-row">
            <button className="primary" type="submit">Submit</button>
            <button type="button" onClick={() => void props.terminateExecution()}>Stop</button>
          </div>
        </form>
      )}
    </div>
  );
}

function FilePanel({ files, artifactTasks, previewFile, deleteFile, downloadUrl, artifactUrl, downloadArtifact, stageArtifact, stagingArtifactIds }: {
  files: FileInfo[];
  artifactTasks: ArtifactTask[];
  previewFile: (file: FileInfo) => void;
  deleteFile: (file: FileInfo) => void;
  downloadUrl: (file: FileInfo) => string;
  artifactUrl: (artifact: ArtifactInfo) => string;
  downloadArtifact: (artifact: ArtifactInfo) => void;
  stageArtifact: (artifact: ArtifactInfo) => Promise<void>;
  stagingArtifactIds: Set<string>;
}) {
  const groups = groupFiles(files);
  const artifactCount = artifactTasks.reduce((count, task) => count + task.artifacts.length, 0);
  return (
    <div className="panel-content">
      {artifactTasks.length ? (
        <details className="artifact-section" open>
          <summary>Task Artifacts ({artifactCount})</summary>
          <div className="artifact-task-list">
            {artifactTasks.map((task) => (
              <details className="artifact-task" key={task.id}>
                <summary>
                  <strong title={task.task}>{task.task}</strong>
                  <small>{new Date(task.completed_at).toLocaleString()} · {task.artifacts.length}</small>
                </summary>
                {task.artifacts.map((artifact) => (
                  <article className="artifact-item" key={artifact.id}>
                    <dl>
                      <div><dt>Source path</dt><dd title={artifact.source_path}>{artifact.source_path}</dd></div>
                      {artifact.workspace_path ? <div><dt>Workspace path</dt><dd title={artifact.workspace_path}>{artifact.workspace_path}</dd></div> : null}
                    </dl>
                    {artifact.preview_kind === 'image' && artifact.exists && artifact.workspace_path ? (
                      <img src={artifactUrl(artifact)} alt={artifact.file_name || 'artifact'} />
                    ) : null}
                    {artifact.preview_kind === 'text' && artifact.content_preview ? <pre>{artifact.content_preview}</pre> : null}
                    {artifact.preview_kind === 'binary' ? <small>Binary content{artifact.size ? ` · ${(artifact.size / 1024).toFixed(1)} KB` : ''}</small> : null}
                    {artifact.preview_kind === 'unavailable' ? <small className="artifact-unavailable">This artifact is not staged in the workspace yet.</small> : null}
                    {artifact.exists && artifact.workspace_path ? (
                      <button className="artifact-download" onClick={() => downloadArtifact(artifact)}>
                        Download to local...
                      </button>
                    ) : (artifact.can_stage ?? /^(?:bohrium|oss|https?):\/\//i.test(artifact.source_path)) ? (
                      <button
                        className="artifact-download"
                        disabled={stagingArtifactIds.has(artifact.id)}
                        onClick={() => void stageArtifact(artifact)}
                      >
                        {stagingArtifactIds.has(artifact.id) ? 'Downloading...' : 'Download to workspace'}
                      </button>
                    ) : null}
                  </article>
                ))}
              </details>
            ))}
          </div>
        </details>
      ) : null}
      {Object.entries(groups).map(([group, groupFiles]) => groupFiles.length ? (
        <section key={group}>
          <h3>{group} ({groupFiles.length})</h3>
          {groupFiles.map((file) => (
            <div className="file-row" key={file.name}>
              <button onClick={() => void previewFile(file)}>
                {file.name}
                <small>{(file.size / 1024).toFixed(1)} KB</small>
              </button>
              <a href={downloadUrl(file)}>Download</a>
              <button onClick={() => void deleteFile(file)}>Delete</button>
            </div>
          ))}
        </section>
      ) : null)}
      {!files.length && !artifactTasks.length && <div className="empty-state">No files or task artifacts yet.</div>}
    </div>
  );
}

function generateRemoteProperty(key: string, prop: ToolSchema['input_schema']['properties'][string], machine: RemoteMachine, executionMode: ExecutionMode) {
  if (executionMode !== 'Remote') return null;
  const lowerKey = key.toLowerCase();
  if (lowerKey === 'executor') return generateExecutorProperty(prop, machine);
  if (lowerKey === 'storage' && machine.type === 'Bohrium') return generateBohriumStorageProperty(prop, machine.config as BohriumConfig);
  return null;
}

function generateExecutorProperty(prop: ToolSchema['input_schema']['properties'][string], machine: RemoteMachine) {
  if (machine.type === 'Bohrium') {
    const config = machine.config as BohriumConfig;
    return {
      ...prop,
      user_input: {
        type: 'dispatcher',
        machine: {
          batch_type: 'Bohrium',
          context_type: 'Bohrium',
          remote_profile: {
            email: config.username,
            password: config.password,
            program_id: parseInt(config.project_id, 10),
            input_data: {
              image_name: config.image_name || 'registry.dp.tech/dptech/dp/native/prod-19853/dpa-mcp:0.0.0',
              job_type: 'container',
              platform: 'ali',
              scass_type: config.scass_type || '1 * NVIDIA V100_32g'
            }
          }
        }
      }
    };
  }

  const config = machine.config as SlurmConfig;
  return {
    ...prop,
    user_input: {
      type: 'dispatcher',
      machine: {
        batch_type: 'Slurm',
        context_type: 'SSHContext',
        local_root: './',
        remote_root: config.remote_root,
        remote_profile: {
          hostname: config.hostname,
          username: config.username,
          timeout: 600,
          port: 22,
          key_filename: config.key_filename
        }
      },
      resources: {
        number_node: parseInt(config.number_node, 10),
        gpu_per_node: config.gpu_per_node ? parseInt(config.gpu_per_node, 10) : 0,
        cpu_per_node: config.cpu_per_node ? parseInt(config.cpu_per_node, 10) : 1,
        queue_name: config.queue_name,
        custom_flags: [config.custom_flags || '', ''],
        source_list: [],
        module_list: []
      }
    }
  };
}

function generateBohriumStorageProperty(prop: ToolSchema['input_schema']['properties'][string], config: BohriumConfig) {
  return {
    ...prop,
    user_input: {
      type: 'bohrium',
      username: config.username,
      password: config.password,
      project_id: config.project_id
    }
  };
}

function makeSession(userId: string): ChatSession {
  const now = new Date();
  return {
    chat_id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    user_id: userId,
    title: `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`,
    history: [],
    created_at: now.toISOString(),
    last_active: now.toISOString(),
    message_count: 0
  };
}

function createRemoteMachineId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function remoteMachineSecretKey(machineId: string): string {
  return `dptbPilot.remoteMachine.${machineId}.password`;
}

function isMissingExecutionConfigEndpoint(error: unknown): boolean {
  return error instanceof DptbPilotApiError && error.status === 404;
}

function executionConfigCompatibilityMessage(): string {
  return '已连接到旧版后端；该后端不支持连接后更新远程执行配置。请更新并重启 dptb-pilot 后端，当前仍可使用原有参数确认模式。';
}

function groupFiles(files: FileInfo[]) {
  return files.reduce((acc, file) => {
    const ext = extensionOf(file.name);
    const key = ['xyz', 'cif', 'poscar', 'vasp', 'xsf'].includes(ext) ? 'structure' : isImage(ext) || ext === 'pdf' ? 'plot' : ['json', 'csv', 'txt', 'log', 'out', 'npy', 'npz'].includes(ext) ? 'data' : 'other';
    acc[key].push(file);
    return acc;
  }, { structure: [] as FileInfo[], plot: [] as FileInfo[], data: [] as FileInfo[], other: [] as FileInfo[] });
}

function parseFormValue(value: string, type: string) {
  if (type === 'number') return Number(value);
  if (type === 'integer') return parseInt(value, 10);
  if (type === 'boolean') return value === 'true' || value === 'on';
  if (type === 'array' || type === 'object') {
    try { return JSON.parse(value); } catch { return value; }
  }
  return value;
}

function extensionOf(filename: string) {
  return filename.split('.').pop()?.toLowerCase() || '';
}

function isImage(ext: string) {
  return ['png', 'jpg', 'jpeg', 'svg', 'bmp', 'gif', 'webp'].includes(ext);
}

function readLocal<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) as T : fallback;
  } catch {
    return fallback;
  }
}

function normalizeUsage(value: unknown): ChatMessage['usage_metadata'] | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Record<string, unknown>;
  const numberValue = (...keys: string[]) => {
    for (const key of keys) {
      const parsed = Number(raw[key]);
      if (Number.isFinite(parsed)) return Math.max(0, parsed);
    }
    return 0;
  };
  const prompt_tokens = numberValue('prompt_tokens', 'promptTokenCount', 'prompt_token_count', 'input_tokens');
  const candidates_tokens = numberValue('candidates_tokens', 'candidatesTokenCount', 'candidates_token_count', 'completion_tokens', 'output_tokens');
  const total_tokens = numberValue('total_tokens', 'totalTokenCount', 'total_token_count') || prompt_tokens + candidates_tokens;
  if (!prompt_tokens && !candidates_tokens && !total_tokens) return undefined;
  return { prompt_tokens, candidates_tokens, total_tokens };
}

function addUsage(left: ChatMessage['usage_metadata'] | undefined, right: ChatMessage['usage_metadata'] | undefined) {
  if (!left && !right) return undefined;
  return {
    prompt_tokens: (left?.prompt_tokens || 0) + (right?.prompt_tokens || 0),
    candidates_tokens: (left?.candidates_tokens || 0) + (right?.candidates_tokens || 0),
    total_tokens: (left?.total_tokens || 0) + (right?.total_tokens || 0),
  };
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value || 0);
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}


