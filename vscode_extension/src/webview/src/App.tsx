import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { DptbPilotClient } from '../../api/client';
import { DEFAULT_SERVER_URL, generateSessionId, isValidSessionId, normalizeBaseUrl } from '../../api/url';
import type {
  ChatMessage,
  ChatSession,
  ExecutionMode,
  FileInfo,
  HostToWebviewMessage,
  ModifyMode,
  BohriumConfig,
  RemoteMachine,
  SlurmConfig,
  ToolSchema,
  VisualizationPayload,
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
  const [loading, setLoading] = useState(false);
  const [responding, setResponding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>('');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [input, setInput] = useState('');
  const [schema, setSchema] = useState<ToolSchema | null>(null);
  const [modifyMode, setModifyMode] = useState<ModifyMode>('individual');
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('Local');
  const [schemaJson, setSchemaJson] = useState('');
  const [activePanel, setActivePanel] = useState<'params' | 'files'>('params');
  const [preview, setPreview] = useState<{ file: FileInfo; content?: string; imageUrl?: string } | null>(null);
  const [remoteMachines, setRemoteMachines] = useState<RemoteMachine[]>(() => readLocal('dptb.remoteMachines', []));
  const [selectedMachineId, setSelectedMachineId] = useState<string>(() => readLocal('dptb.selectedMachineId', ''));
  const [matterViz, setMatterViz] = useState(defaultMatterViz);

  const wsRef = useRef<WebSocket | null>(null);
  const currentChatIdRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const client = useMemo(() => new DptbPilotClient(activeServerUrl), [activeServerUrl]);
  const currentSession = sessions.find((item) => item.chat_id === currentChatId) || null;

  useEffect(() => {
    vscode?.postMessage({ type: 'ready' });

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
        setMatterViz({ ...defaultMatterViz, ...message.payload.matterViz });
      } else if (message.type === 'serverUrlChanged') {
        setServerUrl(message.payload.serverUrl);
        try {
          setActiveServerUrl(normalizeBaseUrl(message.payload.serverUrl));
        } catch {
          setError('服务器地址无效，请输入正确的 HTTP 或 HTTPS 地址。');
        }
        setConnected(false);
      } else if (message.type === 'sessionChanged') {
        setSessionId(message.payload.sessionId);
      } else if (message.type === 'uploadFile') {
        void uploadHostFile(message.payload.name, new Uint8Array(message.payload.bytes));
      } else if (message.type === 'openNewSession') {
        void createNewSession();
      }
    };

    window.addEventListener('message', listener);
    return () => window.removeEventListener('message', listener);
  }, [client, sessionId, sessions]);

  useEffect(() => {
    currentChatIdRef.current = currentChatId;
  }, [currentChatId]);

  useEffect(() => {
    localStorage.setItem('dptb.remoteMachines', JSON.stringify(remoteMachines));
  }, [remoteMachines]);

  useEffect(() => {
    localStorage.setItem('dptb.selectedMachineId', selectedMachineId);
  }, [selectedMachineId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.history]);

  useEffect(() => {
    if (!connected || !sessionId) return;
    const interval = window.setInterval(() => void refreshSchema(), 1200);
    return () => window.clearInterval(interval);
  }, [connected, sessionId, client]);

  async function login(targetSessionId = sessionId) {
    const cleanSessionId = targetSessionId.trim();
    if (!isValidSessionId(cleanSessionId)) {
      setError('Session ID must be 32 alphanumeric characters.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
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
      await activeClient.login({ session_id: cleanSessionId });
      setConnected(true);
      await loadSessions(cleanSessionId, activeClient);
      await loadFiles(cleanSessionId, activeClient);
      connectWebSocket(cleanSessionId, activeClient);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
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

  function connectWebSocket(userId = sessionId, activeClient = client) {
    wsRef.current?.close();
    const ws = activeClient.createWebSocket(userId);
    wsRef.current = ws;
    ws.onmessage = (event) => handleWSMessage(DptbPilotClient.parseWSMessage(event.data));
    ws.onerror = () => setError('WebSocket connection failed. Please check that the backend is reachable.');
    ws.onclose = () => setResponding(false);
  }

  function handleWSMessage(message: WSMessage) {
    if (message.type === 'streaming_response') {
      appendAssistantContent(message.content || '', false, message, 'append');
    } else if (message.type === 'final_response') {
      appendAssistantContent(message.content || '', true, message, 'final');
      setResponding(false);
      void loadFiles();
    } else if (message.type === 'tool_modify_required') {
      if (message.schema) {
        setSchema(message.schema);
        setSchemaJson(JSON.stringify(message.schema, null, 2));
        setActivePanel('params');
      }
    } else if (message.type === 'error') {
      setError(message.message || '后端返回错误');
      setResponding(false);
    } else if (message.type === 'charge_result') {
      patchLastAssistant({ charge_result: message.charge_result });
    }
  }

  async function sendMessage(text = input) {
    const content = text.trim();
    if (!content || !currentSession || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const nextHistory: ChatMessage[] = [
      ...currentSession.history,
      { role: 'user', content, timestamp: new Date().toISOString() },
      { role: 'assistant', content: '' }
    ];
    updateCurrentSession({ history: nextHistory, last_active: new Date().toISOString(), message_count: nextHistory.length });
    setInput('');
    setResponding(true);
    wsRef.current.send(JSON.stringify({ message: content, chat_id: currentSession.chat_id }));
  }

  function appendAssistantContent(content: string, final: boolean, message: WSMessage, mode: 'append' | 'final') {
    setSessions((old) => old.map((session) => {
      if (session.chat_id !== currentChatIdRef.current) return session;
      const history = [...session.history];
      const last = history[history.length - 1];
      const finalPatch = final ? { timestamp: new Date().toISOString(), usage_metadata: message.usage_metadata, charge_result: message.charge_result } : {};
      if (last?.role === 'assistant') {
        const nextContent = mode === 'final' ? mergeFinalContent(last.content, content) : last.content + content;
        history[history.length - 1] = {
          ...last,
          content: nextContent,
          ...finalPatch
        };
      } else {
        history.push({ role: 'assistant', content, ...finalPatch });
      }
      return { ...session, history, last_active: new Date().toISOString(), message_count: history.length };
    }));
  }

  function patchLastAssistant(patch: Partial<ChatMessage>) {
    setSessions((old) => old.map((session) => {
      if (session.chat_id !== currentChatIdRef.current) return session;
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
      modified_schema: modifiedSchema,
      execution_mode: executionMode,
      selected_machine_id: selectedMachineId,
      remote_machine: selectedMachine
    });
    setSchema(null);
    setSchemaJson('');
  }

  async function terminateExecution() {
    const response = await client.terminateExecution({ session_id: sessionId });
    setResponding(false);
    setError(response.status === 'terminating' ? null : response.message);
  }

  async function uploadHostFile(name: string, bytes: Uint8Array) {
    if (!connected || !sessionId) return;
    await client.uploadFiles(sessionId, [{ name, bytes }]);
    await loadFiles();
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

  function addRemoteMachine(type: 'Bohrium' | 'Slurm') {
    const name = prompt('机器名称');
    if (!name) return;
    const machine: RemoteMachine = type === 'Bohrium'
      ? { id: String(Date.now()), name, type, config: { username: '', password: '', project_id: '', image_name: '', scass_type: '' } }
      : { id: String(Date.now()), name, type, config: { remote_root: '', hostname: '', username: '', key_filename: '', number_node: '1', queue_name: '' } };
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
          {currentSession?.history.map((message, index) => <MessageBubble key={index} message={message} matterViz={matterViz} responding={responding && index === currentSession.history.length - 1} />)}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <button title="Select files to upload" onClick={() => vscode?.postMessage({ type: 'pickFiles' })}>Upload</button>
          <button title="Upload current editor file" onClick={() => vscode?.postMessage({ type: 'uploadCurrentFile' })}>Current</button>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              void sendMessage();
            }
          }} placeholder="Type a message. Enter sends, Shift+Enter inserts a new line." />
          <button className="primary" disabled={!connected || !input.trim() || responding} onClick={() => void sendMessage()}>{responding ? 'Responding...' : 'Send'}</button>
        </div>
      </main>

      <aside className="right-panel">
        <div className="tabs">
          <button className={activePanel === 'params' ? 'selected' : ''} onClick={() => setActivePanel('params')}>Params</button>
          <button className={activePanel === 'files' ? 'selected' : ''} onClick={() => setActivePanel('files')}>Files</button>
        </div>
        {activePanel === 'params' ? (
          <ParameterPanel
            schema={schema}
            schemaJson={schemaJson}
            setSchemaJson={setSchemaJson}
            modifyMode={modifyMode}
            setModifyMode={setModifyMode}
            executionMode={executionMode}
            setExecutionMode={setExecutionMode}
            remoteMachines={remoteMachines}
            selectedMachineId={selectedMachineId}
            setSelectedMachineId={setSelectedMachineId}
            setRemoteMachines={setRemoteMachines}
            addRemoteMachine={addRemoteMachine}
            submitParams={submitParams}
            terminateExecution={terminateExecution}
          />
        ) : (
          <FilePanel files={files} previewFile={previewFile} deleteFile={deleteFile} downloadUrl={(file) => client.getDownloadUrl(sessionId, file.name)} />
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

function MessageBubble({ message, matterViz, responding }: { message: ChatMessage; matterViz: typeof defaultMatterViz; responding: boolean }) {
  const isUser = message.role === 'user';
  return (
    <article className={isUser ? 'message user' : 'message assistant'}>
      <div className="avatar">{isUser ? 'U' : 'AI'}</div>
      <div className="message-content">
        {isUser ? <p>{message.content}</p> : <MarkdownWithVisualize content={message.content} matterViz={matterViz} />}
        {responding && <span className="typing">...</span>}
        {!isUser && message.usage_metadata && (
          <small className="metadata">输入 token {message.usage_metadata.prompt_tokens || 0} · 输出 token {message.usage_metadata.candidates_tokens || 0}</small>
        )}
        {!isUser && message.charge_result && (
          <small className="metadata">光子 {message.charge_result.photon_amount || 0} · RMB {(message.charge_result.rmb_amount || 0).toFixed(2)}</small>
        )}
      </div>
    </article>
  );
}

function MarkdownWithVisualize({ content, matterViz }: { content: string; matterViz: typeof defaultMatterViz }) {
  const visualizeRegex = /:::visualize\n([\s\S]*?)\n:::/g;
  const parts: Array<{ type: 'markdown' | 'visualize'; content: string }> = [];
  let cursor = 0;
  for (const match of content.matchAll(visualizeRegex)) {
    if (match.index && match.index > cursor) parts.push({ type: 'markdown', content: content.slice(cursor, match.index) });
    parts.push({ type: 'visualize', content: match[1] });
    cursor = (match.index || 0) + match[0].length;
  }
  if (cursor < content.length) parts.push({ type: 'markdown', content: content.slice(cursor) });

  if (!parts.length) return <ReactMarkdown>{content}</ReactMarkdown>;
  return <>{parts.map((part, index) => part.type === 'markdown'
    ? <ReactMarkdown key={index}>{part.content}</ReactMarkdown>
    : <VisualizationCard key={index} json={part.content} matterViz={matterViz} />)}</>;
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

function ParameterPanel(props: {
  schema: ToolSchema | null;
  schemaJson: string;
  setSchemaJson: (value: string) => void;
  modifyMode: ModifyMode;
  setModifyMode: (value: ModifyMode) => void;
  executionMode: ExecutionMode;
  setExecutionMode: (value: ExecutionMode) => void;
  remoteMachines: RemoteMachine[];
  selectedMachineId: string;
  setSelectedMachineId: (value: string) => void;
  setRemoteMachines: (value: RemoteMachine[] | ((old: RemoteMachine[]) => RemoteMachine[])) => void;
  addRemoteMachine: (type: 'Bohrium' | 'Slurm') => void;
  submitParams: (formData?: FormData) => Promise<void>;
  terminateExecution: () => Promise<void>;
}) {
  const selectedMachine = props.remoteMachines.find((machine) => machine.id === props.selectedMachineId) || null;

  return (
    <div className="panel-content">
      <label>Execution mode</label>
      <select value={props.executionMode} onChange={(event) => props.setExecutionMode(event.target.value as ExecutionMode)}>
        <option value="Local">Local</option>
        <option value="Remote">Remote</option>
      </select>

      {props.executionMode === 'Remote' && (
        <section className="remote-machines">
          <label>Remote machine
            <select value={props.selectedMachineId} onChange={(event) => props.setSelectedMachineId(event.target.value)}>
              <option value="">Select a remote machine</option>
              {props.remoteMachines.map((machine) => <option key={machine.id} value={machine.id}>{machine.name} ({machine.type})</option>)}
            </select>
          </label>
          <div className="inline-row remote-actions">
            <button type="button" onClick={() => props.addRemoteMachine('Bohrium')}>Add Bohrium</button>
            <button type="button" onClick={() => props.addRemoteMachine('Slurm')}>Add Slurm</button>
          </div>
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

function FilePanel({ files, previewFile, deleteFile, downloadUrl }: { files: FileInfo[]; previewFile: (file: FileInfo) => void; deleteFile: (file: FileInfo) => void; downloadUrl: (file: FileInfo) => string }) {
  const groups = groupFiles(files);
  return (
    <div className="panel-content">
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
      {!files.length && <div className="empty-state">No files yet.</div>}
    </div>
  );
}

function mergeFinalContent(existing: string, finalContent: string) {
  if (!finalContent || existing.endsWith(finalContent) || existing === finalContent) return existing;
  if (!existing) return finalContent;
  if (finalContent.startsWith(existing)) return finalContent;
  return existing + finalContent;
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
      resource: {
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
      program_id: parseInt(config.project_id, 10)
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

function groupFiles(files: FileInfo[]) {
  return files.reduce((acc, file) => {
    const ext = extensionOf(file.name);
    const key = ['xyz', 'cif', 'poscar', 'vasp', 'xsf'].includes(ext) ? 'structure' : isImage(ext) || ext === 'pdf' ? 'plot' : ['json', 'csv', 'txt', 'log', 'out'].includes(ext) ? 'data' : 'other';
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

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}


