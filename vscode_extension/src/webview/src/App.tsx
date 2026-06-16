import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { DptbPilotClient } from '../../api/client';
import { generateSessionId, isValidSessionId, normalizeBaseUrl } from '../../api/url';
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
  WebviewToHostMessage,
  WSMessage
} from '../../shared/types';

declare global {
  interface Window {
    acquireVsCodeApi?: () => { postMessage: (message: WebviewToHostMessage) => void; getState: () => unknown; setState: (state: unknown) => void };
  }
}

const vscode = window.acquireVsCodeApi?.();

const shortcutPrompts = [
  { title: '分析结构', description: '分析上传的晶体结构文件', prompt: '请帮我分析一下上传的晶体结构文件' },
  { title: '计算能带', description: '计算并绘制能带图', prompt: '请帮我计算并绘制能带图' },
  { title: '生成配置', description: '生成 DeepTB 训练配置文件', prompt: '请帮我生成DeepTB训练配置文件' },
  { title: '搜索材料', description: '搜索相关材料数据', prompt: '请帮我搜索相关的材料数据' }
];

export function App() {
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');
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

  const wsRef = useRef<WebSocket | null>(null);
  const currentChatIdRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const client = useMemo(() => new DptbPilotClient(serverUrl), [serverUrl]);
  const currentSession = sessions.find((item) => item.chat_id === currentChatId) || null;

  useEffect(() => {
    vscode?.postMessage({ type: 'ready' });

    const listener = (event: MessageEvent<HostToWebviewMessage>) => {
      const message = event.data;
      if (message.type === 'bootstrap') {
        setServerUrl(message.payload.serverUrl || 'http://localhost:8000');
        if (message.payload.sessionId) setSessionId(message.payload.sessionId);
      } else if (message.type === 'serverUrlChanged') {
        setServerUrl(message.payload.serverUrl);
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
    const cleanServerUrl = normalizeBaseUrl(serverUrl);
    const cleanSessionId = targetSessionId.trim();
    if (!isValidSessionId(cleanSessionId)) {
      setError('Session ID 必须是 32 位字母或数字。');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      setServerUrl(cleanServerUrl);
      setSessionId(cleanSessionId);
      vscode?.postMessage({ type: 'setServerUrl', payload: { serverUrl: cleanServerUrl } });
      vscode?.postMessage({ type: 'setSessionId', payload: { sessionId: cleanSessionId } });
      await new DptbPilotClient(cleanServerUrl).login({ session_id: cleanSessionId });
      setConnected(true);
      await loadSessions(cleanSessionId, new DptbPilotClient(cleanServerUrl));
      await loadFiles(cleanSessionId, new DptbPilotClient(cleanServerUrl));
      connectWebSocket(cleanSessionId, new DptbPilotClient(cleanServerUrl));
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
    ws.onerror = () => setError('WebSocket 连接失败，请确认后端服务可访问。');
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
      setError('请选择远程机器后再提交参数。');
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
          <input value={serverUrl} onChange={(event) => setServerUrl(event.target.value)} placeholder="http://localhost:8000" />
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
              <span className="delete" onClick={(event) => { event.stopPropagation(); void deleteSession(session.chat_id); }}>×</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-main">
        <div className="message-list">
          {!currentSession?.history.length && (
            <div className="welcome">
              <h2>选择一个操作开始</h2>
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
          {currentSession?.history.map((message, index) => <MessageBubble key={index} message={message} responding={responding && index === currentSession.history.length - 1} />)}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <button title="选择文件上传" onClick={() => vscode?.postMessage({ type: 'pickFiles' })}>📎</button>
          <button title="上传当前编辑器文件" onClick={() => vscode?.postMessage({ type: 'uploadCurrentFile' })}>⇧</button>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              void sendMessage();
            }
          }} placeholder="输入消息，Enter 发送，Shift+Enter 换行" />
          <button className="primary" disabled={!connected || !input.trim() || responding} onClick={() => void sendMessage()}>{responding ? '响应中' : '发送'}</button>
        </div>
      </main>

      <aside className="right-panel">
        <div className="tabs">
          <button className={activePanel === 'params' ? 'selected' : ''} onClick={() => setActivePanel('params')}>参数</button>
          <button className={activePanel === 'files' ? 'selected' : ''} onClick={() => setActivePanel('files')}>文件</button>
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
              <button onClick={() => setPreview(null)}>×</button>
            </header>
            {preview.imageUrl ? <img src={preview.imageUrl} alt={preview.file.name} /> : <pre>{preview.content}</pre>}
            <a href={client.getDownloadUrl(sessionId, preview.file.name)}>下载</a>
          </div>
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message, responding }: { message: ChatMessage; responding: boolean }) {
  const isUser = message.role === 'user';
  return (
    <article className={isUser ? 'message user' : 'message assistant'}>
      <div className="avatar">{isUser ? '你' : 'AI'}</div>
      <div className="message-content">
        {isUser ? <p>{message.content}</p> : <MarkdownWithVisualize content={message.content} />}
        {responding && <span className="typing">● ● ●</span>}
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

function MarkdownWithVisualize({ content }: { content: string }) {
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
  return <>{parts.map((part, index) => part.type === 'markdown' ? <ReactMarkdown key={index}>{part.content}</ReactMarkdown> : <VisualizationFallback key={index} json={part.content} />)}</>;
}

function VisualizationFallback({ json }: { json: string }) {
  try {
    const parsed = JSON.parse(json);
    return <div className="visualize-box"><strong>结构可视化数据 ({parsed.format || 'unknown'})</strong><pre>{JSON.stringify(parsed.data, null, 2)}</pre></div>;
  } catch {
    return <div className="visualize-box"><strong>结构可视化数据</strong><pre>{json}</pre></div>;
  }
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
      <label>运行模式</label>
      <select value={props.executionMode} onChange={(event) => props.setExecutionMode(event.target.value as ExecutionMode)}>
        <option value="Local">Local</option>
        <option value="Remote">Remote</option>
      </select>

      {props.executionMode === 'Remote' && (
        <section className="remote-machines">
          <label>远程机器
            <select value={props.selectedMachineId} onChange={(event) => props.setSelectedMachineId(event.target.value)}>
              <option value="">选择远程机器</option>
              {props.remoteMachines.map((machine) => <option key={machine.id} value={machine.id}>{machine.name} ({machine.type})</option>)}
            </select>
          </label>
          <div className="inline-row remote-actions">
            <button type="button" onClick={() => props.addRemoteMachine('Bohrium')}>添加 Bohrium</button>
            <button type="button" onClick={() => props.addRemoteMachine('Slurm')}>添加 Slurm</button>
          </div>
          {selectedMachine && <small className="selected-machine">当前选择：{selectedMachine.name} ({selectedMachine.type})</small>}
          {props.remoteMachines.map((machine) => (
            <details key={machine.id} open={machine.id === props.selectedMachineId}>
              <summary>{machine.name} ({machine.type})</summary>
              <textarea value={JSON.stringify(machine.config, null, 2)} onChange={(event) => {
                try {
                  const config = JSON.parse(event.target.value);
                  props.setRemoteMachines((old) => old.map((item) => item.id === machine.id ? { ...item, config } : item));
                } catch {}
              }} />
            </details>
          ))}
        </section>
      )}

      <div className="inline-row">
        <button className={props.modifyMode === 'individual' ? 'selected' : ''} onClick={() => props.setModifyMode('individual')}>表单</button>
        <button className={props.modifyMode === 'json' ? 'selected' : ''} onClick={() => props.setModifyMode('json')}>JSON</button>
      </div>

      {!props.schema ? <div className="empty-state">等待 Agent 请求参数确认。</div> : (
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
            <button className="primary" type="submit">提交参数</button>
            <button type="button" onClick={() => void props.terminateExecution()}>终止执行</button>
          </div>
        </form>
      )}
    </div>
  );
}

function FilePanel({ files, previewFile, deleteFile, downloadUrl }: { files: FileInfo[]; previewFile: (file: FileInfo) => void; deleteFile: (file: FileInfo) => void; downloadUrl: (file: FileInfo) => string }) {
  const groups = groupFiles(files);
  return <div className="panel-content">{Object.entries(groups).map(([group, groupFiles]) => groupFiles.length ? <section key={group}><h3>{group} ({groupFiles.length})</h3>{groupFiles.map((file) => <div className="file-row" key={file.name}><button onClick={() => void previewFile(file)}>{file.name}<small>{(file.size / 1024).toFixed(1)} KB</small></button><a href={downloadUrl(file)}>下载</a><button onClick={() => void deleteFile(file)}>删除</button></div>)}</section> : null)}{!files.length && <div className="empty-state">暂无文件。</div>}</div>;
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
