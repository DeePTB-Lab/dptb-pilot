import type {
  AppConfig,
  AgentStatus,
  ArtifactTask,
  ChatSession,
  ExecutionConfigRequest,
  FileInfo,
  LoginRequest,
  LoginResponse,
  ModifyParamsRequest,
  TerminateExecutionRequest,
  TerminateExecutionResponse,
  ToolSchema,
  WorkflowDecision,
  WorkflowPlan,
  WSMessage
} from '../shared/types';
import { apiUrl, normalizeBaseUrl, websocketUrl } from './url';

export interface UploadPayload {
  name: string;
  content: Blob | Uint8Array | ArrayBuffer;
  type?: string;
}

export class DptbPilotApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'DptbPilotApiError';
    this.status = status;
  }
}

async function parseError(response: Response): Promise<DptbPilotApiError> {
  try {
    const data = await response.json();
    return new DptbPilotApiError(
      response.status,
      data.detail || data.message || `HTTP ${response.status}`
    );
  } catch {
    return new DptbPilotApiError(
      response.status,
      `HTTP ${response.status}: ${response.statusText}`
    );
  }
}

export class DptbPilotClient {
  readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const requestUrl = apiUrl(this.baseUrl, path);
    let response: Response;
    try {
      response = await fetch(requestUrl, {
        ...init,
        headers: {
          ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
          ...init?.headers
        }
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`Cannot reach the DPTB Pilot backend at ${this.baseUrl}: ${detail}`);
    }

    if (!response.ok) {
      throw await parseError(response);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    try {
      return await response.json() as T;
    } catch {
      throw new Error(`The server at ${this.baseUrl} did not return a DPTB Pilot API response. Check that this is the backend port, not the frontend URL.`);
    }
  }

  login(data: LoginRequest): Promise<LoginResponse> {
    return this.request('/api/login', { method: 'POST', body: JSON.stringify(data) });
  }

  setExecutionConfig(data: ExecutionConfigRequest): Promise<{
    message: string;
    execution_mode: string;
    remote_enabled: boolean;
  }> {
    return this.request('/api/execution-config', {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  getHealth(): Promise<{ status: string; message?: string }> {
    return this.request('/api/health');
  }

  getConfig(): Promise<AppConfig> {
    return this.request('/api/config');
  }

  getAgentStatus(sessionId: string, chatId?: string): Promise<AgentStatus> {
    const query = chatId ? `?chat_id=${encodeURIComponent(chatId)}` : '';
    return this.request(`/api/agent-status/${encodeURIComponent(sessionId)}${query}`);
  }

  getUserSessions(userId: string): Promise<{ sessions: ChatSession[] }> {
    return this.request(`/api/user/${encodeURIComponent(userId)}/sessions`);
  }

  saveUserSessions(userId: string, sessions: ChatSession[]): Promise<{ message: string }> {
    return this.request(`/api/user/${encodeURIComponent(userId)}/sessions`, {
      method: 'POST',
      body: JSON.stringify({ sessions })
    });
  }

  getFiles(sessionId: string): Promise<{ files: FileInfo[] }> {
    return this.request(`/api/files/${encodeURIComponent(sessionId)}`);
  }

  getArtifacts(sessionId: string): Promise<{ tasks: ArtifactTask[] }> {
    return this.request(`/api/artifacts/${encodeURIComponent(sessionId)}`);
  }

  stageArtifact(sessionId: string, artifactId: string): Promise<{ artifact: ArtifactTask['artifacts'][number] }> {
    return this.request(
      `/api/artifacts/${encodeURIComponent(sessionId)}/${encodeURIComponent(artifactId)}/stage`,
      { method: 'POST' }
    );
  }

  async uploadFiles(sessionId: string, files: UploadPayload[]): Promise<{ uploaded_files: FileInfo[] }> {
    const formData = new FormData();
    files.forEach((file) => {
      const content = file.content;
      const blob = content instanceof Blob
        ? content
        : new Blob([
          content instanceof ArrayBuffer
            ? content
            : content.buffer.slice(content.byteOffset, content.byteOffset + content.byteLength) as ArrayBuffer
        ], { type: file.type || 'application/octet-stream' });
      formData.append('files', blob, file.name);
    });

    return this.request(`/api/upload/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      body: formData
    });
  }

  getDownloadUrl(sessionId: string, filename: string): string {
    return apiUrl(this.baseUrl, `/api/download/${encodeURIComponent(sessionId)}/${encodeWorkspacePath(filename)}`);
  }

  async downloadFile(sessionId: string, filename: string): Promise<Blob> {
    const response = await fetch(this.getDownloadUrl(sessionId, filename));
    if (!response.ok) {
      throw await parseError(response);
    }
    return response.blob();
  }

  async previewTextFile(sessionId: string, filename: string): Promise<string> {
    const response = await fetch(this.getDownloadUrl(sessionId, filename));
    if (!response.ok) {
      throw await parseError(response);
    }
    return response.text();
  }

  deleteFile(sessionId: string, filename: string): Promise<unknown> {
    return this.request(`/api/files/${encodeURIComponent(sessionId)}/${encodeURI(filename)}`, { method: 'DELETE' });
  }

  getCurrentSchema(sessionId: string): Promise<{ schema: ToolSchema }> {
    return this.request(`/api/schema/${encodeURIComponent(sessionId)}`);
  }

  getCurrentWorkflowPlan(sessionId: string, chatId?: string): Promise<{ workflow_plan: WorkflowPlan | Record<string, never> }> {
    const query = chatId ? `?chat_id=${encodeURIComponent(chatId)}` : '';
    return this.request(`/api/workflow-plan/${encodeURIComponent(sessionId)}${query}`);
  }

  decideWorkflowPlan(sessionId: string, decision: WorkflowDecision, chatId?: string): Promise<{
    message: string;
    decision: WorkflowDecision;
    workflow_plan?: WorkflowPlan;
  }> {
    return this.request('/api/workflow-plan/decision', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, chat_id: chatId, decision })
    });
  }

  modifyParameters(request: ModifyParamsRequest): Promise<{ message: string; modified_args: unknown }> {
    return this.request('/api/modify-params', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  terminateExecution(request: TerminateExecutionRequest): Promise<TerminateExecutionResponse> {
    return this.request('/api/terminate-execution', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  createWebSocket(sessionId: string, chatId?: string): WebSocket {
    return new WebSocket(websocketUrl(this.baseUrl, sessionId, chatId));
  }

  static parseWSMessage(data: string): WSMessage {
    return JSON.parse(data) as WSMessage;
  }
}

export function encodeWorkspacePath(filename: string): string {
  return normalizeWorkspacePath(filename)
    .replace(/\\/g, '/')
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

/** Normalize artifact paths from older backends before building a download URL. */
export function normalizeWorkspacePath(filename: string): string {
  return String(filename || '').trim().replace(/[\\/]+$/, '');
}
