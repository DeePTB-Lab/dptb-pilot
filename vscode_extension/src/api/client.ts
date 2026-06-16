import type {
  AppConfig,
  ChatSession,
  FileInfo,
  LoginRequest,
  LoginResponse,
  ModifyParamsRequest,
  TerminateExecutionRequest,
  TerminateExecutionResponse,
  ToolSchema,
  WSMessage
} from '../shared/types';
import { apiUrl, normalizeBaseUrl, websocketUrl } from './url';

export interface UploadPayload {
  name: string;
  bytes: Uint8Array | ArrayBuffer;
  type?: string;
}

async function parseError(response: Response): Promise<Error> {
  try {
    const data = await response.json();
    return new Error(data.detail || data.message || `HTTP ${response.status}`);
  } catch {
    return new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
}

export class DptbPilotClient {
  readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(apiUrl(this.baseUrl, path), {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers
      }
    });

    if (!response.ok) {
      throw await parseError(response);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json() as Promise<T>;
  }

  login(data: LoginRequest): Promise<LoginResponse> {
    return this.request('/api/login', { method: 'POST', body: JSON.stringify(data) });
  }

  getConfig(): Promise<AppConfig> {
    return this.request('/api/config');
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

  async uploadFiles(sessionId: string, files: UploadPayload[]): Promise<{ uploaded_files: FileInfo[] }> {
    const formData = new FormData();
    files.forEach((file) => {
      const bytes = file.bytes instanceof ArrayBuffer
        ? file.bytes
        : file.bytes.buffer.slice(file.bytes.byteOffset, file.bytes.byteOffset + file.bytes.byteLength) as ArrayBuffer;
      const blob = new Blob([bytes], { type: file.type || 'application/octet-stream' });
      formData.append('files', blob, file.name);
    });

    return this.request(`/api/upload/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      body: formData
    });
  }

  getDownloadUrl(sessionId: string, filename: string): string {
    return apiUrl(this.baseUrl, `/api/download/${encodeURIComponent(sessionId)}/${encodeURI(filename)}`);
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

  createWebSocket(sessionId: string): WebSocket {
    return new WebSocket(websocketUrl(this.baseUrl, sessionId));
  }

  static parseWSMessage(data: string): WSMessage {
    return JSON.parse(data) as WSMessage;
  }
}
