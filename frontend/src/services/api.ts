import axios from 'axios';
import type { LoginRequest, LoginResponse, FileInfo, ToolSchema, WSMessage, AppConfig, ChatMessage } from '../types';

// 强制使用相对路径，确保通过Vite代理
const API_BASE_URL = '/api';

console.log('API基础URL:', API_BASE_URL);
console.log('当前环境:', import.meta.env.MODE);

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器处理错误
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API响应错误:', error);

    if (error.response) {
      // 服务器响应了，但状态码不在2xx范围内
      const status = error.response.status;
      const data = error.response.data;

      if (data && data.detail) {
        throw new Error(data.detail);
      } else if (data && data.message) {
        throw new Error(data.message);
      } else {
        throw new Error(`服务器错误 (${status})`);
      }
    } else if (error.request) {
      // 请求已发出，但没有收到响应
      throw new Error('网络连接失败，请检查后端服务器是否正在运行');
    } else {
      // 在设置请求时触发了错误
      throw new Error(`请求配置错误: ${error.message}`);
    }
  }
);

// 请求拦截器处理网络错误
api.interceptors.request.use(
  (config) => {
    console.log('发起API请求:', config.method?.toUpperCase(), config.url, config.baseURL);
    return config;
  },
  (error) => {
    console.error('API请求配置错误:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  // 认证相关
  async login(data: LoginRequest): Promise<LoginResponse> {
    console.log('尝试登录，会话ID:', data.session_id);
    const response = await api.post('/login', data);
    console.log('登录响应:', response.data);
    return response.data;
  },

  // 获取应用配置
  async getConfig(): Promise<AppConfig> {
    console.log('获取应用配置...');
    const response = await api.get('/config');
    console.log('配置响应:', response.data);
    return response.data;
  },

  // 聊天相关
  async sendMessage(sessionId: string, message: string): Promise<{ response: string; is_final: boolean }> {
    const response = await api.post('/chat', {
      session_id: sessionId,
      message,
    });
    return response.data;
  },

  // 获取聊天历史
  async getChatHistory(sessionId: string): Promise<{ history: ChatMessage[] }> {
    const response = await api.get(`/sessions/${sessionId}/history`);
    return response.data;
  },

  // 清空聊天历史
  async clearChatHistory(sessionId: string): Promise<{ message: string }> {
    const response = await api.post(`/sessions/${sessionId}/clear`);
    return response.data;
  },

  // 获取用户的所有会话
  async getUserSessions(userId: string): Promise<{ sessions: any[] }> {
    const response = await api.get(`/user/${userId}/sessions`);
    return response.data;
  },

  // 保存用户的所有会话
  async saveUserSessions(userId: string, sessions: any[]): Promise<{ message: string }> {
    const response = await api.post(`/user/${userId}/sessions`, { sessions });
    return response.data;
  },

  // 获取当前工具参数schema
  async getCurrentSchema(sessionId: string): Promise<{ schema: ToolSchema }> {
    const response = await api.get(`/schema/${sessionId}`);
    return response.data;
  },

  // 提交修改后的参数
  async modifyParameters(sessionId: string, modifiedSchema: ToolSchema): Promise<{ message: string; modified_args: any }> {
    const response = await api.post('/modify-params', {
      session_id: sessionId,
      modified_schema: modifiedSchema,
    });
    return response.data;
  },

  // 文件管理相关
  async getFiles(sessionId: string): Promise<{ files: FileInfo[] }> {
    const response = await api.get(`/files/${sessionId}`);
    return response.data;
  },

  async uploadFiles(sessionId: string, files: File[]): Promise<{ uploaded_files: FileInfo[] }> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/upload/${sessionId}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Failed to upload files');
    return response.json();
  },

  async deleteFile(sessionId: string, filename: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/files/${sessionId}/${filename}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete file');
    return response.json();
  },

  // 下载文件
  getFileDownloadUrl(sessionId: string, filename: string): string {
    return `${API_BASE_URL}/download/${sessionId}/${filename}`;
  },
};

// WebSocket连接
export class WebSocketService {
  private ws: WebSocket | null = null;

  private messageHandlers: ((message: WSMessage) => void)[] = [];

  connect(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {

      // 使用相对路径或根据环境构造WebSocket URL
      let wsUrl: string;
      if (API_BASE_URL.startsWith('/')) {
        // 开发环境使用代理
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let host = window.location.host;
        // 如果host是0.0.0.0，替换为127.0.0.1，因为浏览器通常不支持连接到0.0.0.0的WebSocket
        if (host.startsWith('0.0.0.0')) {
          host = host.replace('0.0.0.0', '127.0.0.1');
        }
        wsUrl = `${protocol}//${host}/ws/chat/${sessionId}`;
      } else {
        // 生产环境使用完整URL
        wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/chat/${sessionId}`;
      }

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket连接已建立');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WSMessage = JSON.parse(event.data);
            this.messageHandlers.forEach(handler => handler(message));
          } catch (error) {
            console.error('解析WebSocket消息失败:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket错误:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket连接已关闭');
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageHandlers = [];
  }

  sendMessage(message: string, chatId?: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const payload = { message, chat_id: chatId };
      console.log('即将通过WebSocket发送消息:', JSON.stringify(payload));
      this.ws.send(JSON.stringify(payload));
    } else {
      throw new Error('WebSocket连接未建立');
    }
  }

  onMessage(handler: (message: WSMessage) => void) {
    this.messageHandlers.push(handler);
  }

  removeMessageHandler(handler: (message: WSMessage) => void) {
    const index = this.messageHandlers.indexOf(handler);
    if (index > -1) {
      this.messageHandlers.splice(index, 1);
    }
  }
}

export const wsService = new WebSocketService();