// 聊天消息类型
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

// 登录请求类型
export interface LoginRequest {
  session_id: string;
}

// 登录响应类型
export interface LoginResponse {
  message: string;
  session_id: string;
}

// 文件信息类型
export interface FileInfo {
  name: string;
  path: string;
  size: number;
  updated_at?: number;
}

// 工具参数Schema类型
export interface ToolSchema {
  name: string;
  description: string;
  input_schema: {
    properties: Record<string, PropertySchema>;
  };
  parameters?: Record<string, any>;
}

export interface PropertySchema {
  title: string;
  type: string;
  default?: any;
  agent_input?: any;
  user_input?: any;
  description?: string;
}

// WebSocket消息类型
export interface WSMessage {
  type: 'streaming_response' | 'final_response' | 'error' | 'tool_modify_required';
  content?: string;
  is_final?: boolean;
  message?: string;
  schema?: ToolSchema;
  tool_name?: string;
}

// 应用配置类型
export interface AppConfig {
  agent_info: {
    name: string;
    description: string;
    instruction: string;
  };
  mcp_server_url: string;
  target_tools: string[];
}

// 用户会话信息类型（32位用户会话ID）
export interface UserSession {
  user_id: string; // 32位用户会话ID
  agent_name: string;
  created_at: string;
}

// 聊天会话类型（用户会话下的具体聊天）
export interface ChatSession {
  chat_id: string; // 聊天会话唯一标识
  user_id: string; // 所属用户ID
  title: string;
  history: ChatMessage[];
  created_at: string;
  last_active: string;
  message_count: number;
}

// 当前聊天会话信息
export interface CurrentChatSession {
  chat_id: string;
  user_id: string;
  title: string;
  history: ChatMessage[];
}

// MCP工具执行模式
export type ExecutionMode = 'Local' | 'Bohr' | 'None';

// 参数修改模式
export type ModifyMode = 'individual' | 'json';

// React应用状态类型
export interface AppState {
  isAuthenticated: boolean;
  userId: string; // 32位用户会话ID
  currentChatSession: CurrentChatSession | null;
  chatSessions: ChatSession[]; // 当前用户的所有聊天会话
  files: FileInfo[]; // 用户文件列表 (全局)
  config: AppConfig | null;
  executionMode: ExecutionMode;
  modifyMode: ModifyMode;
  loading: boolean;
  error: string | null;
  responding: boolean; // Agent是否正在响应
  pendingToolResponse: string; // 待处理的工具响应内容
  language: 'zh' | 'en';
}