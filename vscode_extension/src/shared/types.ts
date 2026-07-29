export type Role = 'user' | 'assistant';

export interface ChatMessage {
  role: Role;
  content: string;
  timestamp?: string;
  usage_metadata?: {
    prompt_tokens?: number;
    candidates_tokens?: number;
    total_tokens?: number;
  };
  charge_result?: {
    success: boolean;
    code: string | number;
    message: string;
    biz_no?: string | null;
    photon_amount?: number;
    rmb_amount?: number;
  };
}

export type VisualizationKind = 'structure' | 'bz' | 'band' | 'dos' | 'curve' | 'image' | 'table' | 'text';

export interface VisualizationSource {
  sessionId?: string;
  file?: string;
  tool?: string;
}

export interface VisualizationPayload {
  kind: VisualizationKind;
  title?: string;
  source?: VisualizationSource;
  data: unknown;
  meta?: Record<string, unknown>;
}

export interface LoginRequest {
  session_id: string;
}

export interface LoginResponse {
  message: string;
  session_id: string;
}

export interface FileInfo {
  name: string;
  path: string;
  size: number;
  updated_at?: number;
}

export interface ToolSchema {
  name: string;
  description?: string;
  input_schema: {
    properties: Record<string, PropertySchema>;
  };
  parameters?: Record<string, unknown>;
}

export interface PropertySchema {
  title?: string;
  type: string;
  default?: unknown;
  agent_input?: unknown;
  user_input?: unknown;
  description?: string;
}

export interface WSMessage {
  type: 'streaming_response' | 'final_response' | 'error' | 'tool_modify_required' | 'usage_update' | 'charge_result';
  content?: string;
  is_final?: boolean;
  message?: string;
  schema?: ToolSchema;
  tool_name?: string;
  usage?: Record<string, unknown>;
  usage_metadata?: ChatMessage['usage_metadata'];
  charge_result?: ChatMessage['charge_result'];
}

export interface AppConfig {
  agent_info: {
    name: string;
    description: string;
    instruction: string;
  };
  mcp_server_url: string;
  target_tools: string[];
}

export interface ChatSession {
  chat_id: string;
  user_id: string;
  title: string;
  history: ChatMessage[];
  created_at: string;
  last_active: string;
  message_count: number;
}

export type ExecutionMode = 'Local' | 'Remote' | 'None';
export type RemoteMachineType = 'Bohrium' | 'Slurm';
export type ModifyMode = 'individual' | 'json';

export interface BohriumConfig {
  username: string;
  password?: string;
  project_id: string;
  image_name?: string;
  scass_type?: string;
}

export interface SlurmConfig {
  remote_root: string;
  hostname: string;
  username: string;
  key_filename: string;
  number_node: string;
  gpu_per_node?: string;
  cpu_per_node?: string;
  queue_name: string;
  custom_flags?: string;
}

export interface RemoteMachine {
  id: string;
  name: string;
  type: RemoteMachineType;
  config: BohriumConfig | SlurmConfig;
}

export interface ModifyParamsRequest {
  session_id: string;
  modified_schema: ToolSchema;
  execution_mode: ExecutionMode;
  selected_machine_id?: string;
  remote_machine?: RemoteMachine | null;
}

export interface ExecutionConfigRequest {
  session_id: string;
  execution_mode: ExecutionMode;
  selected_machine_id?: string;
  remote_machine?: RemoteMachine | null;
}

export interface TerminateExecutionRequest {
  session_id: string;
}

export interface TerminateExecutionResponse {
  message: string;
  status: string;
}

export interface ExtensionBootstrap {
  serverUrl: string;
  sessionId: string;
  extensionUri: string;
  matterViz?: {
    commandId?: string;
    uriTemplate?: string;
    autoOpen?: boolean;
  };
}

export type HostToWebviewMessage =
  | { type: 'bootstrap'; payload: ExtensionBootstrap }
  | { type: 'serverUrlChanged'; payload: { serverUrl: string } }
  | { type: 'sessionChanged'; payload: { sessionId: string } }
  | { type: 'uploadFile'; payload: { name: string; bytes: number[] } }
  | { type: 'secret'; payload: { requestId: string; value?: string } }
  | { type: 'openNewSession' };

export type WebviewToHostMessage =
  | { type: 'ready' }
  | { type: 'setServerUrl'; payload: { serverUrl: string } }
  | { type: 'setSessionId'; payload: { sessionId: string } }
  | { type: 'pickFiles' }
  | { type: 'uploadCurrentFile' }
  | { type: 'saveSecret'; payload: { key: string; value: string } }
  | { type: 'getSecret'; payload: { key: string; requestId: string } }
  | { type: 'openInMatterViz'; payload: { visualization: VisualizationPayload } };
