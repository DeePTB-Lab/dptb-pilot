export type Role = 'user' | 'assistant';

export interface TokenUsage {
  prompt_tokens: number;
  candidates_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
  event_count?: number;
  [key: string]: number | undefined;
}

export interface ToolLog {
  path: string;
  workspace_path?: string;
  content?: string;
  truncated?: boolean;
}

export interface ToolResult {
  tool: string;
  status?: string;
  input?: unknown;
  result?: unknown;
  logs?: ToolLog[];
  execution_mode?: string;
  remote_task_id?: string;
  bohrium_task_id?: string;
  remote_task_status?: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  role: Role;
  content: string;
  timestamp?: string;
  message_id?: string;
  prompt_id?: string;
  workflow_mode?: string;
  enqueued_at?: string;
  run_id?: string;
  stage_id?: string;
  sequence?: number;
  status?: string;
  tools?: string[];
  tool_results?: ToolResult[];
  usage_metadata?: {
    prompt_tokens?: number;
    candidates_tokens?: number;
    total_tokens?: number;
    cached_tokens?: number;
    reasoning_tokens?: number;
    event_count?: number;
    [key: string]: number | undefined;
  };
  usage_events?: TokenUsage[];
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
  remote_enabled?: boolean;
  capabilities?: string[];
  build_id?: string;
}

export interface FileInfo {
  name: string;
  path: string;
  size: number;
  updated_at?: number;
}

export interface ArtifactInfo {
  id: string;
  source_path: string;
  workspace_path?: string | null;
  file_name?: string | null;
  size?: number | null;
  mime_type: string;
  exists: boolean;
  can_stage?: boolean;
  preview_kind: 'image' | 'text' | 'binary' | 'unavailable';
  content_preview?: string | null;
}

export interface ArtifactTask {
  id: string;
  chat_id: string;
  task: string;
  completed_at: string;
  artifacts: ArtifactInfo[];
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

export interface WorkflowStep {
  id: string;
  title: string;
  purpose: string;
  tools: string[];
  inputs: string[];
  outputs: string[];
  depends_on: string[];
}

export interface WorkflowReview {
  verdict: 'pass' | 'needs_attention' | 'block';
  summary: string;
  scientific_logic: string[];
  execution_logic: string[];
  blocking_issues: string[];
  warnings: string[];
  recommendations: string[];
  evidence: string[];
  reviewed_at?: string;
}

export interface WorkflowPlan {
  title: string;
  summary: string;
  task: string;
  steps: WorkflowStep[];
  risks?: string[];
  questions?: string[];
  review?: WorkflowReview | null;
}

export type WorkflowDecision = 'approved' | 'rejected' | 'pilot_review';

export interface AgentStatus {
  status: 'idle' | 'waiting_for_reply' | 'in_progress';
  status_revision?: number;
  phase?: string;
  progress?: string;
  tool_name?: string;
  started_at?: string;
  updated_at?: string;
  elapsed_ms?: number;
  event_count?: number;
  usage?: TokenUsage;
  last_event?: string;
  chat_id?: string;
  method_name?: string;
  remote_task_id?: string;
  bohrium_task_id?: string;
  remote_task_status?: string;
  execution_mode?: string;
  remote_tasks?: Array<Record<string, unknown>>;
}

export interface QueuedPromptInfo {
  prompt_id: string;
  message: string;
  workflow_mode?: string;
  protocol_version?: number;
  enqueued_at?: string;
  position: number;
}

export interface RunSnapshot {
  run_id: string;
  prompt_id: string;
  message: string;
  protocol_version?: number;
  status: string;
  sequence: number;
  started_at?: string;
  finished_at?: string | null;
  stages?: Array<Record<string, unknown>>;
  tool_results?: Array<Record<string, unknown>>;
}

export interface ChatRuntimeSnapshot {
  active_run: RunSnapshot | null;
  queue: QueuedPromptInfo[];
  recent_runs?: RunSnapshot[];
}

export interface WSMessage {
  type: 'streaming_response' | 'final_response' | 'error' | 'tool_modify_required' | 'workflow_plan_required' | 'usage_update' | 'charge_result' | 'agent_status'
    | 'runtime_snapshot' | 'prompt_queued' | 'queued_prompt_removed' | 'run_started'
    | 'stage_started' | 'stage_completed' | 'run_completed' | 'run_failed' | 'run_cancelled';
  protocol_version?: number;
  content?: string;
  is_final?: boolean;
  message?: string;
  schema?: ToolSchema;
  tool_name?: string;
  workflow_plan?: WorkflowPlan;
  usage?: Record<string, unknown> | TokenUsage;
  usage_total?: TokenUsage;
  usage_events?: TokenUsage[];
  usage_metadata?: ChatMessage['usage_metadata'];
  charge_result?: ChatMessage['charge_result'];
  chat_id?: string;
  status?: 'idle' | 'waiting_for_reply' | 'in_progress' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted' | 'succeeded';
  progress?: string;
  phase?: string;
  started_at?: string;
  updated_at?: string;
  status_revision?: number;
  elapsed_ms?: number;
  event_count?: number;
  last_event?: string;
  method_name?: string;
  remote_task_id?: string;
  bohrium_task_id?: string;
  remote_task_status?: string;
  execution_mode?: string;
  remote_tasks?: Array<Record<string, unknown>>;
  prompt_id?: string;
  workflow_mode?: string;
  enqueued_at?: string;
  run_id?: string;
  stage_id?: string;
  message_id?: string;
  sequence?: number;
  position?: number;
  tools?: string[];
  tool_results?: ToolResult[];
  queue?: QueuedPromptInfo[];
  active_run?: RunSnapshot | null;
  recent_runs?: RunSnapshot[];
  timestamp?: string;
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
  token_usage?: TokenUsage;
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

export interface ExecutionPreferences {
  executionMode: ExecutionMode;
  remoteMachines: RemoteMachine[];
  selectedMachineId: string;
  workflowMode?: WorkflowMode;
  parallelMode?: 'sequential' | 'auto' | 'parallel';
  maxParallelism?: number;
}

export type WorkflowMode = 'always' | 'auto' | 'disabled';

export interface ModifyParamsRequest {
  session_id: string;
  chat_id?: string;
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
  parallel_mode?: 'sequential' | 'auto' | 'parallel';
  max_parallelism?: number;
}

export interface TerminateExecutionRequest {
  session_id: string;
  chat_id?: string;
}

export interface TerminateExecutionResponse {
  message: string;
  status: string;
}

export interface ExtensionBootstrap {
  serverUrl: string;
  sessionId: string;
  extensionUri: string;
  executionPreferences?: ExecutionPreferences;
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
  | { type: 'filesChanged'; payload: { uploadedFiles: string[] } }
  | { type: 'uploadFailed'; payload: { message: string } }
  | { type: 'secret'; payload: { requestId: string; value?: string } }
  | { type: 'openNewSession' };

export type WebviewToHostMessage =
  | { type: 'ready' }
  | { type: 'setServerUrl'; payload: { serverUrl: string } }
  | { type: 'setSessionId'; payload: { sessionId: string } }
  | { type: 'setExecutionPreferences'; payload: ExecutionPreferences }
  | { type: 'pickFiles'; payload: { serverUrl: string; sessionId: string } }
  | { type: 'uploadCurrentFile'; payload: { serverUrl: string; sessionId: string } }
  | { type: 'downloadArtifact'; payload: { sessionId: string; workspacePath: string; suggestedName: string } }
  | { type: 'copyText'; payload: { text: string } }
  | { type: 'saveSecret'; payload: { key: string; value: string } }
  | { type: 'getSecret'; payload: { key: string; requestId: string } }
  | { type: 'openInMatterViz'; payload: { visualization: VisualizationPayload } };
