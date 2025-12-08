import React, { createContext, useContext, useReducer, ReactNode, useEffect } from 'react';
import { AppState, CurrentChatSession, ExecutionMode } from '../types';
import {
  ChatMessage,
  ChatSession,
  FileInfo,
  ModifyMode,
  ToolSchema
} from '../types';
import { apiService, wsService } from '../services/api';
import { message } from 'antd';

// Action类型定义
type AppAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'LOGIN_SUCCESS'; payload: string } // userId
  | { type: 'LOGOUT' }
  | { type: 'SET_CONFIG'; payload: any }
  | { type: 'SET_CURRENT_CHAT_SESSION'; payload: CurrentChatSession | null }
  | { type: 'SET_CHAT_SESSIONS'; payload: ChatSession[] }
  | { type: 'CREATE_NEW_CHAT_SESSION'; payload: ChatSession }
  | { type: 'ADD_CHAT_MESSAGE'; payload: ChatMessage }
  | { type: 'UPDATE_FILES'; payload: FileInfo[] }
  | { type: 'SET_EXECUTION_MODE'; payload: ExecutionMode }
  | { type: 'SET_MODIFY_MODE'; payload: ModifyMode }
  | { type: 'UPDATE_STREAMING_RESPONSE'; payload: string }
  | { type: 'SET_RESPONDING'; payload: boolean }
  | { type: 'SET_PENDING_TOOL_RESPONSE'; payload: string }
  | { type: 'SET_LANGUAGE'; payload: 'zh' | 'en' };

// 初始状态
const initialState: AppState = {
  isAuthenticated: false,
  userId: '',
  currentChatSession: null,
  chatSessions: [],
  files: [],
  config: null,
  executionMode: 'Local',
  modifyMode: 'individual',
  loading: false,
  error: null,
  responding: false,
  pendingToolResponse: '',
  language: 'zh',
};

// Reducer函数
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload };

    case 'LOGIN_SUCCESS':
      return {
        ...state,
        isAuthenticated: true,
        userId: action.payload,
        error: null,
      };

    case 'LOGOUT':
      return {
        ...initialState,
        config: state.config, // 保留配置信息
      };

    case 'SET_CONFIG':
      return { ...state, config: action.payload };

    case 'SET_CURRENT_CHAT_SESSION':
      return { ...state, currentChatSession: action.payload };

    case 'SET_CHAT_SESSIONS':
      return { ...state, chatSessions: action.payload };

    case 'CREATE_NEW_CHAT_SESSION':
      return {
        ...state,
        chatSessions: [...state.chatSessions, action.payload],
        currentChatSession: {
          chat_id: action.payload.chat_id,
          user_id: action.payload.user_id,
          title: action.payload.title,
          history: action.payload.history,
        }
      };

    case 'ADD_CHAT_MESSAGE':
      const currentChatSession = state.currentChatSession;
      if (!currentChatSession) return state;

      const updatedHistory = [...currentChatSession.history, action.payload];
      return {
        ...state,
        currentChatSession: {
          ...currentChatSession,
          history: updatedHistory,
        },
        // 同时更新chatSessions中的对应会话
        chatSessions: state.chatSessions.map(session =>
          session.chat_id === currentChatSession.chat_id
            ? { ...session, history: updatedHistory, last_active: new Date().toISOString(), message_count: updatedHistory.length }
            : session
        ),
      };

    case 'UPDATE_FILES':
      return {
        ...state,
        files: action.payload,
      };

    case 'SET_EXECUTION_MODE':
      return { ...state, executionMode: action.payload };

    case 'SET_MODIFY_MODE':
      return { ...state, modifyMode: action.payload };

    case 'UPDATE_STREAMING_RESPONSE':
      const currentHistory = state.currentChatSession?.history || [];
      const lastMessage = currentHistory[currentHistory.length - 1];

      if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.timestamp) {
        // 更新最后一条助手消息
        const updatedHistory = [...currentHistory];
        updatedHistory[updatedHistory.length - 1] = {
          ...lastMessage,
          content: action.payload,
        };

        return {
          ...state,
          currentChatSession: state.currentChatSession ? {
            ...state.currentChatSession,
            history: updatedHistory,
          } : null,
        };
      } else {
        // 添加新的助手消息
        return {
          ...state,
          currentChatSession: state.currentChatSession ? {
            ...state.currentChatSession,
            history: [...state.currentChatSession.history, {
              role: 'assistant' as const,
              content: action.payload,
            }],
          } : null,
        };
      }

    case 'SET_RESPONDING':
      return { ...state, responding: action.payload };

    case 'SET_PENDING_TOOL_RESPONSE':
      return { ...state, pendingToolResponse: action.payload };

    case 'SET_LANGUAGE':
      return { ...state, language: action.payload };

    default:
      return state;
  }
}

interface AppContextActions {
  login: (userId: string) => Promise<void>;
  logout: () => void;
  sendMessage: (message: string) => Promise<void>;
  loadChatHistory: (userId?: string) => Promise<ChatSession[] | undefined>;
  createNewChatSession: (userId?: string) => Promise<ChatSession | undefined>;
  deleteChatSession: (chatId: string) => Promise<ChatSession[] | undefined>;
  updateSessionTitle: (chatId: string, title: string) => Promise<void>;
  clearCurrentChatHistory: () => Promise<void>;
  loadFiles: () => Promise<void>;
  uploadFiles: (files: File[]) => Promise<void>;
  deleteFile: (filename: string) => Promise<void>;
  getCurrentSchema: () => Promise<ToolSchema | null>;
  modifyParameters: (schema: ToolSchema) => Promise<void>;
  toggleLanguage: () => void;
  switchToChatSession: (chatId: string) => Promise<void>;
}

// Context创建
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  actions: AppContextActions;
} | null>(null);

// Provider组件
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // 初始化应用
  useEffect(() => {
    const initApp = async () => {
      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        const config = await apiService.getConfig();
        dispatch({ type: 'SET_CONFIG', payload: config });
        dispatch({ type: 'SET_ERROR', payload: null }); // 清除之前的错误

        // 尝试自动登录
        const savedUserId = localStorage.getItem('last_session_id');
        if (savedUserId) {
          try {
            console.log('尝试自动登录:', savedUserId);
            await apiService.login({ session_id: savedUserId });
            dispatch({ type: 'LOGIN_SUCCESS', payload: savedUserId });
            
            // 加载用户数据
            // 注意：这里我们不能直接调用actions，因为actions还没定义
            // 我们直接调用apiService并dispatch
            
            // 1. 加载会话
            try {
              const { sessions } = await apiService.getUserSessions(savedUserId);
              if (sessions && sessions.length > 0) {
                dispatch({ type: 'SET_CHAT_SESSIONS', payload: sessions });
                
                // 恢复上次活跃的会话
                const lastActiveChatId = localStorage.getItem(`last_active_chat_${savedUserId}`);
                let targetSession = null;
                
                if (lastActiveChatId) {
                  targetSession = sessions.find((s: any) => s.chat_id === lastActiveChatId);
                }
                
                if (!targetSession) {
                  targetSession = sessions[0];
                }
                
                if (targetSession) {
                  dispatch({ 
                    type: 'SET_CURRENT_CHAT_SESSION', 
                    payload: {
                      chat_id: targetSession.chat_id,
                      user_id: targetSession.user_id,
                      title: targetSession.title,
                      history: targetSession.history,
                    } 
                  });
                }
              }
            } catch (err) {
              console.warn('自动登录加载会话失败:', err);
            }

            // 2. 加载文件
            try {
              const { files } = await apiService.getFiles(savedUserId);
              dispatch({ type: 'UPDATE_FILES', payload: files });
            } catch (err) {
              console.warn('自动登录加载文件失败:', err);
            }

          } catch (loginError) {
            console.error('自动登录失败:', loginError);
            // 登录失败不清除localStorage，让用户在登录页看到ID并重试
          }
        }

      } catch (error) {
        console.error('初始化应用失败:', error);
        dispatch({ type: 'SET_ERROR', payload: null });
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    };

    initApp();
  }, []);

  // WebSocket消息处理
  useEffect(() => {
    if (state.isAuthenticated && state.userId) {
      const handleWSMessage = (message: any) => {
        switch (message.type) {
          case 'streaming_response':
            dispatch({ type: 'UPDATE_STREAMING_RESPONSE', payload: message.content || '' });
            break;
          case 'final_response':
            dispatch({ type: 'UPDATE_STREAMING_RESPONSE', payload: message.content || '' });
            // 标记消息完成
            if (state.currentChatSession?.history.length) {
              const lastMessage = state.currentChatSession.history[state.currentChatSession.history.length - 1];
              if (lastMessage.role === 'assistant') {
                const updatedHistory = [...state.currentChatSession.history];
                updatedHistory[updatedHistory.length - 1] = {
                  ...lastMessage,
                  timestamp: new Date().toISOString(),
                };
                dispatch({
                  type: 'SET_CURRENT_CHAT_SESSION',
                  payload: {
                    ...state.currentChatSession,
                    history: updatedHistory,
                  },
                });
              }
            }
            dispatch({ type: 'SET_RESPONDING', payload: false });
            
            // 任务完成后刷新文件列表，因为Agent可能生成了新文件
            actions.loadFiles();
            break;
          case 'error':
            dispatch({ type: 'SET_ERROR', payload: message.message || 'WebSocket错误' });
            break;
          case 'tool_modify_required':
            // 存储当前的响应内容作为待处理的工具响应
            const currentHistory = state.currentChatSession?.history || [];
            const lastMessage = currentHistory[currentHistory.length - 1];
            if (lastMessage && lastMessage.role === 'assistant') {
              dispatch({ type: 'SET_PENDING_TOOL_RESPONSE', payload: lastMessage.content });
            }
            break;
        }
      };

      wsService.connect(state.userId)
        .then(() => {
          wsService.onMessage(handleWSMessage);
        })
        .catch(error => {
          dispatch({ type: 'SET_ERROR', payload: `WebSocket连接失败: ${error.message}` });
        });

      return () => {
        wsService.disconnect();
      };
    }
  }, [state.isAuthenticated, state.userId]);

  // 自动保存聊天会话到localStorage
  // 注意：不再自动保存到服务器，因为服务器端的历史记录是最新的
  // 前端只负责保存到本地作为缓存，以及在创建新会话时通知服务器
  useEffect(() => {
    if (state.userId && state.chatSessions.length > 0) {
      // 保存到localStorage作为备份
      try {
        localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(state.chatSessions));
      } catch (error) {
        console.error('保存聊天会话到本地失败:', error);
      }
    }
  }, [state.chatSessions, state.userId]);

  // 保存当前活跃的会话ID
  useEffect(() => {
    if (state.userId && state.currentChatSession) {
      localStorage.setItem(`last_active_chat_${state.userId}`, state.currentChatSession.chat_id);
    }
  }, [state.currentChatSession, state.userId]);

  // Actions
  const actions = {
    login: async (userId: string) => {
      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        dispatch({ type: 'SET_ERROR', payload: null }); // 清除之前的错误

        await apiService.login({ session_id: userId });
        dispatch({ type: 'LOGIN_SUCCESS', payload: userId });

        // 加载现有的聊天会话
        const sessions = await actions.loadChatHistory(userId);

        // 如果没有现有聊天会话，创建一个新的
        if (!sessions || sessions.length === 0) {
          await actions.createNewChatSession(userId);
        } else {
          // 如果有现有会话，切换到第一个会话
          if (!state.currentChatSession) {
            // 此时state.chatSessions可能是旧的，localStorage也可能没更新
            // 但我们有sessions变量
            // 为了简化，我们手动在这里设置currentChatSession，或者修改switchToChatSession
            // 既然修改了switchToChatSession接受userId，我们传入它
            // 但switchToChatSession内部查找session的逻辑还是依赖state或localStorage
            
            // 更好的做法：直接在这里dispatch
            const firstSession = sessions[0];
            const currentChatSession: CurrentChatSession = {
              chat_id: firstSession.chat_id,
              user_id: firstSession.user_id,
              title: firstSession.title,
              history: firstSession.history,
            };
            dispatch({ type: 'SET_CURRENT_CHAT_SESSION', payload: currentChatSession });
          }
        }

        await actions.loadFiles();
      } catch (error) {
        console.error('登录失败:', error);
        const errorMessage = error instanceof Error ? error.message : '登录失败，请检查网络连接';
        dispatch({ type: 'SET_ERROR', payload: errorMessage });
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    },

    createNewChatSession: async (userId?: string) => {
      const targetUserId = userId || state.userId;
      if (!targetUserId) return;

      const chatId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const now = new Date();
      const year = now.getFullYear();
      const month = (now.getMonth() + 1).toString().padStart(2, '0');
      const day = now.getDate().toString().padStart(2, '0');
      const hour = now.getHours().toString().padStart(2, '0');
      const minute = now.getMinutes().toString().padStart(2, '0');
      const timeString = `${year}-${month}-${day} ${hour}:${minute}`;
      
      const newChatSession: ChatSession = {
        chat_id: chatId,
        user_id: targetUserId,
        title: timeString,
        history: [],
        created_at: now.toISOString(),
        last_active: now.toISOString(),
        message_count: 0,
      };

      dispatch({ type: 'CREATE_NEW_CHAT_SESSION', payload: newChatSession });
      
      // 创建新会话时，同步保存到服务器
      // 注意：这里我们需要保存完整的会话列表，所以需要获取最新的state
      // 但由于dispatch是异步的，state.chatSessions还没更新
      // 所以我们手动构造新的列表
      const updatedSessions = [...state.chatSessions, newChatSession];
      try {
        await apiService.saveUserSessions(targetUserId, updatedSessions);
      } catch (error) {
        console.error('保存新会话到服务器失败:', error);
      }

      return newChatSession;
    },

    logout: () => {
      wsService.disconnect();
      dispatch({ type: 'LOGOUT' });
    },

    sendMessage: async (message: string) => {
      if (!state.userId || !state.currentChatSession) return;

      try {
        // 设置为响应状态
        dispatch({ type: 'SET_RESPONDING', payload: true });

        // 添加用户消息到历史
        const userMessage: ChatMessage = {
          role: 'user',
          content: message,
          timestamp: new Date().toISOString(),
        };
        dispatch({ type: 'ADD_CHAT_MESSAGE', payload: userMessage });

        // 添加空白的助手消息用于流式更新
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: '',
        };
        dispatch({ type: 'ADD_CHAT_MESSAGE', payload: assistantMessage });

        // 通过WebSocket发送消息
        wsService.sendMessage(message, state.currentChatSession.chat_id);
      } catch (error) {
        console.error("sendMessage 动作捕获到错误:", error);
        dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : '发送消息失败' });
        dispatch({ type: 'SET_RESPONDING', payload: false });
      }
    },

    loadChatHistory: async (userId?: string) => {
      const targetUserId = userId || state.userId;
      // 从服务器加载用户的所有聊天会话
      if (!targetUserId) return [];

      try {
        // 优先尝试从服务器加载
        try {
          const { sessions } = await apiService.getUserSessions(targetUserId);
          if (sessions && sessions.length > 0) {
            dispatch({ type: 'SET_CHAT_SESSIONS', payload: sessions });
            return sessions;
          }
        } catch (serverError) {
          console.warn('从服务器加载会话失败，尝试本地缓存:', serverError);
        }

        // 如果服务器没有或失败，尝试从本地加载
        const savedSessions = localStorage.getItem(`chat_sessions_${targetUserId}`);
        if (savedSessions) {
          const chatSessions: ChatSession[] = JSON.parse(savedSessions);
          dispatch({ type: 'SET_CHAT_SESSIONS', payload: chatSessions });
          return chatSessions;
        }
      } catch (error) {
        console.error('加载聊天会话失败:', error);
      }
      return [];
    },

    switchToChatSession: async (chatId: string, userId?: string) => {
      const targetUserId = userId || state.userId;
      if (!targetUserId) return;

      try {
        // 优先从state中查找，因为state应该是最新的
        // 注意：这里的state.chatSessions可能也是旧的，如果刚加载完
        // 但我们已经通过loadChatHistory更新了state，不过在同一个闭包里state没变
        // 所以这里最好重新获取一下，或者依赖传入的参数（如果能传入sessions最好）
        // 简单起见，我们尝试从localStorage读取作为兜底，或者信任state（如果是在后续渲染周期调用）
        
        // 在login流程中，loadChatHistory刚跑完，state.chatSessions是旧的([])
        // 所以必须从localStorage或者API重新获取，或者...
        // 实际上loadChatHistory返回了sessions，login函数里有sessions变量
        // 但switchToChatSession无法直接访问那个变量除非传进来
        
        // 让我们尝试从localStorage读取，因为loadChatHistory应该已经更新了localStorage (通过useEffect? 不，useEffect也是异步的)
        // 等等，loadChatHistory dispatch了SET_CHAT_SESSIONS，但useEffect依赖state.chatSessions
        // 如果state没变，useEffect不会触发？
        // 不，dispatch触发重渲染，useEffect在重渲染后运行。
        // 但在login函数执行期间，重渲染还没发生。
        
        // 所以：loadChatHistory -> dispatch -> (login continues) -> switchToChatSession
        // 此时 localStorage 可能还没更新！
        
        // 最稳妥的办法：Login直接调用 dispatch({ type: 'SET_CURRENT_CHAT_SESSION', ... }) 
        // 而不是调用 switchToChatSession action?
        // 或者让 switchToChatSession 接受 sessions 数组?
        
        let targetSession = state.chatSessions.find(s => s.chat_id === chatId);
        
        if (!targetSession) {
             const savedSessions = localStorage.getItem(`chat_sessions_${targetUserId}`);
             if (savedSessions) {
               const chatSessions: ChatSession[] = JSON.parse(savedSessions);
               targetSession = chatSessions.find(s => s.chat_id === chatId);
             }
        }
        
        // 如果还是找不到（因为localStorage也没更新），我们需要一种方式
        // 也许我们应该让 switchToChatSession 支持传入 session 对象？
        
        if (targetSession) {
            const currentChatSession: CurrentChatSession = {
              chat_id: targetSession.chat_id,
              user_id: targetSession.user_id,
              title: targetSession.title,
              history: targetSession.history,
            };
            dispatch({ type: 'SET_CURRENT_CHAT_SESSION', payload: currentChatSession });
        }
      } catch (error) {
        console.error('切换聊天会话失败:', error);
      }
    },

    loadFiles: async () => {
      if (!state.userId) return;

      try {
        const { files } = await apiService.getFiles(state.userId);
        dispatch({ type: 'UPDATE_FILES', payload: files });
      } catch (error) {
        console.error('加载文件列表失败:', error);
      }
    },

    uploadFiles: async (files: File[]) => {
      if (!state.userId) return;

      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        await apiService.uploadFiles(state.userId, files);
        await actions.loadFiles(); // 重新加载文件列表
      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : '文件上传失败' });
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    },

    deleteFile: async (filename: string) => {
      if (!state.userId) return;

      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        await apiService.deleteFile(state.userId, filename);
        message.success('文件已删除');
        await actions.loadFiles(); // 重新加载文件列表
      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : '删除文件失败' });
        message.error('删除文件失败');
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    },

    clearCurrentChatHistory: async () => {
      if (!state.currentChatSession) return;

      try {
        // 清空当前聊天会话的历史
        const clearedSession = {
          ...state.currentChatSession,
          history: []
        };
        dispatch({ type: 'SET_CURRENT_CHAT_SESSION', payload: clearedSession });

        // 更新chatSessions中的对应会话
        const updatedSessions = state.chatSessions.map(session =>
          session.chat_id === state.currentChatSession?.chat_id
            ? { ...session, history: [], last_active: new Date().toISOString(), message_count: 0 }
            : session
        );
        dispatch({ type: 'SET_CHAT_SESSIONS', payload: updatedSessions });

        // 保存到localStorage
        if (state.userId) {
          localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(updatedSessions));
        }
      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : '清空聊天历史失败' });
      }
    },

    modifyParameters: async (modifiedSchema: any) => {
      if (!state.userId) return;

      try {
        await apiService.modifyParameters(state.userId, modifiedSchema);
        // 清除待处理的工具响应
        dispatch({ type: 'SET_PENDING_TOOL_RESPONSE', payload: '' });
      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: error instanceof Error ? error.message : '参数修改失败' });
      }
    },

    getCurrentSchema: async () => {
      if (!state.userId) return null;

      try {
        const { schema } = await apiService.getCurrentSchema(state.userId);
        return schema;
      } catch (error) {
        console.error('获取参数schema失败:', error);
        return null;
      }
    },

    deleteChatSession: async (chatId: string) => {
      if (!state.userId) return;

      try {
        const updatedSessions = state.chatSessions.filter(session => session.chat_id !== chatId);
        
        // Update state
        dispatch({ type: 'SET_CHAT_SESSIONS', payload: updatedSessions });
        
        // Sync to backend
        await apiService.saveUserSessions(state.userId, updatedSessions);
        
        // Sync to localStorage
        localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(updatedSessions));
        
        return updatedSessions;
      } catch (error) {
        console.error('删除会话失败:', error);
        message.error('同步删除到服务器失败');
        throw error;
      }
    },

    updateSessionTitle: async (chatId: string, title: string) => {
      if (!state.userId) return;

      try {
        const updatedSessions = state.chatSessions.map(session => 
          session.chat_id === chatId ? { ...session, title } : session
        );
        
        // Update state
        dispatch({ type: 'SET_CHAT_SESSIONS', payload: updatedSessions });
        
        // Sync to backend
        await apiService.saveUserSessions(state.userId, updatedSessions);
        
        // Sync to localStorage
        localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(updatedSessions));
      } catch (error) {
        console.error('更新标题失败:', error);
        message.error('同步标题到服务器失败');
        throw error;
      }
    },

    toggleLanguage: () => {
      const newLang = state.language === 'zh' ? 'en' : 'zh';
      dispatch({ type: 'SET_LANGUAGE', payload: newLang });
      localStorage.setItem('app_language', newLang);
    },
  };

  return (
    <AppContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </AppContext.Provider>
  );
}

// Hook for using context
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}