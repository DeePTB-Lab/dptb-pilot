import { useState } from 'react';
import {
  Button,
  List,
  Typography,
  message,
  Modal,
  Input,
  Tooltip
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { ChatSession } from '../../types';
import { translations } from '../../utils/i18n';

const { Text } = Typography;

function SessionPanel() {
  const { state, actions } = useApp();
  const t = translations[state.language];
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  // Create new chat session
  const createNewChatSession = async () => {
    try {
      await actions.createNewChatSession();
      message.success(t.newChatCreated);
    } catch (error) {
      message.error(t.failedCreateSession);
    }
  };

  const switchToChatSession = async (chatId: string) => {
    if (chatId === state.currentChatSession?.chat_id) {
      return;
    }

    try {
      // 如果AI正在响应，先清除响应状态
      if (state.responding) {
        // 不需要等待，直接切换会话即可
        // 新会话加载后会自动清除响应状态
      }
      await actions.switchToChatSession(chatId);
    } catch (error) {
      message.error(t.failedSwitchSession);
    }
  };

  const handleEditSession = (session: ChatSession) => {
    setEditingSession(session.chat_id);
    setEditingTitle(session.title);
  };

  const saveSessionTitle = async (chatId: string) => {
    if (!editingTitle.trim()) {
      message.error(t.sessionTitleEmpty);
      return;
    }

    try {
      await actions.updateSessionTitle(chatId, editingTitle.trim());
      message.success(t.sessionTitleUpdated);
    } catch (error) {
      // Error handled in action
    }
    
    setEditingSession(null);
    setEditingTitle('');
  };

  const handleDeleteSession = (chatId: string) => {
    Modal.confirm({
      title: t.deleteSessionTitle,
      content: t.deleteSessionConfirm,
      okText: t.delete,
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          const updatedSessions = await actions.deleteChatSession(chatId);
          message.success(t.sessionDeleted);

          // If deleted session was active
          if (chatId === state.currentChatSession?.chat_id) {
            if (updatedSessions && updatedSessions.length > 0) {
              actions.switchToChatSession(updatedSessions[0].chat_id);
            } else {
              actions.createNewChatSession();
            }
          }
        } catch (error) {
          // Error handled in action
        }
      }
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* New Chat Button Area */}
      <div style={{ marginBottom: '24px' }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={createNewChatSession}
          block
          size="large"
          style={{
            height: '48px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            border: 'none',
            boxShadow: '0 4px 14px 0 rgba(37, 99, 235, 0.5)',
            fontSize: '16px',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          {t.newChat}
        </Button>
      </div>

      {/* Header */}
      <div style={{ marginBottom: '12px', paddingLeft: '4px' }}>
        <Text style={{ color: '#94a3b8', fontSize: '14px', fontWeight: 500 }}>
          {t.chatHistory}
        </Text>
      </div>

      {/* Session List */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <List
          itemLayout="horizontal"
          dataSource={state.chatSessions}
          loading={state.loading}
          locale={{ emptyText: <Text style={{ color: '#64748b' }}>{t.noChatHistory}</Text> }}
          renderItem={(session: ChatSession) => {
            const isSelected = session.chat_id === state.currentChatSession?.chat_id;
            return (
              <List.Item
                style={{
                  padding: '12px 16px',
                  marginBottom: '8px',
                  borderRadius: '12px',
                  border: isSelected ? '1px solid rgba(56, 189, 248, 0.5)' : '1px solid transparent',
                  cursor: 'pointer',
                  backgroundColor: isSelected ? 'rgba(14, 165, 233, 0.15)' : 'transparent',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
                onClick={() => switchToChatSession(session.chat_id)}
                actions={isSelected ? [
                  <Tooltip title="Edit Title" key="edit">
                    <Button
                      type="text"
                      icon={<EditOutlined />}
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEditSession(session);
                      }}
                      style={{ color: '#94a3b8' }}
                    />
                  </Tooltip>,
                  <Tooltip title="Delete" key="delete">
                    <Button
                      type="text"
                      icon={<DeleteOutlined />}
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteSession(session.chat_id);
                      }}
                      style={{ color: '#94a3b8' }}
                    />
                  </Tooltip>
                ] : []}
              >
                <div style={{ flex: 1, overflow: 'hidden', marginRight: '8px' }}>
                  {editingSession === session.chat_id ? (
                      <Input
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onPressEnter={() => saveSessionTitle(session.chat_id)}
                        onBlur={() => saveSessionTitle(session.chat_id)}
                        size="small"
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                        style={{ 
                          background: 'rgba(0,0,0,0.2)', 
                          color: 'white', 
                          border: '1px solid #3b82f6',
                          borderRadius: '4px'
                        }}
                      />
                  ) : (
                    <Text
                      ellipsis
                      style={{
                        color: isSelected ? '#e2e8f0' : '#94a3b8',
                        fontSize: '14px',
                        fontWeight: isSelected ? 500 : 400,
                        display: 'block'
                      }}
                    >
                      {session.title || t.untitled}
                    </Text>
                  )}
                </div>
                
                {!isSelected && (
                  <Text style={{ color: '#475569', fontSize: '12px' }}>
                    {new Date(session.last_active || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </Text>
                )}
              </List.Item>
            );
          }}
        />
      </div>
    </div>
  );
}

export default SessionPanel;