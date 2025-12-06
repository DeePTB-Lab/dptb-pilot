import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layout,
  Button,
  Input,
  Select,
  Typography,
  Space,
  message,
  Spin,
  Divider,
  Alert,
  Tooltip
} from 'antd';
import {
  SendOutlined,
  ClearOutlined,
  LogoutOutlined,
  MessageOutlined,
  UserOutlined,
  RobotOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
// import Prism from 'prismjs';
// import 'prismjs/themes/prism-tomorrow.css';
import StructureViewer from '../StructureViewer';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { useApp } from '../../contexts/AppContext';
import FilePanel from '../FilePanel';
import ParamPanel from '../ParamPanel';
import SessionPanel from '../SessionPanel';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { TextArea } = Input;

// 示例对话选项
const EXAMPLE_MESSAGES = [
  {
    text: '请帮我生成碳的训练输入配置文件，基组为{"C":"2s1p"}，截断半径{"C":6.0}，训练数据路径"my_data"，前缀"C16"，其余按默认配置',
    display: '生成训练输入配置文件'
  },
  {
    text: '使用poly4基准模型绘制能带图，结构文件为xxx',
    display: '使用基准模型绘制能带图'
  },
  {
    text: '请帮我生成sp轨道的Si的ploy4基准模型',
    display: '生成基准模型'
  },
  {
    text: '请使用我的模型预测并绘制能带图',
    display: '使用模型预测并绘制能带图'
  }
];

function Chat() {
  const navigate = useNavigate();
  const { state, actions } = useApp();
  const [inputValue, setInputValue] = useState('');
  const [selectedExample, setSelectedExample] = useState<string>('-');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!state.isAuthenticated) {
      navigate('/login');
      return;
    }

    // 初始化Prism代码高亮（如果需要的话）
    // Prism.highlightAll();
  }, [state.isAuthenticated, navigate]);

  useEffect(() => {
    // 滚动到底部
    scrollToBottom();
  }, [state.currentChatSession?.history]);

  useEffect(() => {
    // 示例选择时自动填充消息
    if (selectedExample && selectedExample !== '-') {
      const example = EXAMPLE_MESSAGES.find(ex => ex.text === selectedExample);
      if (example) {
        setInputValue(example.text);
      }
    }
  }, [selectedExample]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    console.log('handleSendMessage function called');
    
    // 调试日志：检查发送前的状态
    console.log('Checking conditions:', {
      hasMessage: !!inputValue.trim(),
      hasUserId: !!state.userId,
      hasCurrentChatSession: !!state.currentChatSession,
    });

    if (!inputValue.trim() || !state.userId || !state.currentChatSession) {
      console.error('发送被阻止，因为上述条件之一不满足。');
      return;
    }

    const messageToSend = inputValue;
    setInputValue('');
    setSelectedExample('-');

    try {
      await actions.sendMessage(messageToSend);
    } catch (error) {
      message.error('发送消息失败');
      console.error('Send message error:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = async () => {
    try {
      await actions.clearCurrentChatHistory();
      message.success('当前聊天记录已清空');
    } catch (error) {
      message.error('清空聊天记录失败');
    }
  };

  const handleLogout = () => {
    actions.logout();
    navigate('/login');
  };

  const formatMessage = (content: string) => {
    const visualizeRegex = /:::visualize\n([\s\S]*?)\n:::/;
    const match = content.match(visualizeRegex);

    if (match) {
      try {
        const jsonStr = match[1];
        const { format, data } = JSON.parse(jsonStr);
        const parts = content.split(match[0]);

        return (
          <div>
            {parts[0] && <ReactMarkdown components={{
              code(props) {
                const {children, className, node, ...rest} = props;
                const match = /language-(\w+)/.exec(className || '');
                const isInline = (props as any).inline;
                const { ref, ...propsWithoutRef } = rest as any;
                return !isInline && match ? (
                  <SyntaxHighlighter
                    {...propsWithoutRef}
                    children={String(children).replace(/\n$/, '')}
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              },
              img: (props) => (
                <img {...props} style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }} />
              )
            }}>{parts[0]}</ReactMarkdown>}
            
            <div style={{ margin: '10px 0' }}>
              <StructureViewer data={data} format={format} />
            </div>
            
            {parts[1] && <ReactMarkdown components={{
              code(props) {
                const {children, className, node, ...rest} = props;
                const match = /language-(\w+)/.exec(className || '');
                const isInline = (props as any).inline;
                const { ref, ...propsWithoutRef } = rest as any;
                return !isInline && match ? (
                  <SyntaxHighlighter
                    {...propsWithoutRef}
                    children={String(children).replace(/\n$/, '')}
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              },
              img: (props) => (
                <img {...props} style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }} />
              )
            }}>{parts[1]}</ReactMarkdown>}
          </div>
        );
      } catch (e) {
        console.error("Failed to parse visualization block", e);
      }
    }

    return (
      <ReactMarkdown
        components={{
          code(props) {
            const {children, className, node, ...rest} = props;
            const match = /language-(\w+)/.exec(className || '');
            const isInline = (props as any).inline;
            const { ref, ...propsWithoutRef } = rest as any;
            return !isInline && match ? (
              <SyntaxHighlighter
                {...propsWithoutRef}
                children={String(children).replace(/\n$/, '')}
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
              />
            ) : (
              <code {...rest} className={className}>
                {children}
              </code>
            )
          },
          img: (props) => (
            <img {...props} style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }} />
          )
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  const renderMessage = (msg: any, index: number) => {
    const isUser = msg.role === 'user';
    const isComplete = !!msg.timestamp;

    return (
      <div
        key={index}
        className={`chat-message ${isUser ? 'user' : 'assistant'}`}
        style={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          marginBottom: '16px'
        }}
      >
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          maxWidth: isUser ? 'fit-content' : '70%',
          width: isUser ? 'auto' : '100%'
        }}>
          {!isUser && (
            <RobotOutlined style={{ marginRight: '8px', marginTop: '4px', color: '#1677ff' }} />
          )}
          <div
            style={{
              padding: '12px 16px',
              borderRadius: '12px',
              backgroundColor: isUser ? '#1677ff' : '#f5f5f5',
              color: isUser ? 'white' : '#262626',
              border: isUser ? 'none' : '1px solid #d9d9d9',
              width: isUser ? 'auto' : '100%',
              wordBreak: 'break-word'
            }}
          >
            {isUser ? (
              <Text>{msg.content}</Text>
            ) : (
              <div className="markdown-content">
                {formatMessage(msg.content)}
              </div>
            )}
            {!isComplete && state.responding && (
              <div style={{ display: 'inline-block', marginLeft: '8px' }}>
                <Spin size="small" />
              </div>
            )}
          </div>
          {isUser && (
            <UserOutlined style={{ marginLeft: '8px', marginTop: '4px', color: '#52c41a' }} />
          )}
        </div>
      </div>
    );
  };

  if (!state.isAuthenticated) {
    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{
        background: '#fff',
        borderBottom: '1px solid #f0f0f0',
        padding: '0 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <MessageOutlined style={{ marginRight: '8px', color: '#1677ff' }} />
          <Title level={4} style={{ margin: 0 }}>
            与 {state.config?.agent_info.name} 协作
          </Title>
        </div>

        <Space>
          <Tooltip title="用户会话ID">
            <Text code>{state.userId?.slice(0, 4)}****{state.userId?.slice(-4)}</Text>
          </Tooltip>
          <Button
            icon={<ClearOutlined />}
            onClick={handleClearChat}
            disabled={state.loading}
          >
            清空对话
          </Button>
          <Button
            icon={<LogoutOutlined />}
            onClick={handleLogout}
            type="default"
          >
            离开会话
          </Button>
        </Space>
      </Header>

      <Layout>
        <Sider width={300} style={{ background: '#fff', borderRight: '1px solid #f0f0f0' }}>
          <div style={{ padding: '16px', height: '100%', overflowY: 'auto' }}>
            <SessionPanel />
          </div>
        </Sider>

        <Content style={{ display: 'flex' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div
              ref={chatContainerRef}
              style={{
                flex: 1,
                padding: '16px 24px',
                overflowY: 'auto',
                backgroundColor: '#fafafa'
              }}
            >
              {state.currentChatSession?.history.map((msg, index) => renderMessage(msg, index))}
              <div ref={messagesEndRef} />
            </div>

            {state.error && (
              <Alert
                message="错误"
                description={state.error}
                type="error"
                showIcon
                closable
                onClose={() => {/* 清除错误 */}}
                style={{ margin: '0 24px 16px' }}
              />
            )}

            <div style={{ padding: '16px 24px', backgroundColor: '#fff', borderTop: '1px solid #f0f0f0' }}>
              <Space.Compact style={{ width: '100%', marginBottom: '12px' }}>
                <Select
                  value={selectedExample}
                  onChange={setSelectedExample}
                  style={{ flex: 1 }}
                  placeholder="选择示例对话"
                  options={[
                    { value: '-', label: '-' },
                    ...EXAMPLE_MESSAGES.map(ex => ({
                      value: ex.text,
                      label: ex.display
                    }))
                  ]}
                />
              </Space.Compact>

              <Space.Compact style={{ width: '100%' }}>
                <TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={`输入你想对 ${state.config?.agent_info.name} 说的话...`}
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  style={{ flex: 1, resize: 'none' }}
                  disabled={state.responding || state.loading}
                />
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleSendMessage}
                  loading={state.responding || state.loading}
                  disabled={!inputValue.trim()}
                  style={{ height: 'auto', minHeight: '32px' }}
                >
                  发送
                </Button>
              </Space.Compact>
            </div>
          </div>

          <Sider width={300} style={{ background: '#fff', borderLeft: '1px solid #f0f0f0' }}>
            <div style={{ padding: '16px', height: '100%', overflowY: 'auto' }}>
              <ParamPanel />
              <Divider />
              <FilePanel />
            </div>
          </Sider>
        </Content>
      </Layout>
    </Layout>
  );
}

export default Chat;