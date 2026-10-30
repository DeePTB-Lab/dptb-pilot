import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layout,
  Button,
  Input,
  Typography,
  Space,
  message,
  Spin,
  Alert,
  Tooltip,
  Upload,
  Avatar,
  Tag
} from 'antd';
import {
  SendOutlined,
  ClearOutlined,
  LogoutOutlined,
  UserOutlined,
  RobotOutlined,
  PaperClipOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  SettingOutlined,
  SearchOutlined,
  GlobalOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import StructureViewer from '../StructureViewer';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { useApp } from '../../contexts/AppContext';
import FilePanel from '../FilePanel';
import ParamPanel from '../ParamPanel';
import SessionPanel from '../SessionPanel';
import WorkflowPanel from '../WorkflowPanel';
import { translations } from '../../utils/i18n';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { TextArea } = Input;

function Chat() {
  const navigate = useNavigate();
  const { state, actions } = useApp();
  const [inputValue, setInputValue] = useState('');
  const [activeTab, setActiveTab] = useState('params');
  // 判断是否为移动端，设置侧边栏初始状态
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(
    window.innerWidth <= 768
  );
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(
    window.innerWidth <= 768
  );

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const t = translations[state.language];

  // Shortcut Cards Data
  const SHORTCUT_CARDS = [
    {
      icon: <ExperimentOutlined style={{ fontSize: '24px', color: '#0ea5e9' }} />,
      title: t.analyzeStructures,
      desc: t.analyzeStructuresDesc,
      prompt: '请帮我分析一下上传的晶体结构文件'
    },
    {
      icon: <LineChartOutlined style={{ fontSize: '24px', color: '#8b5cf6' }} />,
      title: t.calculateBands,
      desc: t.calculateBandsDesc,
      prompt: '请帮我计算并绘制能带图'
    },
    {
      icon: <SettingOutlined style={{ fontSize: '24px', color: '#10b981' }} />,
      title: t.generateConfigs,
      desc: t.generateConfigsDesc,
      prompt: '请帮我生成DeepTB训练配置文件'
    },
    {
      icon: <SearchOutlined style={{ fontSize: '24px', color: '#f59e0b' }} />,
      title: t.searchMaterials,
      desc: t.searchMaterialsDesc,
      prompt: '请帮我搜索相关的材料数据'
    }
  ];

  useEffect(() => {
    if (!state.isAuthenticated) {
      navigate('/login');
      return;
    }
  }, [state.isAuthenticated, navigate]);

  useEffect(() => {
    scrollToBottom();
  }, [state.currentChatSession?.history]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (text?: string) => {
    const content = text || inputValue;
    if (!content.trim() || !state.userId || !state.currentChatSession) return;

    setInputValue('');
    try {
      await actions.sendMessage(content);
    } catch (error) {
      message.error(t.sendFailed);
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
      message.success(t.clearSuccess);
    } catch (error) {
      message.error(t.clearFailed);
    }
  };

  const handleLogout = () => {
    actions.logout();
    navigate('/login');
  };

  const [uploading, setUploading] = useState(false);

  // 处理左侧栏折叠/展开（移动端：互斥；电脑端：独立）
  const handleLeftSidebarToggle = () => {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
      // 移动端：打开左侧栏时关闭右侧栏
      if (leftSidebarCollapsed) {
        setRightSidebarCollapsed(true);
        setLeftSidebarCollapsed(false);
      } else {
        setLeftSidebarCollapsed(true);
      }
    } else {
      // 电脑端：独立切换
      setLeftSidebarCollapsed(!leftSidebarCollapsed);
    }
  };

  // 处理右侧栏折叠/展开（移动端：互斥；电脑端：独立）
  const handleRightSidebarToggle = () => {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
      // 移动端：打开右侧栏时关闭左侧栏
      if (rightSidebarCollapsed) {
        setLeftSidebarCollapsed(true);
        setRightSidebarCollapsed(false);
      } else {
        setRightSidebarCollapsed(true);
      }
    } else {
      // 电脑端：独立切换
      setRightSidebarCollapsed(!rightSidebarCollapsed);
    }
  };

  // 监听schema变化，当有参数需要输入时自动展开右侧栏并切换到参数页面
  useEffect(() => {
    const checkSchemaAndOpenRightPanel = async () => {
      if (!state.userId) return;
      try {
        const schema = await actions.getCurrentSchema();
        if (schema && Object.keys(schema).length > 0) {
          // 有参数需要输入，展开右侧栏并切换到参数页面
          setActiveTab('params');
          if (rightSidebarCollapsed) {
            // 移动端：关闭左侧栏，展开右侧栏
            const isMobile = window.innerWidth <= 768;
            if (isMobile && !leftSidebarCollapsed) {
              setLeftSidebarCollapsed(true);
            }
            setRightSidebarCollapsed(false);
          }
        }
      } catch (error) {
        console.error('检查schema失败:', error);
      }
    };

    // 每2秒检查一次
    const interval = setInterval(checkSchemaAndOpenRightPanel, 2000);
    return () => clearInterval(interval);
  }, [state.userId, actions, rightSidebarCollapsed, leftSidebarCollapsed]);

  // 监听activeTab变化，当切换到参数页面时检查是否需要展开右侧栏
  useEffect(() => {
    if (activeTab === 'params') {
      // 检查是否有schema需要输入
      actions.getCurrentSchema().then(schema => {
        if (schema && Object.keys(schema).length > 0 && rightSidebarCollapsed) {
          // 移动端：关闭左侧栏，展开右侧栏
          const isMobile = window.innerWidth <= 768;
          if (isMobile && !leftSidebarCollapsed) {
            setLeftSidebarCollapsed(true);
          }
          setRightSidebarCollapsed(false);
        }
      });
    }
  }, [activeTab, actions, rightSidebarCollapsed, leftSidebarCollapsed]);

  useEffect(() => {
    if (state.pendingWorkflowPlan) {
      setActiveTab('workflow');
      if (rightSidebarCollapsed) {
        const isMobile = window.innerWidth <= 768;
        if (isMobile && !leftSidebarCollapsed) {
          setLeftSidebarCollapsed(true);
        }
        setRightSidebarCollapsed(false);
      }
    }
  }, [state.pendingWorkflowPlan, rightSidebarCollapsed, leftSidebarCollapsed]);

  const handleUpload = async (file: File) => {
    if (!state.userId) {
      message.error(t.loginRequired);
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      message.error(t.fileSizeLimit);
      return false;
    }

    setUploading(true);
    try {
      await actions.uploadFiles([file]);
      message.success(`${t.uploadSuccess}: ${file.name}`);
    } catch (error) {
      message.error(`${t.uploadFailed}: ${file.name}`);
    } finally {
      setUploading(false);
    }

    return false;
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
              img(props) {
                return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
              }
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
              img(props) {
                return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
              }
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
          img(props) {
            return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  const renderMessage = (msg: any, index: number) => {
    const isUser = msg.role === 'user';
    const isComplete = !!msg.timestamp;
    const history = state.currentChatSession?.history || [];
    const isLastMessage = index === history.length - 1;

    return (
      <div
        key={index}
        style={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          marginBottom: '24px'
        }}
      >
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: isUser ? 'flex-end' : 'flex-start',
          maxWidth: '75%',
          width: 'auto'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            width: 'auto'
          }}>
            {!isUser && (
              <Avatar
                icon={<RobotOutlined />}
                style={{ marginRight: '16px', marginTop: '4px', backgroundColor: '#1e293b', flexShrink: 0 }}
              />
            )}
            <div className={`chat-message ${isUser ? 'user' : 'assistant'}`} style={{
              backgroundColor: isUser ? '#2563eb' : '#1e293b', // blue-600 : slate-800
              color: isUser ? '#ffffff' : '#e2e8f0', // white : slate-200
              padding: '12px 16px',
              borderRadius: '12px',
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)'
            }}>
              {isUser ? (
                <Text style={{ color: 'white' }}>{msg.content}</Text>
              ) : (
                <div className="markdown-content">
                  {formatMessage(msg.content)}
                </div>
              )}
              {/* 只在最后一条未完成的AI消息显示加载图标 */}
              {!isUser && !isComplete && isLastMessage && state.responding && (
                <div style={{ display: 'inline-block', marginLeft: '8px' }}>
                  <Spin size="small" />
                </div>
              )}
            </div>
            {isUser && (
              <Avatar
                icon={<UserOutlined />}
                style={{ marginLeft: '16px', marginTop: '4px', backgroundColor: '#0ea5e9', flexShrink: 0 }}
              />
            )}
          </div>

          {/* Token使用量和收费信息 - 仅对AI消息显示 */}
          {!isUser && isComplete && (
            <div style={{
              marginTop: '8px',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
              padding: '0 12px'
            }}>
              {/* Token使用量显示 */}
              {msg.usage_metadata && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '12px',
                  color: '#94a3b8'
                }}>
                  <span>字数: {msg.content.length}</span>
                  <span style={{ color: '#475569' }}>|</span>
                  <span>输入token: {msg.usage_metadata.prompt_tokens || 0}</span>
                  <span style={{ color: '#475569' }}>|</span>
                  <span>输出token: {msg.usage_metadata.candidates_tokens || 0}</span>
                </div>
              )}

              {/* 收费信息显示 */}
              {msg.charge_result && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  backgroundColor: msg.charge_result.success
                    ? ((msg.charge_result.photon_amount || 0) > 0
                        ? 'rgba(34, 197, 94, 0.1)'  // green-100
                        : (msg.charge_result.message?.includes('累积')
                            ? 'rgba(59, 130, 246, 0.1)'  // blue-500
                            : 'rgba(148, 163, 184, 0.1)'))  // gray-400
                    : 'rgba(239, 68, 68, 0.1)',  // red-500
                  border: `1px solid ${msg.charge_result.success
                    ? ((msg.charge_result.photon_amount || 0) > 0
                        ? 'rgba(34, 197, 94, 0.2)'
                        : (msg.charge_result.message?.includes('累积')
                            ? 'rgba(59, 130, 246, 0.2)'
                            : 'rgba(148, 163, 184, 0.2)'))
                    : 'rgba(239, 68, 68, 0.2)'}`,
                  color: msg.charge_result.success
                    ? ((msg.charge_result.photon_amount || 0) > 0
                        ? '#22c55e'  // green-500
                        : (msg.charge_result.message?.includes('累积')
                            ? '#3b82f6'  // blue-500
                            : '#94a3b8'))  // gray-400
                    : '#ef4444'  // red-500
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: msg.charge_result.success
                      ? ((msg.charge_result.photon_amount || 0) > 0
                          ? '#22c55e'
                          : (msg.charge_result.message?.includes('累积')
                              ? '#3b82f6'
                              : '#94a3b8'))
                      : '#ef4444'
                  }}></div>
                  <span style={{ fontWeight: 500 }}>
                    {msg.charge_result.success
                      ? ((msg.charge_result.photon_amount || 0) > 0
                          ? '✓ 收费成功'
                          : (msg.charge_result.message?.includes('累积')
                              ? '⏳ 费用累积中'
                              : '✓ 免费使用'))
                      : '✗ 收费失败'
                    }
                  </span>
                  <span style={{ color: '#64748b' }}>|</span>
                  <span>
                    消耗光子 {msg.charge_result.photon_amount || 0} | RMB {(msg.charge_result.rmb_amount || 0).toFixed(2)} 元
                  </span>
                  {msg.charge_result.biz_no && (
                    <>
                      <span style={{ color: '#64748b' }}>|</span>
                      <span style={{ fontFamily: 'monospace' }}>订单: {msg.charge_result.biz_no}</span>
                    </>
                  )}
                </div>
              )}
            </div>
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
        alignItems: 'center',
        background: '#020617' // slate-950
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <Layout style={{ height: '100vh', maxHeight: '100vh', overflow: 'hidden', background: '#020617' }}> {/* slate-950 */}
      {/* Top Navigation */}
      <Header className="app-header" style={{
        background: 'rgba(15, 23, 42, 0.8)', // slate-900/80
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '0 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        height: '64px',
        zIndex: 10,
        position: 'relative',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* 左侧栏折叠/展开按钮 - 仅移动端 */}
          <Tooltip title={t.chatHistory}>
            <Button
              icon={leftSidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={handleLeftSidebarToggle}
              className="glass-btn mobile-only"
              shape="circle"
              size="small"
            />
          </Tooltip>
          <img
            src="/pilot_logo_white.png"
            alt="DeepTB Pilot"
            style={{
              height: '32px',
              filter: 'drop-shadow(0 0 5px rgba(59, 130, 246, 0.5))'
            }}
          />
        </div>

        <Space size="small">
          {/* 右侧栏折叠/展开按钮 - 仅移动端 */}
          <Tooltip title={t.parameters}>
            <Button
              icon={<SettingOutlined />}
              onClick={handleRightSidebarToggle}
              className="glass-btn mobile-only"
              shape="circle"
              size="small"
              style={{ color: rightSidebarCollapsed ? '#94a3b8' : '#0ea5e9' }}
            />
          </Tooltip>
          {/* 语言切换按钮 - 桌面端显示文字 */}
          <Button
            icon={<GlobalOutlined />}
            onClick={actions.toggleLanguage}
            className="glass-btn desktop-only"
            shape="round"
            size="small"
            style={{ color: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)' }}
          >
            {state.language === 'zh' ? '中 / En' : 'En / 中'}
          </Button>
          {/* 移动端只显示语言图标 */}
          <Button
            icon={<GlobalOutlined />}
            onClick={actions.toggleLanguage}
            className="glass-btn mobile-only"
            shape="circle"
            size="small"
            style={{ color: '#94a3b8' }}
          />
          <Tooltip title={`Session ID: ${state.userId}${state.clientName ? `\nClient Name: ${state.clientName}` : ''}`}>
            <Tag icon={<UserOutlined />} color="blue" className="desktop-only" style={{
              margin: 0,
              padding: '4px 10px',
              borderRadius: '20px',
              background: 'rgba(14, 165, 233, 0.1)',
              border: '1px solid rgba(14, 165, 233, 0.2)',
              color: '#0ea5e9',
              fontSize: '12px'
            }}>
              {state.clientName || `${state.userId?.slice(0, 4)}...${state.userId?.slice(-4)}`}
            </Tag>
          </Tooltip>
          <Tooltip title={t.clearChat}>
            <Button
              icon={<ClearOutlined />}
              onClick={handleClearChat}
              disabled={state.loading}
              className="glass-btn"
              shape="circle"
              size="small"
            />
          </Tooltip>
          <Tooltip title={t.logout}>
            <Button
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              className="glass-btn"
              shape="circle"
              size="small"
            />
          </Tooltip>
        </Space>
      </Header>

      <Layout style={{ background: 'transparent', overflow: 'hidden' }}>
        {/* Left Sidebar: History */}
        <Sider
          width={280}
          collapsed={leftSidebarCollapsed}
          collapsedWidth={0}
          className="glass-panel mobile-sidebar"
          style={{
            borderRight: '1px solid rgba(255, 255, 255, 0.05)',
            background: '#0f172a', // slate-900
            transition: 'all 0.2s',
            position: 'relative',
            zIndex: 20,
            height: '100%',
            overflow: 'auto'
          }}
        >
          <div style={{ padding: '20px', height: '100%' }}>
            <SessionPanel />
          </div>
        </Sider>

        {/* Center: Chat Area */}
        <Content style={{ display: 'flex', position: 'relative', background: '#020617', zIndex: 10, overflow: 'hidden' }}> {/* slate-950 */}
          <div 
            style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDrop={(e) => {
              e.preventDefault(); e.stopPropagation();
              const files = Array.from(e.dataTransfer.files);
              if (files.length > 0) files.forEach(file => handleUpload(file));
            }}
          >
            <div
              ref={chatContainerRef}
              className="chat-container"
              style={{
                flex: 1,
                padding: '24px 40px 100px 40px', // Extra padding at bottom for floating input
                overflowY: 'auto',
                backgroundColor: 'transparent'
              }}
            >
              {(!state.currentChatSession?.history || state.currentChatSession.history.length === 0) ? (
                <div className="welcome-container" style={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  maxWidth: '800px',
                  margin: '0 auto',
                  width: '100%'
                }}>
                  <Title level={2} style={{ marginBottom: '10px', textAlign: 'center', color: '#e2e8f0' }}>
                    {t.welcomeTitle}
                  </Title>
                  <Text style={{ marginBottom: '40px', textAlign: 'center', color: '#94a3b8', fontSize: '16px' }}>
                    {t.welcomeSubtitle}
                  </Text>

                  <div className="shortcut-cards-grid">
                    {SHORTCUT_CARDS.map((card, idx) => (
                      <div
                        key={idx}
                        className="glass-card"
                        style={{
                          padding: '24px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '16px',
                          background: '#1e293b', // slate-800
                          border: '1px solid #334155' // slate-700
                        }}
                        onClick={() => handleSendMessage(card.prompt)}
                      >
                        <div style={{
                          width: '48px',
                          height: '48px',
                          borderRadius: '12px',
                          background: 'rgba(255,255,255,0.05)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          {card.icon}
                        </div>
                        <div>
                          <Text strong style={{ fontSize: '16px', display: 'block', color: '#e2e8f0' }}>{card.title}</Text>
                          <Text style={{ fontSize: '13px', color: '#94a3b8' }}>{card.desc}</Text>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {state.currentChatSession?.history.map((msg, index) => renderMessage(msg, index))}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Floating Input Area */}
            <div style={{ 
              position: 'absolute', 
              bottom: '30px', 
              left: '50%', 
              transform: 'translateX(-50%)',
              width: '90%',
              maxWidth: '800px',
              zIndex: 20
            }}>
              {state.error && (
                <Alert
                  message={state.error}
                  type="error"
                  showIcon
                  closable
                  style={{ marginBottom: '16px', borderRadius: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#fca5a5' }}
                />
              )}
              
              {state.agentStatus !== 'idle' && (
                <div style={{ color: '#94a3b8', fontSize: '12px', margin: '0 8px 8px', minHeight: '18px' }}>
                  {state.agentStatus === 'waiting_for_reply' ? 'Agent 正在等待回复' : 'Agent 执行中'}
                  {state.agentProgress ? ` · ${state.agentProgress}` : ''}
                </div>
              )}
              <div style={{
                backgroundColor: '#1e293b', // slate-800
                borderRadius: '30px',
                padding: '8px 16px',
                boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
                border: '1px solid #334155', // slate-700
                display: 'flex',
                alignItems: 'flex-end',
                gap: '12px'
              }}>
                <Upload
                  beforeUpload={handleUpload}
                  showUploadList={false}
                  multiple
                  disabled={uploading || state.loading}
                >
                  <Button 
                    icon={<PaperClipOutlined style={{ fontSize: '20px' }} />} 
                    loading={uploading}
                    type="text"
                    style={{ 
                      color: '#94a3b8', 
                      width: '40px', 
                      height: '40px',
                      marginBottom: '2px'
                    }}
                  />
                </Upload>
                
                <TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onPaste={(e) => {
                    const items = e.clipboardData.items;
                    for (let i = 0; i < items.length; i++) {
                      if (items[i].kind === 'file') {
                        const file = items[i].getAsFile();
                        if (file) handleUpload(file);
                      }
                    }
                  }}
                  placeholder={t.inputPlaceholder}
                  autoSize={{ minRows: 1, maxRows: 6 }}
                  variant="borderless"
                  style={{ 
                    flex: 1, 
                    resize: 'none', 
                    padding: '10px 0',
                    fontSize: '16px',
                    color: '#f1f5f9',
                    background: 'transparent'
                  }}
                  disabled={state.responding || state.loading}
                />
                
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={() => handleSendMessage()}
                  loading={state.responding || state.loading}
                  disabled={!inputValue.trim()}
                  className="send-btn"
                  style={{ marginBottom: '4px' }}
                />
              </div>
            </div>
          </div>

          {/* Right Sidebar */}
          <Sider
            width={340}
            collapsed={rightSidebarCollapsed}
            collapsedWidth={0}
            theme="light"
            className="mobile-sidebar"
            style={{
              borderLeft: '1px solid rgba(51, 65, 85, 0.5)', // border-slate-700
              background: '#0f172a', // slate-900
              padding: rightSidebarCollapsed ? '0' : '16px', // p-4
              transition: 'all 0.2s',
              position: 'relative',
              zIndex: 20,
              height: '100%',
              overflow: 'auto'
            }}
          >
            {!rightSidebarCollapsed && (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                {/* Segmented Control */}
                <div style={{
                  width: '100%',
                  backgroundColor: '#1e293b', // slate-800
                  padding: '4px',
                  borderRadius: '8px',
                  display: 'flex',
                  marginBottom: '24px'
                }}>
                  <div
                    onClick={() => setActiveTab('params')}
                    style={{
                      flex: 1,
                      textAlign: 'center',
                      padding: '6px 0',
                      borderRadius: '6px',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      backgroundColor: activeTab === 'params' ? '#2563eb' : 'transparent', // blue-600
                      color: activeTab === 'params' ? 'white' : '#94a3b8', // slate-400
                      boxShadow: activeTab === 'params' ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'
                    }}
                  >
                    {t.parameters}
                  </div>
                  <div
                    onClick={() => setActiveTab('files')}
                    style={{
                      flex: 1,
                      textAlign: 'center',
                      padding: '6px 0',
                      borderRadius: '6px',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      backgroundColor: activeTab === 'files' ? '#2563eb' : 'transparent',
                      color: activeTab === 'files' ? 'white' : '#94a3b8',
                      boxShadow: activeTab === 'files' ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'
                    }}
                  >
                    {t.files}
                  </div>
                  <div
                    onClick={() => setActiveTab('workflow')}
                    style={{
                      flex: 1,
                      textAlign: 'center',
                      padding: '6px 0',
                      borderRadius: '6px',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      backgroundColor: activeTab === 'workflow' ? '#2563eb' : 'transparent',
                      color: activeTab === 'workflow' ? 'white' : '#94a3b8',
                      boxShadow: activeTab === 'workflow' ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'
                    }}
                  >
                    Workflow
                  </div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto' }}>
                  {activeTab === 'params' ? <ParamPanel /> : activeTab === 'workflow' ? <WorkflowPanel /> : <FilePanel />}
                </div>
              </div>
            )}
          </Sider>
        </Content>
      </Layout>
    </Layout>
  );
}

export default Chat;
