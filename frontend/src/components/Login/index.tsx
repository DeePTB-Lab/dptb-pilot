import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Input, Button, Typography, Form, Alert, Spin } from 'antd';
import { KeyOutlined } from '@ant-design/icons';
import { useApp } from '../../contexts/AppContext';

const { Title, Paragraph, Text } = Typography;

function generateRandomString(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

function Login() {
  const navigate = useNavigate();
  const { state, actions, dispatch } = useApp();
  const [sessionId, setSessionId] = useState('');
  const [form] = Form.useForm();

  useEffect(() => {
    if (state.isAuthenticated) {
      navigate('/chat');
    }
  }, [state.isAuthenticated, navigate]);

  useEffect(() => {
    const savedSessionId = localStorage.getItem('last_session_id');
    if (savedSessionId) {
      setSessionId(savedSessionId);
      form.setFieldsValue({ session_id: savedSessionId });
    } else {
      setSessionId('');
      form.setFieldsValue({ session_id: '' });
    }
  }, [form]);

  const handleGenerateRandom = () => {
    const newId = generateRandomString(32);
    setSessionId(newId);
    form.setFieldsValue({ session_id: newId });
  };

  const handleSubmit = async (values: { session_id: string }) => {
    if (!values.session_id || values.session_id.length !== 32) {
      return;
    }

    try {
      localStorage.setItem('last_session_id', values.session_id);
      await actions.login(values.session_id);
      navigate('/chat');
    } catch (error) {
      console.error('Login submit error:', error);
    }
  };

  const handleSessionIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const filteredValue = value.replace(/[^a-zA-Z0-9]/g, '').slice(0, 32);
    setSessionId(filteredValue);
    form.setFieldsValue({ session_id: filteredValue });
  };

  if (state.loading && !state.error) {
    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: '#020617'
      }}>
        <Spin size="large" tip="Connecting..." />
      </div>
    );
  }

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      background: '#020617', // slate-950
      backgroundImage: 'radial-gradient(circle at center, #0f172a 0%, #020617 100%)',
      padding: '20px',
      overflow: 'hidden'
    }}>
      <style>
        {`
          @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
          }
          .hologram-card {
            animation: float 6s ease-in-out infinite;
          }
          .login-input::placeholder {
            color: #0e7490 !important; /* cyan-700 */
          }
          .login-input:hover, .login-input:focus {
            border-bottom-color: #22d3ee !important; /* cyan-400 */
          }
        `}
      </style>

      <Card
        className="hologram-card"
        style={{
          width: '100%',
          maxWidth: 480,
          background: 'linear-gradient(to bottom, rgba(15, 23, 42, 0.1), rgba(30, 58, 138, 0.2))', // slate-900/10 to blue-900/20
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(6, 182, 212, 0.3)', // cyan-500/30
          boxShadow: '0 0 15px rgba(6, 182, 212, 0.15)', // cyan glow
          borderRadius: '16px'
        }}
        bodyStyle={{ padding: '40px' }}
      >
        {/* Top Logo Section */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ marginBottom: '24px' }}>
            <img 
              src="/pilot_logo_white.png" 
              alt="DeePTB Pilot" 
              style={{ 
                height: '64px', 
                maxWidth: '100%',
                filter: 'drop-shadow(0 0 10px rgba(59, 130, 246, 0.5))' // blue glow
              }} 
            />
          </div>

          <Paragraph style={{ color: '#0e7490', margin: '4px 0 0 0', fontSize: '10px', letterSpacing: '0.25em', textTransform: 'uppercase', fontWeight: 500 }}>
            // AI AGENT FOR DEEPTB //
          </Paragraph>
        </div>

        {state.error && (
          <Alert
            message="Connection Error"
            description={state.error}
            type="error"
            showIcon
            closable
            onClose={() => dispatch({ type: 'SET_ERROR', payload: null })}
            style={{ 
              marginBottom: '24px', 
              background: 'rgba(239, 68, 68, 0.1)', 
              border: '1px solid rgba(239, 68, 68, 0.2)', 
              color: '#fca5a5' 
            }}
          />
        )}

        <Form
          form={form}
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            label={<span style={{ color: '#22d3ee', fontSize: '10px', fontWeight: 500, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.1em', opacity: 0.8 }}>&gt; ENTER SESSION_KEY</span>}
            name="session_id"
            required={false}
            rules={[
              { required: true, message: 'Please enter Session ID' },
              { len: 32, message: 'Session ID must be 32 characters' }
            ]}
            style={{ marginBottom: '32px' }}
          >
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <KeyOutlined style={{ 
                position: 'absolute', 
                left: '16px', 
                zIndex: 1, 
                color: '#0e7490',
                fontSize: '16px'
              }} />
              
              <Input
                placeholder="ENTER 32-CHAR ID"
                value={sessionId}
                onChange={handleSessionIdChange}
                style={{ 
                  background: 'rgba(2, 6, 23, 0.5)', // slate-950/50
                  border: 'none',
                  borderBottom: '1px solid rgba(6, 182, 212, 0.5)', // cyan-500/50
                  color: '#cffafe', // cyan-100
                  fontFamily: 'monospace',
                  fontSize: '13px',
                  height: '48px',
                  borderRadius: '4px 4px 0 0',
                  letterSpacing: '0.02em',
                  textAlign: 'center',
                  paddingLeft: '40px',
                  paddingRight: '90px'
                }}
                className="login-input"
              />

              <Button
                type="text"
                onClick={handleGenerateRandom}
                style={{
                  position: 'absolute',
                  right: '8px',
                  zIndex: 1,
                  color: '#22d3ee', // cyan-400
                  fontSize: '12px',
                  fontWeight: 600,
                  padding: '4px 8px',
                  height: 'auto',
                  border: '1px solid rgba(34, 211, 238, 0.3)',
                  background: 'rgba(6, 182, 212, 0.1)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(6, 182, 212, 0.2)';
                  e.currentTarget.style.boxShadow = '0 0 8px rgba(34, 211, 238, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(6, 182, 212, 0.1)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                GENERATE
              </Button>
            </div>
            <div style={{ marginTop: '8px', textAlign: 'center' }}>
              <Text style={{ color: '#0e7490', fontSize: '10px', letterSpacing: '0.05em' }}>
                INPUT OR AUTO-GENERATE 32-BIT SESSION KEY
              </Text>
            </div>
          </Form.Item>

          <Form.Item style={{ marginBottom: '0' }}>
            <Button
              htmlType="submit"
              loading={state.loading}
              block
              style={{ 
                height: '48px', 
                fontSize: '14px', 
                fontWeight: 700,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                background: 'rgba(8, 145, 178, 0.2)', // cyan-600/20
                border: '1px solid #22d3ee', // cyan-400
                color: '#67e8f9', // cyan-300
                boxShadow: '0 0 10px rgba(34, 211, 238, 0.2)', // cyan glow
                borderRadius: '4px',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(8, 145, 178, 0.4)';
                e.currentTarget.style.boxShadow = '0 0 20px rgba(34, 211, 238, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(8, 145, 178, 0.2)';
                e.currentTarget.style.boxShadow = '0 0 10px rgba(34, 211, 238, 0.2)';
              }}
            >
              Initialize Session
            </Button>
          </Form.Item>
        </Form>
        
        {/* Decorative Divider */}
        <div style={{ 
          marginTop: '40px', 
          height: '1px', 
          background: 'linear-gradient(to right, transparent, rgba(6, 182, 212, 0.5), transparent)',
          width: '100%'
        }} />
        
        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Text style={{ color: '#0e7490', fontSize: '10px', fontFamily: 'monospace' }}>
            SECURE CONNECTION :: V1.0.0
          </Text>
        </div>
      </Card>
    </div>
  );
}

export default Login;