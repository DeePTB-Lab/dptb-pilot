import { useEffect, useState } from 'react';
import { Alert, Button, Divider, Space, Tag, Timeline, Typography, message } from 'antd';
import { AuditOutlined, CheckOutlined, CloseOutlined, DeploymentUnitOutlined } from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { WorkflowPlan } from '../../types';

const { Text, Title, Paragraph } = Typography;

function WorkflowPanel() {
  const { state, actions } = useApp();
  const [plan, setPlan] = useState<WorkflowPlan | null>(state.pendingWorkflowPlan);
  const [submitting, setSubmitting] = useState<'approved' | 'rejected' | 'pilot_review' | null>(null);

  useEffect(() => {
    setPlan(state.pendingWorkflowPlan);
  }, [state.pendingWorkflowPlan]);

  useEffect(() => {
    const loadPlan = async () => {
      const current = await actions.getCurrentWorkflowPlan();
      if (current) setPlan(current);
    };

    loadPlan();
    const interval = setInterval(loadPlan, 1500);
    return () => clearInterval(interval);
  }, [actions]);

  const submitDecision = async (decision: 'approved' | 'rejected' | 'pilot_review') => {
    setSubmitting(decision);
    try {
      await actions.decideWorkflowPlan(decision);
      if (decision !== 'pilot_review') {
        setPlan(null);
      }
    } catch (error) {
      message.error('Workflow review failed');
    } finally {
      setSubmitting(null);
    }
  };

  if (!plan) {
    return (
      <Alert
        message="No workflow waiting for review"
        description="Complex tasks will appear here as an auditable workflow before Pilot starts execution."
        type="info"
        showIcon
        style={{ fontSize: '12px' }}
      />
    );
  }

  return (
    <div>
      <div style={{
        backgroundColor: 'rgba(30, 41, 59, 0.5)',
        border: '1px solid rgba(51, 65, 85, 0.5)',
        borderRadius: '8px',
        padding: '14px',
        marginBottom: '16px'
      }}>
        <Space align="start" size="middle">
          <DeploymentUnitOutlined style={{ color: '#38bdf8', fontSize: '20px', marginTop: '4px' }} />
          <div>
            <Title level={5} style={{ color: '#e2e8f0', margin: 0 }}>
              {plan.title || 'Workflow plan'}
            </Title>
            <Paragraph style={{ color: '#94a3b8', fontSize: '12px', margin: '8px 0 0' }}>
              {plan.summary}
            </Paragraph>
          </div>
        </Space>
      </div>

      <div style={{
        display: 'flex',
        gap: '8px',
        overflowX: 'auto',
        paddingBottom: '12px',
        marginBottom: '12px'
      }}>
        {(plan.steps || []).map((step, index) => (
          <div key={step.id} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
            <div style={{
              minWidth: '112px',
              maxWidth: '132px',
              minHeight: '64px',
              padding: '8px',
              borderRadius: '8px',
              border: '1px solid rgba(56, 189, 248, 0.35)',
              background: 'rgba(14, 165, 233, 0.10)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center'
            }}>
              <Text style={{ color: '#38bdf8', fontSize: '11px', fontWeight: 700 }}>{step.id}</Text>
              <Text style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: 1.25 }}>
                {step.title}
              </Text>
            </div>
            {index < (plan.steps || []).length - 1 && (
              <div style={{ width: '24px', height: '1px', backgroundColor: 'rgba(56, 189, 248, 0.5)' }} />
            )}
          </div>
        ))}
      </div>

      <Timeline
        items={(plan.steps || []).map(step => ({
          color: '#38bdf8',
          children: (
            <div style={{ paddingBottom: '12px' }}>
              <Text strong style={{ color: '#e2e8f0', display: 'block' }}>
                {step.id}. {step.title}
              </Text>
              <Text style={{ color: '#94a3b8', fontSize: '12px', display: 'block', marginTop: '4px' }}>
                {step.purpose}
              </Text>
              {step.depends_on?.length > 0 && (
                <Text style={{ color: '#64748b', fontSize: '11px', display: 'block', marginTop: '6px' }}>
                  Depends on: {step.depends_on.join(', ')}
                </Text>
              )}
              <div style={{ marginTop: '8px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                {(step.tools || []).map(tool => (
                  <Tag key={tool} color="blue" style={{ margin: 0, fontSize: '11px' }}>
                    {tool}
                  </Tag>
                ))}
              </div>
            </div>
          )
        }))}
      />

      {(plan.risks?.length || plan.questions?.length) ? (
        <>
          <Divider style={{ borderColor: 'rgba(255,255,255,0.08)', margin: '12px 0' }} />
          {plan.risks?.length ? (
            <div style={{ marginBottom: '12px' }}>
              <Text style={{ color: '#fbbf24', fontSize: '12px', fontWeight: 600 }}>Checks</Text>
              {plan.risks.map((risk, index) => (
                <Text key={index} style={{ color: '#cbd5e1', fontSize: '12px', display: 'block', marginTop: '4px' }}>
                  {risk}
                </Text>
              ))}
            </div>
          ) : null}
          {plan.questions?.length ? (
            <div>
              <Text style={{ color: '#fbbf24', fontSize: '12px', fontWeight: 600 }}>Open Questions</Text>
              {plan.questions.map((question, index) => (
                <Text key={index} style={{ color: '#cbd5e1', fontSize: '12px', display: 'block', marginTop: '4px' }}>
                  {question}
                </Text>
              ))}
            </div>
          ) : null}
        </>
      ) : null}

      {plan.review ? (
        <>
          <Divider style={{ borderColor: 'rgba(255,255,255,0.08)', margin: '12px 0' }} />
          <div style={{
            border: '1px solid rgba(34, 197, 94, 0.28)',
            background: 'rgba(22, 163, 74, 0.08)',
            borderRadius: '8px',
            padding: '12px'
          }}>
            <Space align="center" style={{ marginBottom: '8px' }}>
              <AuditOutlined style={{ color: '#22c55e' }} />
              <Text style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 700 }}>Pilot Review</Text>
              <Tag
                color={plan.review.verdict === 'pass' ? 'green' : plan.review.verdict === 'block' ? 'red' : 'gold'}
                style={{ margin: 0, fontSize: '11px' }}
              >
                {plan.review.verdict}
              </Tag>
            </Space>
            <Paragraph style={{ color: '#cbd5e1', fontSize: '12px', marginBottom: '10px' }}>
              {plan.review.summary}
            </Paragraph>
            {[
              ['Scientific Logic', plan.review.scientific_logic],
              ['Execution Logic', plan.review.execution_logic],
              ['Blocking Issues', plan.review.blocking_issues],
              ['Warnings', plan.review.warnings],
              ['Recommendations', plan.review.recommendations],
            ].map(([title, items]) => (
              Array.isArray(items) && items.length ? (
                <div key={title as string} style={{ marginTop: '8px' }}>
                  <Text style={{ color: '#94a3b8', fontSize: '11px', fontWeight: 700 }}>
                    {title as string}
                  </Text>
                  {(items as string[]).map((item, index) => (
                    <Text key={index} style={{ color: '#cbd5e1', fontSize: '12px', display: 'block', marginTop: '3px' }}>
                      {item}
                    </Text>
                  ))}
                </div>
              ) : null
            ))}
          </div>
        </>
      ) : null}

      <Divider style={{ borderColor: 'rgba(255,255,255,0.08)', margin: '16px 0' }} />

      <Space direction="vertical" style={{ width: '100%' }} size="small">
        <Button
          icon={<AuditOutlined />}
          loading={submitting === 'pilot_review'}
          onClick={() => submitDecision('pilot_review')}
          style={{ width: '100%' }}
        >
          Pilot Review
        </Button>
        <Space style={{ width: '100%' }} size="small">
        <Button
          type="primary"
          icon={<CheckOutlined />}
          loading={submitting === 'approved'}
          disabled={submitting === 'pilot_review'}
          onClick={() => submitDecision('approved')}
          style={{ flex: 1 }}
        >
          Approve
        </Button>
        <Button
          danger
          icon={<CloseOutlined />}
          loading={submitting === 'rejected'}
          disabled={submitting === 'pilot_review'}
          onClick={() => submitDecision('rejected')}
          style={{ flex: 1 }}
        >
          Reject
        </Button>
        </Space>
      </Space>
    </div>
  );
}

export default WorkflowPanel;
