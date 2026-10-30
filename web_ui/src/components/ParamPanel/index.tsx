import { useState, useEffect } from 'react';
import {
  Select,
  Radio,
  Button,
  Input,
  Form,
  Space,
  Divider,
  Typography,
  InputNumber,
  Switch,
  message,
  Alert,
  Tooltip,
  Modal,
  Card,
  Popconfirm
} from 'antd';
import {
  CheckOutlined,
  InfoCircleOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { ToolSchema, PropertySchema, ExecutionMode, ModifyMode, RemoteMachine, RemoteMachineType, BohriumConfig, SlurmConfig } from '../../types';
import { translations } from '../../utils/i18n';

const { Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;

function ParamPanel() {
  const { state, actions } = useApp();
  const t = translations[state.language];
  const [form] = Form.useForm();
  const [currentSchema, setCurrentSchema] = useState<ToolSchema | null>(null);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('Local');
  const [modifyMode, setModifyMode] = useState<ModifyMode>('individual');
  const [jsonText, setJsonText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // 远程机器列表
  const [remoteMachines, setRemoteMachines] = useState<RemoteMachine[]>(() => {
    const saved = localStorage.getItem('remoteMachines');
    return saved ? JSON.parse(saved) : [];
  });
  const [selectedMachineId, setSelectedMachineId] = useState<string>(() => {
    const saved = localStorage.getItem('selectedMachineId');
    return saved || '';
  });

  // 添加机器弹窗状态
  const [addModalVisible, setAddModalVisible] = useState(false);
  const [addMachineForm] = Form.useForm();
  const [addMachineType, setAddMachineType] = useState<RemoteMachineType>('Bohrium');

  // 编辑机器弹窗状态
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingMachine, setEditingMachine] = useState<RemoteMachine | null>(null);
  const [editMachineForm] = Form.useForm();
  const [editMachineType, setEditMachineType] = useState<RemoteMachineType>('Bohrium');

  const [lastSchemaHash, setLastSchemaHash] = useState<string>('');

  // 持久化远程机器列表到 localStorage
  useEffect(() => {
    localStorage.setItem('remoteMachines', JSON.stringify(remoteMachines));
  }, [remoteMachines]);

  // 持久化选中的机器ID到 localStorage
  useEffect(() => {
    localStorage.setItem('selectedMachineId', selectedMachineId);
  }, [selectedMachineId]);

  // 从localStorage加载executionMode
  useEffect(() => {
    const savedMode = localStorage.getItem('executionMode') as ExecutionMode;
    if (savedMode && (savedMode === 'Local' || savedMode === 'Remote')) {
      setExecutionMode(savedMode);
    }
  }, []);

  // 定时检查schema更新
  useEffect(() => {
    const checkForSchema = async () => {
      if (!state.userId) return;

      try {
        const schema = await actions.getCurrentSchema();
        if (schema && Object.keys(schema).length > 0) {
          const schemaHash = JSON.stringify(schema);

          if (schemaHash !== lastSchemaHash) {
            setCurrentSchema(schema);
            setLastSchemaHash(schemaHash);

            if (modifyMode === 'json') {
              setJsonText(JSON.stringify(schema, null, 2));
            } else {
              const formData: any = {};
              if (schema.input_schema?.properties) {
                Object.entries(schema.input_schema.properties).forEach(([key, prop]: [string, any]) => {
                  formData[key] = prop.agent_input || prop.default || '';
                });
              }
              form.setFieldsValue(formData);
            }
          }
        } else {
          if (lastSchemaHash !== '') {
            setCurrentSchema(null);
            setLastSchemaHash('');
          }
        }
      } catch (error) {
        console.error('获取参数schema失败:', error);
      }
    };

    const interval = setInterval(checkForSchema, 1000);
    return () => clearInterval(interval);
  }, [state.userId, actions, form, modifyMode, lastSchemaHash]);

  const handleExecutionModeChange = (value: ExecutionMode) => {
    setExecutionMode(value);
    localStorage.setItem('executionMode', value);
    console.log('执行模式变更为:', value);
  };

  const handleModifyModeChange = (e: any) => {
    setModifyMode(e.target.value);
  };

  // 获取选中的机器
  const selectedMachine = remoteMachines.find(m => m.id === selectedMachineId);

  // 添加新机器
  const handleAddMachine = () => {
    addMachineForm.validateFields().then(values => {
      const newMachine: RemoteMachine = {
        id: Date.now().toString(),
        name: values.name,
        type: addMachineType,
        config: addMachineType === 'Bohrium' ? {
          username: values.username,
          password: values.password,
          project_id: values.project_id,
          image_name: values.image_name || '',
          scass_type: values.scass_type || ''
        } as BohriumConfig : {
          remote_root: values.remote_root,
          hostname: values.hostname,
          username: values.username,
          key_filename: values.key_filename,
          number_node: values.number_node,
          gpu_per_node: values.gpu_per_node || '',
          cpu_per_node: values.cpu_per_node || '',
          queue_name: values.queue_name,
          custom_flags: values.custom_flags || ''
        } as SlurmConfig
      };

      setRemoteMachines(prev => [...prev, newMachine]);
      if (remoteMachines.length === 0) {
        setSelectedMachineId(newMachine.id);
      }
      addMachineForm.resetFields();
      setAddModalVisible(false);
      message.success('机器添加成功');
    });
  };

  // 删除机器
  const handleDeleteMachine = (id: string) => {
    setRemoteMachines(prev => prev.filter(m => m.id !== id));
    if (selectedMachineId === id) {
      setSelectedMachineId(remoteMachines[0]?.id || '');
    }
    message.success('机器已删除');
  };

  // 打开编辑机器弹窗
  const handleEditMachine = (machine: RemoteMachine) => {
    setEditingMachine(machine);
    setEditMachineType(machine.type);
    editMachineForm.setFieldsValue({
      name: machine.name,
      ...(machine.type === 'Bohrium' ? {
        username: (machine.config as BohriumConfig).username,
        password: (machine.config as BohriumConfig).password,
        project_id: (machine.config as BohriumConfig).project_id,
        image_name: (machine.config as BohriumConfig).image_name || '',
        scass_type: (machine.config as BohriumConfig).scass_type || ''
      } : {
        remote_root: (machine.config as SlurmConfig).remote_root,
        hostname: (machine.config as SlurmConfig).hostname,
        username: (machine.config as SlurmConfig).username,
        key_filename: (machine.config as SlurmConfig).key_filename,
        number_node: (machine.config as SlurmConfig).number_node,
        gpu_per_node: (machine.config as SlurmConfig).gpu_per_node || '',
        cpu_per_node: (machine.config as SlurmConfig).cpu_per_node || '',
        queue_name: (machine.config as SlurmConfig).queue_name,
        custom_flags: (machine.config as SlurmConfig).custom_flags || ''
      })
    });
    setEditModalVisible(true);
  };

  // 保存编辑后的机器
  const handleSaveEditMachine = () => {
    editMachineForm.validateFields().then(values => {
      if (!editingMachine) return;

      const updatedMachine: RemoteMachine = {
        id: editingMachine.id,
        name: values.name,
        type: editMachineType,
        config: editMachineType === 'Bohrium' ? {
          username: values.username,
          password: values.password,
          project_id: values.project_id,
          image_name: values.image_name || '',
          scass_type: values.scass_type || ''
        } as BohriumConfig : {
          remote_root: values.remote_root,
          hostname: values.hostname,
          username: values.username,
          key_filename: values.key_filename,
          number_node: values.number_node,
          gpu_per_node: values.gpu_per_node || '',
          cpu_per_node: values.cpu_per_node || '',
          queue_name: values.queue_name,
          custom_flags: values.custom_flags || ''
        } as SlurmConfig
      };

      setRemoteMachines(prev => prev.map(m => m.id === editingMachine.id ? updatedMachine : m));
      setEditModalVisible(false);
      setEditingMachine(null);
      editMachineForm.resetFields();
      message.success('机器配置已更新');
    });
  };

  // 生成 Executor 和 Storage 配置
  const generateExecutorAndStorage = (key: string, prop: any) => {
    if (executionMode !== 'Remote' || !selectedMachine) {
      return null;
    }

    const config = selectedMachine.config;

    if (key === 'Executor') {
      if (selectedMachine.type === 'Bohrium') {
        const bohrConfig = config as BohriumConfig;
        const executorConfig = {
          ...prop,
          user_input: {
            type: 'dispatcher',
            machine: {
              batch_type: 'Bohrium',
              context_type: 'Bohrium',
              remote_profile: {
                email: bohrConfig.username,
                password: bohrConfig.password,
                program_id: parseInt(bohrConfig.project_id),
                input_data: {
                  image_name: bohrConfig.image_name || 'registry.dp.tech/dptech/dp/native/prod-19853/dpa-mcp:0.0.0',
                  job_type: 'container',
                  platform: 'ali',
                  scass_type: bohrConfig.scass_type || '1 * NVIDIA V100_32g'
                }
              }
            }
          }
        };
        console.log('='.repeat(80));
        console.log('[Remote Execution] 生成 Executor 配置:');
        console.log('  机器名称:', selectedMachine.name);
        console.log('  机器类型:', selectedMachine.type);
        console.log('  Executor 配置:', JSON.stringify(executorConfig.user_input, null, 2));
        console.log('='.repeat(80));
        return executorConfig;
      } else {
        const slurmConfig = config as SlurmConfig;
        const executorConfig = {
          ...prop,
          user_input: {
            type: 'dispatcher',
            machine: {
              batch_type: 'Slurm',
              context_type: 'SSHContext',
              local_root: './',
              remote_root: slurmConfig.remote_root,
              remote_profile: {
                hostname: slurmConfig.hostname,
                username: slurmConfig.username,
                timeout: 600,
                port: 22,
                key_filename: slurmConfig.key_filename
              }
            },
            resource: {
              number_node: parseInt(slurmConfig.number_node),
              gpu_per_node: slurmConfig.gpu_per_node ? parseInt(slurmConfig.gpu_per_node) : 0,
              cpu_per_node: slurmConfig.cpu_per_node ? parseInt(slurmConfig.cpu_per_node) : 1,
              queue_name: slurmConfig.queue_name,
              custom_flags: [slurmConfig.custom_flags, ''],
              source_list: [],
              module_list: []
            }
          }
        };
        console.log('='.repeat(80));
        console.log('[Remote Execution] 生成 Executor 配置:');
        console.log('  机器名称:', selectedMachine.name);
        console.log('  机器类型:', selectedMachine.type);
        console.log('  Executor 配置:', JSON.stringify(executorConfig.user_input, null, 2));
        console.log('='.repeat(80));
        return executorConfig;
      }
    } else if (key === 'Storage' && selectedMachine.type === 'Bohrium') {
      const bohrConfig = config as BohriumConfig;
      const storageConfig = {
        ...prop,
        user_input: {
          type: 'bohrium',
          username: bohrConfig.username,
          password: bohrConfig.password,
          program_id: parseInt(bohrConfig.project_id)
        }
      };
      console.log('='.repeat(80));
      console.log('[Remote Execution] 生成 Storage 配置:');
      console.log('  机器名称:', selectedMachine.name);
      console.log('  机器类型:', selectedMachine.type);
      console.log('  Storage 配置:', JSON.stringify(storageConfig.user_input, null, 2));
      console.log('='.repeat(80));
      return storageConfig;
    }

    return null;
  };

  const handleSubmitIndividual = async (values: any) => {
    if (!currentSchema || !state.userId) return;

    setSubmitting(true);
    try {
      const modifiedSchema = {
        ...currentSchema,
        input_schema: {
          properties: {}
        }
      };

      if (currentSchema.input_schema?.properties) {
        Object.entries(currentSchema.input_schema.properties).forEach(([key, prop]) => {
          const generatedProp = generateExecutorAndStorage(key, prop);
          if (generatedProp) {
            (modifiedSchema.input_schema.properties as any)[key] = generatedProp;
            return;
          }
          // 其他参数使用用户输入的值
          (modifiedSchema.input_schema.properties as any)[key] = {
            ...prop,
            user_input: values[key]
          };
        });
      }

      console.log('='.repeat(80));
      console.log('[ParamPanel] 提交修改后的参数:');
      console.log('  工具名称:', currentSchema.name);
      console.log('  执行模式:', executionMode);
      console.log('  选中的机器:', selectedMachine?.name, '(', selectedMachine?.type, ')');
      console.log('  修改后的Schema:', JSON.stringify(modifiedSchema, null, 2));
      console.log('='.repeat(80));

      await actions.modifyParameters({
        session_id: state.userId,
        modified_schema: modifiedSchema,
        execution_mode: executionMode,
        selected_machine_id: selectedMachineId,
        remote_machine: selectedMachine
      });
      message.success('参数已提交');
      setCurrentSchema(null);
      setLastSchemaHash('');
      form.resetFields();
    } catch (error) {
      message.error('参数提交失败');
      console.error('Submit error:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitJson = async () => {
    if (!state.userId) return;

    try {
      const modifiedSchema = JSON.parse(jsonText);
      setSubmitting(true);
      await actions.modifyParameters(modifiedSchema);
      message.success('参数已提交');
      setCurrentSchema(null);
      setLastSchemaHash('');
      setJsonText('');
    } catch (error) {
      if (error instanceof SyntaxError) {
        message.error('JSON格式错误');
      } else {
        message.error('参数提交失败');
        console.error('Submit error:', error);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const renderFormInput = (_key: string, prop: PropertySchema) => {
    const commonProps = {
      style: { width: '100%' }
    };

    switch (prop.type) {
      case 'string':
        return (
          <Input
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || ''}
            placeholder={prop.default}
          />
        );
      case 'number':
      case 'integer':
        return (
          <InputNumber
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || 0}
            precision={prop.type === 'integer' ? 0 : 2}
            style={{ width: '100%' }}
          />
        );
      case 'boolean':
        return (
          <Switch
            defaultChecked={prop.agent_input || prop.default || false}
          />
        );
      case 'array':
      case 'object':
        return (
          <TextArea
            {...commonProps}
            defaultValue={JSON.stringify(prop.agent_input || prop.default || {}, null, 2)}
            rows={3}
            placeholder="请输入JSON格式数据"
          />
        );
      default:
        return (
          <Input
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || ''}
          />
        );
    }
  };

  // 渲染 Bohrium 配置表单
  const renderBohriumForm = () => (
    <>
      <Form.Item
        name="username"
        label={<span>{t.bohrUsername} (username)</span>}
        rules={[{ required: true, message: `${t.bohrUsername} ${t.required}` }]}
      >
        <Input placeholder="username" />
      </Form.Item>
      <Form.Item
        name="password"
        label={<span>{t.bohrPassword} (password)</span>}
        rules={[{ required: true, message: `${t.bohrPassword} ${t.required}` }]}
      >
        <Input.Password placeholder="password" />
      </Form.Item>
      <Form.Item
        name="project_id"
        label={<span>{t.bohrProjectId} (project_id)</span>}
        rules={[{ required: true, message: `${t.bohrProjectId} ${t.required}` }]}
      >
        <Input placeholder="project_id" />
      </Form.Item>
      <Form.Item
        name="image_name"
        label={<span>{t.bohrImageName} (image_name)</span>}
      >
        <Input placeholder={t.optional} />
      </Form.Item>
      <Form.Item
        name="scass_type"
        label={<span>{t.bohrScassType} (scass_type)</span>}
      >
        <Input placeholder={t.optional} />
      </Form.Item>
    </>
  );

  // 渲染 Slurm 配置表单
  const renderSlurmForm = () => (
    <>
      <Form.Item
        name="remote_root"
        label={<span>{t.slurmRemoteRoot} (remote_root)</span>}
        rules={[{ required: true, message: `${t.slurmRemoteRoot} ${t.required}` }]}
      >
        <Input placeholder="/path/to/remote/root" />
      </Form.Item>
      <Form.Item
        name="hostname"
        label={<span>{t.slurmHostname} (hostname)</span>}
        rules={[{ required: true, message: `${t.slurmHostname} ${t.required}` }]}
      >
        <Input placeholder="hostname" />
      </Form.Item>
      <Form.Item
        name="username"
        label={<span>{t.slurmUsername} (username)</span>}
        rules={[{ required: true, message: `${t.slurmUsername} ${t.required}` }]}
      >
        <Input placeholder="username" />
      </Form.Item>
      <Form.Item
        name="key_filename"
        label={<span>{t.slurmKeyFilename} (key_filename)</span>}
        rules={[{ required: true, message: `${t.slurmKeyFilename} ${t.required}` }]}
      >
        <Input placeholder="/path/to/key/file" />
      </Form.Item>
      <Form.Item
        name="number_node"
        label={<span>{t.slurmNumberNode} (number_node)</span>}
        rules={[{ required: true, message: `${t.slurmNumberNode} ${t.required}` }]}
      >
        <Input placeholder="1" />
      </Form.Item>
      <Form.Item
        name="gpu_per_node"
        label={<span>{t.slurmGpuPerNode} (gpu_per_node)</span>}
      >
        <Input placeholder={t.optional} />
      </Form.Item>
      <Form.Item
        name="cpu_per_node"
        label={<span>{t.slurmCpuPerNode} (cpu_per_node)</span>}
      >
        <Input placeholder={t.optional} />
      </Form.Item>
      <Form.Item
        name="queue_name"
        label={<span>{t.slurmQueueName} (queue_name)</span>}
        rules={[{ required: true, message: `${t.slurmQueueName} ${t.required}` }]}
      >
        <Input placeholder="queue_name" />
      </Form.Item>
      <Form.Item
        name="custom_flags"
        label={<span>{t.slurmCustomFlags} (custom_flags)</span>}
      >
        <Input placeholder={t.optional} />
      </Form.Item>
    </>
  );

  return (
    <div>
      {/* Execution Mode Section */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {t.executionMode}
          </Text>
        </div>
        <Select
          value={executionMode}
          onChange={handleExecutionModeChange}
          style={{ width: '100%' }}
          className="glass-select-dense"
          dropdownStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid rgba(255,255,255,0.1)'
          }}
        >
          <Option value="Local" style={{ color: '#e2e8f0' }}>{t.localExecution}</Option>
          <Option value="Remote" style={{ color: '#e2e8f0' }}>{t.remoteExecution}</Option>
        </Select>
      </div>

      {/* Remote Machines (Conditional) */}
      {executionMode === 'Remote' && (
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {t.remoteMachines}
            </Text>
            <Button
              type="text"
              icon={<PlusOutlined />}
              size="small"
              onClick={() => {
                addMachineForm.resetFields();
                setAddModalVisible(true);
              }}
              style={{ color: '#94a3b8' }}
            >
              {t.addMachine}
            </Button>
          </div>

          {remoteMachines.length === 0 ? (
            <Alert
              message="暂无远程机器"
              description="请添加远程机器以使用远程执行模式"
              type="info"
              showIcon
              style={{ fontSize: '12px' }}
            />
          ) : (
            <Select
              value={selectedMachineId}
              onChange={setSelectedMachineId}
              style={{ width: '100%' }}
              placeholder={t.selectMachine}
              className="glass-select-dense"
              dropdownStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              {remoteMachines.map(machine => (
                <Option key={machine.id} value={machine.id} style={{ color: '#e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{machine.name}</span>
                    <span style={{ fontSize: '10px', color: '#94a3b8', marginLeft: '8px' }}>{machine.type}</span>
                  </div>
                </Option>
              ))}
            </Select>
          )}

          {/* 机器列表显示 */}
          {remoteMachines.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              {remoteMachines.map(machine => (
                <Card
                  key={machine.id}
                  size="small"
                  style={{
                    marginBottom: '8px',
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: selectedMachineId === machine.id ? '1px solid #3b82f6' : '1px solid rgba(51, 65, 85, 0.5)'
                  }}
                  bodyStyle={{ padding: '8px 12px' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Text strong style={{ color: '#e2e8f0', fontSize: '12px' }}>{machine.name}</Text>
                      <Text type="secondary" style={{ fontSize: '11px', marginLeft: '8px' }}>{machine.type}</Text>
                    </div>
                    <Space size="small">
                      <Button
                        type="text"
                        icon={<EditOutlined />}
                        size="small"
                        onClick={() => handleEditMachine(machine)}
                        style={{ color: '#3b82f6' }}
                      />
                      <Popconfirm
                        title={t.confirmDelete}
                        onConfirm={() => handleDeleteMachine(machine.id)}
                        okText="确定"
                        cancelText="取消"
                      >
                        <Button
                          type="text"
                          icon={<DeleteOutlined />}
                          size="small"
                          danger
                          style={{ color: '#ef4444' }}
                        />
                      </Popconfirm>
                    </Space>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Agent Status Bar */}
      <div style={{
        backgroundColor: 'rgba(30, 41, 59, 0.5)',
        border: '1px solid rgba(51, 65, 85, 0.5)',
        borderRadius: '8px',
        padding: '10px 12px',
        marginBottom: '24px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: (() => {
            if (state.responding) return '#eab308';
            if (currentSchema && Object.keys(currentSchema).length > 0) return '#f97316';
            return '#22c55e';
          })(),
          boxShadow: (() => {
            if (state.responding) return '0 0 8px rgba(234, 179, 8, 0.4)';
            if (currentSchema && Object.keys(currentSchema).length > 0) return '0 0 8px rgba(249, 115, 22, 0.4)';
            return '0 0 8px rgba(34, 197, 94, 0.4)';
          })()
        }} />
        <Text style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 500 }}>
          {t.agentStatus}: <span style={{ color: '#cbd5e1' }}>
            {state.responding
              ? t.statusWorking
              : (currentSchema && Object.keys(currentSchema).length > 0)
                ? '等待用户传参'
                : t.statusIdle
            }
          </span>
        </Text>
      </div>

      <Divider style={{ borderColor: 'rgba(255,255,255,0.05)', margin: '24px 0' }} />

      {/* Parameter Modification Header & Toggle */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {t.paramModifyMode}
        </Text>
        <Radio.Group
          value={modifyMode}
          onChange={handleModifyModeChange}
          size="small"
          buttonStyle="solid"
          className="dense-toggle"
        >
          <Radio.Button value="individual" style={{ fontSize: '11px', padding: '0 8px' }}>{t.individual}</Radio.Button>
          <Radio.Button value="json" style={{ fontSize: '11px', padding: '0 8px' }}>{t.json}</Radio.Button>
        </Radio.Group>
      </div>

      {currentSchema ? (
        <>
          {state.pendingToolResponse && (
            <Alert
              message="Agent响应"
              description={
                <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '12px' }}>
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {state.pendingToolResponse}
                  </pre>
                </div>
              }
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
          )}
          <div className="glass-card" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <Text strong>工具: {currentSchema.name}</Text>
              <Tooltip title={currentSchema.description}>
                <InfoCircleOutlined />
              </Tooltip>
            </div>
            {modifyMode === 'individual' ? (
              <>
                {/* 在参数确认时选择远程机器 */}
                {executionMode === 'Remote' && (
                  <div style={{ marginBottom: '16px', padding: '12px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '6px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    <Text style={{ fontSize: '12px', color: '#60a5fa', marginBottom: '8px', display: 'block' }}>{t.selectMachine}:</Text>
                    <Select
                      value={selectedMachineId}
                      onChange={setSelectedMachineId}
                      style={{ width: '100%' }}
                      placeholder={t.selectMachine}
                    >
                      {remoteMachines.map(machine => (
                        <Option key={machine.id} value={machine.id}>
                          {machine.name} ({machine.type})
                        </Option>
                      ))}
                    </Select>
                  </div>
                )}
                <Form
                  form={form}
                  onFinish={handleSubmitIndividual}
                  layout="vertical"
                  size="small"
                >
                  {currentSchema.input_schema?.properties && Object.entries(currentSchema.input_schema.properties).map(([key, prop]) => {
                    // 隐藏 executor 和 storage 参数（支持大写和小写）
                    const lowerKey = key.toLowerCase();
                    if (lowerKey === 'executor' || lowerKey === 'storage') {
                      return null;
                    }

                    return (
                      <Form.Item
                        key={key}
                        label={
                          <Space>
                            <Text>{prop.title || key}</Text>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              ({prop.type})
                            </Text>
                          </Space>
                        }
                        extra={
                          <div>
                            {prop.default !== undefined && (
                              <Text type="secondary" style={{ fontSize: '11px' }}>
                                默认: {JSON.stringify(prop.default)}
                              </Text>
                            )}
                            {prop.agent_input !== undefined && (
                              <div>
                                <Text type="warning" style={{ fontSize: '11px' }}>
                                  Agent值: {JSON.stringify(prop.agent_input)}
                                </Text>
                              </div>
                            )}
                          </div>
                        }
                        name={key}
                      >
                        {renderFormInput(key, prop)}
                      </Form.Item>
                    );
                  })}

                  <Form.Item style={{ marginTop: '16px', marginBottom: '0' }}>
                    <Space style={{ width: '100%' }} size="small">
                      <Button
                        type="primary"
                        htmlType="submit"
                        icon={<CheckOutlined />}
                        loading={submitting}
                        style={{ flex: 1 }}
                      >
                        {t.submit}
                      </Button>
                      <Button
                        danger
                        onClick={() => {
                          actions.terminateExecution();
                        }}
                        disabled={!state.responding && (!currentSchema || Object.keys(currentSchema).length === 0)}
                        style={{ flex: 1 }}
                      >
                        终止执行
                      </Button>
                    </Space>
                  </Form.Item>
                </Form>
              </>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                <TextArea
                  value={jsonText}
                  onChange={(e) => setJsonText(e.target.value)}
                  placeholder="请输入JSON格式的参数"
                  rows={8}
                  style={{ fontFamily: 'monospace', fontSize: '12px' }}
                />
                <Button
                  type="primary"
                  icon={<CheckOutlined />}
                  onClick={handleSubmitJson}
                  loading={submitting}
                  block
                >
                  {t.submitJson}
                </Button>
              </Space>
            )}
          </div>
        </>
      ) : (
        <Alert
          message={t.waitingForTool}
          description={t.waitingForToolDesc}
          type="info"
          showIcon
          style={{ fontSize: '12px' }}
        />
      )}

      {/* 添加机器弹窗 */}
      <Modal
        title={t.addMachineTitle}
        open={addModalVisible}
        onCancel={() => setAddModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setAddModalVisible(false)}>
            {t.cancel}
          </Button>,
          <Button key="submit" type="primary" onClick={handleAddMachine}>
            {t.save}
          </Button>
        ]}
        width={600}
      >
        <Form form={addMachineForm} layout="vertical">
          <Form.Item
            name="name"
            label={t.machineName}
            rules={[{ required: true, message: `${t.machineName} ${t.required}` }]}
          >
            <Input placeholder="输入机器名称" />
          </Form.Item>
          <Form.Item
            label={t.machineType}
            required
          >
            <Select value={addMachineType} onChange={setAddMachineType}>
              <Option value="Bohrium">{t.bohriumType}</Option>
              <Option value="Slurm">{t.slurmType}</Option>
            </Select>
          </Form.Item>
          <Divider style={{ margin: '16px 0' }} />
          {addMachineType === 'Bohrium' ? renderBohriumForm() : renderSlurmForm()}
        </Form>
      </Modal>

      {/* 编辑机器弹窗 */}
      <Modal
        title={`编辑机器: ${editingMachine?.name || ''}`}
        open={editModalVisible}
        onCancel={() => {
          setEditModalVisible(false);
          setEditingMachine(null);
          editMachineForm.resetFields();
        }}
        footer={[
          <Button key="cancel" onClick={() => {
            setEditModalVisible(false);
            setEditingMachine(null);
            editMachineForm.resetFields();
          }}>
            {t.cancel}
          </Button>,
          <Button key="submit" type="primary" onClick={handleSaveEditMachine}>
            {t.save}
          </Button>
        ]}
        width={600}
      >
        <Form form={editMachineForm} layout="vertical">
          <Form.Item
            name="name"
            label={t.machineName}
            rules={[{ required: true, message: `${t.machineName} ${t.required}` }]}
          >
            <Input placeholder="输入机器名称" />
          </Form.Item>
          <Form.Item
            label={t.machineType}
            required
          >
            <Select value={editMachineType} onChange={setEditMachineType}>
              <Option value="Bohrium">{t.bohriumType}</Option>
              <Option value="Slurm">{t.slurmType}</Option>
            </Select>
          </Form.Item>
          <Divider style={{ margin: '16px 0' }} />
          {editMachineType === 'Bohrium' ? renderBohriumForm() : renderSlurmForm()}
        </Form>
      </Modal>
    </div>
  );
}

export default ParamPanel;
