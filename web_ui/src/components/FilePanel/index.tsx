import { useState, useEffect } from 'react';
import {
  Button,
  List,
  Typography,
  Spin,
  Collapse,
  Modal,
  Dropdown,
  message,
  Space
} from 'antd';
import {
  DownloadOutlined,
  DeleteOutlined,
  MoreOutlined,
  CopyOutlined,
  FileOutlined,
  FileTextOutlined,
  FileImageOutlined,
  LinkOutlined,
  CodeOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { ArtifactInfo, ArtifactTask, FileInfo } from '../../types';
import type { MenuProps } from 'antd';
import { translations } from '../../utils/i18n';

const { Text } = Typography;
const { Panel } = Collapse;

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg', 'svg', 'bmp'].includes(ext || '')) {
    return <FileImageOutlined style={{ color: '#fa8c16' }} />;
  }
  if (['json', 'yaml', 'yml', 'xml', 'npy', 'npz'].includes(ext || '')) {
    return <CodeOutlined style={{ color: '#722ed1' }} />;
  }
  if (['py', 'js', 'ts', 'tsx', 'jsx', 'c', 'cpp', 'h'].includes(ext || '')) {
    return <CodeOutlined style={{ color: '#1890ff' }} />;
  }
  if (['xyz', 'cif', 'poscar', 'vasp', 'xsf'].includes(ext || '')) {
    return <ExperimentOutlined style={{ color: '#52c41a' }} />;
  }
  return <FileTextOutlined style={{ color: '#8c8c8c' }} />;
}

function FilePanel() {
  const { state, actions } = useApp();
  const t = translations[state.language];
  
  useEffect(() => {
    if (state.isAuthenticated && state.userId && !state.files.length) {
      actions.loadFiles();
    }
    if (state.isAuthenticated && state.userId && !state.artifactTasks.length) {
      actions.loadArtifacts();
    }
  }, [state.isAuthenticated, state.userId]);

  /* New Preview State */
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewFile, setPreviewFile] = useState<FileInfo | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const getDownloadUrl = (path: string) => {
    const encodedPath = path
      .replace(/\\/g, '/')
      .split('/')
      .filter(Boolean)
      .map(encodeURIComponent)
      .join('/');
    return `/api/download/${state.userId}/${encodedPath}`;
  };

  const handlePreview = async (file: FileInfo) => {
    setPreviewFile(file);
    setPreviewVisible(true);
    setPreviewContent(null);
    setPreviewLoading(true);

    const ext = file.name.split('.').pop()?.toLowerCase();
    
    // For images, we just use the URL directly in the img tag
    if (['png', 'jpg', 'jpeg', 'svg', 'bmp'].includes(ext || '')) {
      setPreviewLoading(false);
      return;
    }

    try {
      const downloadUrl = getDownloadUrl(file.name);
      const response = await fetch(downloadUrl);
      if (!response.ok) throw new Error('Failed to fetch file content');
      
      const text = await response.text();
      setPreviewContent(text);
    } catch (error) {
      message.error('Failed to load file preview');
      setPreviewContent('Error loading content');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleDownload = (file: FileInfo) => {
    const downloadUrl = getDownloadUrl(file.name);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyName = (name: string) => {
    navigator.clipboard.writeText(name);
    message.success('Copied to clipboard');
  };

  const handleArtifactDownload = (artifact: ArtifactInfo) => {
    if (!artifact.workspace_path || !artifact.exists) return;
    const link = document.createElement('a');
    link.href = getDownloadUrl(artifact.workspace_path);
    link.download = artifact.file_name || artifact.workspace_path.split('/').pop() || 'artifact';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDelete = (file: FileInfo) => {
    Modal.confirm({
      title: t.deleteConfirm,
      content: file.name,
      okText: t.delete,
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          await actions.deleteFile(file.name);
          message.success('File deleted');
        } catch (error) {
          message.error('Delete failed');
        }
      }
    });
  };

  const getFileGroups = () => {
    const groups = {
      structure: [] as FileInfo[],
      plot: [] as FileInfo[],
      data: [] as FileInfo[],
      other: [] as FileInfo[]
    };

    state.files.forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase() || '';
      if (['xyz', 'cif', 'poscar', 'vasp', 'xsf'].includes(ext)) {
        groups.structure.push(file);
      } else if (['png', 'jpg', 'jpeg', 'svg', 'pdf'].includes(ext)) {
        groups.plot.push(file);
      } else if (['json', 'csv', 'txt', 'log', 'out', 'npy', 'npz'].includes(ext)) {
        groups.data.push(file);
      } else {
        groups.other.push(file);
      }
    });

    return groups;
  };

  const groups = getFileGroups();

  const renderFileItem = (file: FileInfo) => {
    const items: MenuProps['items'] = [
      {
        key: 'download',
        label: t.download,
        icon: <DownloadOutlined />,
        onClick: () => handleDownload(file)
      },
      {
        key: 'copy',
        label: t.copyName,
        icon: <CopyOutlined />,
        onClick: () => handleCopyName(file.name)
      },
      {
        type: 'divider'
      },
      {
        key: 'delete',
        label: t.delete,
        icon: <DeleteOutlined />,
        danger: true,
        onClick: () => handleDelete(file)
      }
    ];

    return (
      <List.Item
        style={{
          padding: '8px 12px',
          marginBottom: '8px',
          backgroundColor: '#1e293b', // slate-800
          border: '1px solid #334155', // slate-700
          borderRadius: '8px',
          transition: 'all 0.2s',
          cursor: 'pointer'
        }}
        onClick={() => handlePreview(file)}
        className="file-item-hover"
      >
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          <div style={{ marginRight: '12px', fontSize: '18px' }}>
            {getFileIcon(file.name)}
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <Text 
              ellipsis={{ tooltip: file.name }} 
              style={{ color: '#e2e8f0', fontSize: '13px', display: 'block' }} // slate-200
            >
              {file.name}
            </Text>
            <Text type="secondary" style={{ fontSize: '11px', color: '#94a3b8' }}> {/* slate-400 */}
              {(file.size / 1024).toFixed(1)} KB
            </Text>
          </div>
          <Dropdown menu={{ items }} trigger={['click']} placement="bottomRight">
            <Button 
              type="text" 
              icon={<MoreOutlined />} 
              size="small"
              style={{ color: '#94a3b8' }}
              onClick={(e) => e.stopPropagation()}
            />
          </Dropdown>
        </div>
      </List.Item>
    );
  };

  const artifactLabels = state.language === 'zh' ? {
    title: '任务 Artifacts',
    sourcePath: '原始路径',
    workspacePath: '工作区路径',
    unavailable: '该 artifact 尚未落入工作区，暂不可下载',
    binary: '二进制内容，可下载后查看',
  } : {
    title: 'Task Artifacts',
    sourcePath: 'Source path',
    workspacePath: 'Workspace path',
    unavailable: 'This artifact is not staged in the workspace yet.',
    binary: 'Binary content; download to inspect.',
  };

  const renderArtifact = (artifact: ArtifactInfo) => (
    <div
      key={artifact.id}
      style={{
        padding: '12px',
        marginBottom: '10px',
        backgroundColor: '#1e293b',
        border: '1px solid #334155',
        borderRadius: '8px'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
        <LinkOutlined style={{ color: '#38bdf8', marginTop: '3px' }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <Text style={{ color: '#94a3b8', fontSize: '11px', display: 'block' }}>
            {artifactLabels.sourcePath}
          </Text>
          <Text
            copyable={{ text: artifact.source_path }}
            ellipsis={{ tooltip: artifact.source_path }}
            style={{ color: '#e2e8f0', fontSize: '12px', display: 'block', fontFamily: 'monospace' }}
          >
            {artifact.source_path}
          </Text>

          {artifact.workspace_path && (
            <>
              <Text style={{ color: '#94a3b8', fontSize: '11px', display: 'block', marginTop: '8px' }}>
                {artifactLabels.workspacePath}
              </Text>
              <Text
                copyable={{ text: artifact.workspace_path }}
                ellipsis={{ tooltip: artifact.workspace_path }}
                style={{ color: '#cbd5e1', fontSize: '12px', display: 'block', fontFamily: 'monospace' }}
              >
                {artifact.workspace_path}
              </Text>
            </>
          )}
        </div>
        <Button
          type="text"
          icon={<DownloadOutlined />}
          disabled={!artifact.exists || !artifact.workspace_path}
          onClick={() => handleArtifactDownload(artifact)}
          style={{ color: artifact.exists ? '#38bdf8' : '#64748b' }}
          title={t.download}
        />
      </div>

      {artifact.preview_kind === 'image' && artifact.workspace_path && artifact.exists && (
        <img
          src={getDownloadUrl(artifact.workspace_path)}
          alt={artifact.file_name || 'artifact'}
          style={{ maxWidth: '100%', maxHeight: '180px', display: 'block', margin: '12px auto 0', borderRadius: '6px' }}
        />
      )}
      {artifact.preview_kind === 'text' && artifact.content_preview && (
        <pre style={{
          margin: '12px 0 0',
          padding: '10px',
          maxHeight: '180px',
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          borderRadius: '6px',
          background: '#0f172a',
          color: '#cbd5e1',
          fontSize: '11px'
        }}>
          {artifact.content_preview}
        </pre>
      )}
      {artifact.preview_kind === 'binary' && (
        <Text style={{ color: '#64748b', fontSize: '11px', display: 'block', marginTop: '10px' }}>
          {artifactLabels.binary}{artifact.size ? ` · ${(artifact.size / 1024).toFixed(1)} KB` : ''}
        </Text>
      )}
      {artifact.preview_kind === 'unavailable' && (
        <Text style={{ color: '#f59e0b', fontSize: '11px', display: 'block', marginTop: '10px' }}>
          {artifactLabels.unavailable}
        </Text>
      )}
    </div>
  );

  const renderArtifactTask = (task: ArtifactTask) => (
    <Panel
      key={task.id}
      header={
        <div style={{ minWidth: 0 }}>
          <Text
            ellipsis={{ tooltip: task.task }}
            style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 500, display: 'block' }}
          >
            {task.task}
          </Text>
          <Text style={{ color: '#64748b', fontSize: '11px', display: 'block' }}>
            {new Date(task.completed_at).toLocaleString()} · {task.artifacts.length}
          </Text>
        </div>
      }
    >
      {task.artifacts.map(renderArtifact)}
    </Panel>
  );

  if (state.loading && !state.files.length && !state.artifactTasks.length) {
    return (
      <div style={{ textAlign: 'center', padding: '40px 0' }}>
        <Spin />
      </div>
    );
  }

  if (!state.files.length && !state.artifactTasks.length) {
    return (
      <div style={{ textAlign: 'center', padding: '40px 0', color: '#64748b' }}>
        <FileOutlined style={{ fontSize: '32px', marginBottom: '8px', opacity: 0.5 }} />
        <div>{t.emptyFiles}</div>
      </div>
    );
  }

  return (
    <div className="file-panel">
      <Collapse 
        ghost 
        defaultActiveKey={['artifacts', 'structure', 'plot']}
        expandIconPosition="end"
      >
        {state.artifactTasks.length > 0 && (
          <Panel
            header={
              <Space>
                <LinkOutlined style={{ color: '#38bdf8' }} />
                <Text style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 500 }}>
                  {artifactLabels.title} ({state.artifactTasks.reduce((count, task) => count + task.artifacts.length, 0)})
                </Text>
              </Space>
            }
            key="artifacts"
          >
            <Collapse ghost defaultActiveKey={[]} expandIconPosition="end">
              {state.artifactTasks.map(renderArtifactTask)}
            </Collapse>
          </Panel>
        )}

        {groups.structure.length > 0 && (
          <Panel 
            header={
              <Space>
                <ExperimentOutlined style={{ color: '#52c41a' }} />
                <Text style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 500 }}>
                  {t.structureFiles} ({groups.structure.length})
                </Text>
              </Space>
            } 
            key="structure"
          >
            <List
              dataSource={groups.structure}
              renderItem={renderFileItem}
              split={false}
            />
          </Panel>
        )}

        {groups.plot.length > 0 && (
          <Panel 
            header={
              <Space>
                <FileImageOutlined style={{ color: '#fa8c16' }} />
                <Text style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 500 }}>
                  {t.plotResults} ({groups.plot.length})
                </Text>
              </Space>
            } 
            key="plot"
          >
            <List
              dataSource={groups.plot}
              renderItem={renderFileItem}
              split={false}
            />
          </Panel>
        )}

        {groups.data.length > 0 && (
          <Panel 
            header={
              <Space>
                <CodeOutlined style={{ color: '#1890ff' }} />
                <Text style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 500 }}>
                  {t.dataFiles} ({groups.data.length})
                </Text>
              </Space>
            } 
            key="data"
          >
            <List
              dataSource={groups.data}
              renderItem={renderFileItem}
              split={false}
            />
          </Panel>
        )}

        {groups.other.length > 0 && (
          <Panel 
            header={
              <Space>
                <FileTextOutlined style={{ color: '#8c8c8c' }} />
                <Text style={{ color: '#e2e8f0', fontSize: '13px', fontWeight: 500 }}>
                  {t.otherFiles} ({groups.other.length})
                </Text>
              </Space>
            } 
            key="other"
          >
            <List
              dataSource={groups.other}
              renderItem={renderFileItem}
              split={false}
            />
          </Panel>
        )}
      </Collapse>


      <Modal
        title={previewFile?.name}
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={800}
        styles={{ body: { maxHeight: '70vh', overflow: 'auto' } }}
      >
        {previewLoading ? (
          <div style={{ textAlign: 'center', padding: '20px' }}><Spin /></div>
        ) : (
          <div>
            {previewFile && ['png', 'jpg', 'jpeg', 'svg', 'bmp'].includes(previewFile.name.split('.').pop()?.toLowerCase() || '') ? (
              <img 
                src={getDownloadUrl(previewFile.name)}
                alt={previewFile.name} 
                style={{ maxWidth: '100%', display: 'block', margin: '0 auto' }} 
              />
            ) : (
              <pre style={{ 
                backgroundColor: '#f5f5f5', 
                color: '#000000',
                padding: '12px', 
                borderRadius: '4px',
                overflow: 'auto',
                maxHeight: '60vh',
                fontFamily: 'monospace',
                fontSize: '12px'
              }}>
                {previewContent}
              </pre>
            )}
            <div style={{ marginTop: '16px', textAlign: 'right' }}>
              <Button 
                icon={<DownloadOutlined />} 
                onClick={() => previewFile && handleDownload(previewFile)}
              >
                {t.download}
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

export default FilePanel;
