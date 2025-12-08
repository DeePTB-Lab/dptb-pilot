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
  CodeOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { FileInfo } from '../../types';
import type { MenuProps } from 'antd';
import { translations } from '../../utils/i18n';

const { Text } = Typography;
const { Panel } = Collapse;

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg', 'svg', 'bmp'].includes(ext || '')) {
    return <FileImageOutlined style={{ color: '#fa8c16' }} />;
  }
  if (['json', 'yaml', 'yml', 'xml'].includes(ext || '')) {
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
  }, [state.isAuthenticated, state.userId]);

  /* New Preview State */
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewFile, setPreviewFile] = useState<FileInfo | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

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
      const downloadUrl = `/api/download/${state.userId}/${file.name}`;
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
    const downloadUrl = `/api/download/${state.userId}/${file.name}`;
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
      } else if (['json', 'csv', 'txt', 'log', 'out'].includes(ext)) {
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

  if (state.loading && !state.files.length) {
    return (
      <div style={{ textAlign: 'center', padding: '40px 0' }}>
        <Spin />
      </div>
    );
  }

  if (!state.files.length) {
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
        defaultActiveKey={['structure', 'plot']}
        expandIconPosition="end"
      >
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
                src={`/api/download/${state.userId}/${previewFile.name}`} 
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