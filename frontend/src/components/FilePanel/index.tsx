import { useState, useEffect } from 'react';
import {
  Button,
  List,
  Typography,
  message,
  Tooltip,
  Popconfirm,
  Spin,
  Card,
  Collapse,
  Badge,
  Modal,
  Dropdown
} from 'antd';
import {
  DownloadOutlined,
  DeleteOutlined,
  MoreOutlined,
  ReloadOutlined,
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

const { Title, Text } = Typography;
const { Panel } = Collapse;

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

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
  
  useEffect(() => {
    if (state.isAuthenticated && state.userId && !state.files.length) {
      actions.loadFiles();
    }
  }, [state.isAuthenticated, state.userId]);

  const handleRefresh = () => {
    actions.loadFiles();
  };

  const handleDownload = (file: FileInfo) => {
    // 构建下载链接
    // 注意：这里假设后端API提供了下载端点 /api/download/{session_id}/{filename}
    // 实际项目中应该使用 apiService.getFileDownloadUrl
    const downloadUrl = `/api/download/${state.userId}/${file.name}`;
    
    // 创建临时a标签触发下载
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDelete = async (file: FileInfo) => {
    try {
      await actions.deleteFile(file.name);
    } catch (error) {
      // Error is handled in AppContext
    }
  };

  // 对文件进行分类
  const groupedFiles = {
    structures: [] as FileInfo[],
    images: [] as FileInfo[],
    configs: [] as FileInfo[],
    others: [] as FileInfo[]
  };

  if (state.files) {
    state.files.forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase() || '';
      if (['xyz', 'cif', 'poscar', 'vasp', 'xsf'].includes(ext)) {
        groupedFiles.structures.push(file);
      } else if (['png', 'jpg', 'jpeg', 'svg', 'bmp'].includes(ext)) {
        groupedFiles.images.push(file);
      } else if (['json', 'yaml', 'yml', 'xml', 'toml', 'ini'].includes(ext)) {
        groupedFiles.configs.push(file);
      } else {
        groupedFiles.others.push(file);
      }
    });
  }

  const sortFiles = (files: FileInfo[]) => {
    return [...files].sort((a, b) => {
      // 首先按时间倒序
      if (b.updated_at !== a.updated_at) {
        return (b.updated_at || 0) - (a.updated_at || 0);
      }
      // 然后按名称排序
      return a.name.localeCompare(b.name);
    });
  };


  const getFileActions = (file: FileInfo): MenuProps['items'] => [
    {
      key: 'copy',
      label: '复制文件名',
      icon: <CopyOutlined />,
      onClick: () => {
        navigator.clipboard.writeText(file.name);
        message.success('文件名已复制');
      }
    },
    {
      key: 'download',
      label: '下载文件',
      icon: <DownloadOutlined />,
      onClick: () => handleDownload(file)
    },
    {
      key: 'delete',
      label: '删除文件',
      icon: <DeleteOutlined />,
      danger: true,
      onClick: () => {
        Modal.confirm({
          title: '确定要删除这个文件吗？',
          content: '删除后无法恢复',
          okText: '删除',
          okType: 'danger',
          onOk: () => handleDelete(file)
        });
      }
    }
  ];

  const renderFileList = (files: FileInfo[]) => (
    <List
      size="small"
      dataSource={sortFiles(files)}
      renderItem={(file: FileInfo) => (
        <List.Item
          style={{ padding: '8px 12px', borderBottom: '1px solid #f0f0f0' }}
          actions={[
            <Dropdown menu={{ items: getFileActions(file) }} trigger={['click']} placement="bottomRight">
              <Button type="text" size="small" icon={<MoreOutlined style={{ fontSize: '16px' }} />} />
            </Dropdown>
          ]}
        >
          <List.Item.Meta
            avatar={getFileIcon(file.name)}
            title={
              <Text 
                style={{ fontSize: '13px', wordBreak: 'break-all', userSelect: 'text' }}
                copyable={{ text: file.name, tooltips: ['复制文件名', '复制成功'] }}
              >
                {file.name}
              </Text>
            }
            description={
              <Text type="secondary" style={{ fontSize: '11px' }}>
                {formatFileSize(file.size)}
              </Text>
            }
          />
        </List.Item>
      )}
    />
  );

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', padding: '0 4px' }}>
        <Title level={5} style={{ margin: 0 }}>
          文件管理
        </Title>
        <Tooltip title="刷新文件列表">
          <Button
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
            size="small"
            loading={state.loading}
          />
        </Tooltip>
      </div>

      <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '12px', padding: '0 4px' }}>
        工作目录: workspace/{state.userId}/files
      </Text>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {state.loading && !state.files ? (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Spin />
          </div>
        ) : (state.files && state.files.length > 0) ? (
          <Collapse ghost size="small">
            {groupedFiles.structures.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>⚛️ 结构文件 ({groupedFiles.structures.length})</span>} key="structures">
                {renderFileList(groupedFiles.structures)}
              </Panel>
            )}
            {groupedFiles.images.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>🖼️ 图像结果 ({groupedFiles.images.length})</span>} key="images">
                {renderFileList(groupedFiles.images)}
              </Panel>
            )}
            {groupedFiles.configs.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>⚙️ 配置文件 ({groupedFiles.configs.length})</span>} key="configs">
                {renderFileList(groupedFiles.configs)}
              </Panel>
            )}
            {groupedFiles.others.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>📄 其他文件 ({groupedFiles.others.length})</span>} key="others">
                {renderFileList(groupedFiles.others)}
              </Panel>
            )}
          </Collapse>
        ) : (
          <div style={{ padding: '40px 20px', textAlign: 'center', color: '#8c8c8c' }}>
            <FileOutlined style={{ fontSize: '32px', marginBottom: '16px', color: '#d9d9d9' }} />
            <div>暂无文件</div>
            <div style={{ fontSize: '12px', marginTop: '8px' }}>上传或生成的文件将显示在这里</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FilePanel;