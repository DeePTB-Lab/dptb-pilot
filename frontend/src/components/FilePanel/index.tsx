import { useState } from 'react';
import {
  Button,
  List,
  Typography,
  message,
  Tooltip,
  Popconfirm,
  Spin,
  Card
} from 'antd';
import {
  DownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  FileOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { FileInfo } from '../../types';

const { Title, Text } = Typography;

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function FilePanel() {
  const { state, actions } = useApp();
  // const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDownload = (file: FileInfo) => {
    if (!state.userId) return;

    const downloadUrl = `/api/download/${state.userId}/${file.name}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDelete = async (_file: FileInfo) => {
    // 注意：后端目前没有删除文件的API，这里只是界面实现
    message.info('删除功能尚未实现，请联系管理员');
  };

  const handleRefresh = async () => {
    if (!state.userId) return;

    try {
      await actions.loadFiles();
    } catch (error) {
      message.error('刷新文件列表失败');
      console.error('Refresh error:', error);
    }
  };



  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
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

      <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '12px' }}>
        工作目录: workspace/{state.userId}/files
      </Text>

      {state.loading ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin />
        </div>
      ) : (
        <Card
          title="文件列表"
          size="small"
          styles={{ body: { padding: '0' } }}
        >
          {state.files && state.files.length > 0 ? (
            <List
              size="small"
              dataSource={state.files}
              renderItem={(file: FileInfo) => (
                <List.Item
                  style={{ padding: '8px 12px', borderBottom: '1px solid #f0f0f0' }}
                  actions={[
                    <Tooltip title="下载" key="download">
                      <Button
                        type="text"
                        icon={<DownloadOutlined />}
                        onClick={() => handleDownload(file)}
                        size="small"
                      />
                    </Tooltip>,
                    <Popconfirm
                      title="确定要删除这个文件吗？"
                      onConfirm={() => handleDelete(file)}
                      key="delete"
                    >
                      <Tooltip title="删除">
                        <Button
                          type="text"
                          danger
                          icon={<DeleteOutlined />}
                          size="small"
                        />
                      </Tooltip>
                    </Popconfirm>
                  ]}
                >
                  <List.Item.Meta
                    avatar={<FileOutlined style={{ color: '#1677ff' }} />}
                    title={
                      <Text style={{ fontSize: '14px' }} title={file.name}>
                        {file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name}
                      </Text>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {formatFileSize(file.size)}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: '#8c8c8c' }}>
              <FileOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
              <div>暂无文件</div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

export default FilePanel;