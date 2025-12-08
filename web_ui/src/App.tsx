
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider, App as AntdApp } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { AppProvider } from './contexts/AppContext';
import Login from './components/Login';
import Chat from './components/Chat';

function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <AntdApp>
        <AppProvider>
          <Router>
            <div className="App" style={{ height: '100vh' }}>
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route path="/chat" element={<Chat />} />
                <Route path="/" element={<Navigate to="/login" replace />} />
              </Routes>
            </div>
          </Router>
        </AppProvider>
      </AntdApp>
    </ConfigProvider>
  );
}

export default App;