import { describe, expect, it } from 'vitest';
import { apiUrl, isValidSessionId, normalizeBaseUrl, resolveServerResourceUrl, websocketUrl } from './url';

describe('URL helpers', () => {
  it('normalizes host-only server URLs', () => {
    expect(normalizeBaseUrl('localhost:8000/')).toBe('http://localhost:8000');
  });

  it('rejects empty and malformed server URLs with a readable error', () => {
    expect(() => normalizeBaseUrl('')).toThrow('Server URL is required');
    expect(() => normalizeBaseUrl('http://%')).toThrow('valid HTTP or HTTPS server URL');
  });

  it('builds API URLs with a single slash', () => {
    expect(apiUrl('http://localhost:8000/', '/api/config')).toBe('http://localhost:8000/api/config');
    expect(apiUrl('http://localhost:8000/api', '/api/config')).toBe('http://localhost:8000/api/config');
    expect(apiUrl('https://example.com/dptb', '/api/config')).toBe('https://example.com/dptb/api/config');
  });

  it('derives websocket URLs from HTTP URLs', () => {
    expect(websocketUrl('http://localhost:8000', 'abc123')).toBe('ws://localhost:8000/ws/chat/abc123');
    expect(websocketUrl('https://example.com/base', 'abc123')).toBe('wss://example.com/base/ws/chat/abc123');
    expect(websocketUrl('https://example.com/dptb/api', 'abc123')).toBe('wss://example.com/dptb/ws/chat/abc123');
    expect(websocketUrl('http://localhost:8000', 'abc123', 'chat one')).toBe('ws://localhost:8000/ws/chat/abc123?chat_id=chat%20one');
  });

  it('resolves relative Markdown resources against the backend URL', () => {
    expect(resolveServerResourceUrl('http://localhost:8000', '/api/download/session/band.png'))
      .toBe('http://localhost:8000/api/download/session/band.png');
    expect(resolveServerResourceUrl('https://example.com/dptb/api', 'api/download/session/band%20plot.png?raw=1'))
      .toBe('https://example.com/dptb/api/download/session/band%20plot.png?raw=1');
  });

  it('leaves absolute and embedded Markdown resources unchanged', () => {
    expect(resolveServerResourceUrl('http://localhost:8000', 'https://cdn.example.com/band.png'))
      .toBe('https://cdn.example.com/band.png');
    expect(resolveServerResourceUrl('http://localhost:8000', 'data:image/png;base64,abc'))
      .toBe('data:image/png;base64,abc');
    expect(resolveServerResourceUrl('http://localhost:8000', 'blob:vscode-webview://image'))
      .toBe('blob:vscode-webview://image');
  });

  it('validates 32-character session IDs', () => {
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef')).toBe(true);
    expect(isValidSessionId('too-short')).toBe(false);
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcde!')).toBe(false);
  });
});
