import { describe, expect, it } from 'vitest';
import { apiUrl, isValidSessionId, normalizeBaseUrl, websocketUrl } from './url';

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
  });

  it('validates 32-character session IDs', () => {
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef')).toBe(true);
    expect(isValidSessionId('too-short')).toBe(false);
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcde!')).toBe(false);
  });
});
