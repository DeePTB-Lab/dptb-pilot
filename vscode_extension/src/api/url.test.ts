import { describe, expect, it } from 'vitest';
import { apiUrl, isValidSessionId, normalizeBaseUrl, websocketUrl } from './url';

describe('URL helpers', () => {
  it('normalizes host-only server URLs', () => {
    expect(normalizeBaseUrl('localhost:8000/')).toBe('http://localhost:8000');
  });

  it('builds API URLs with a single slash', () => {
    expect(apiUrl('http://localhost:8000/', '/api/config')).toBe('http://localhost:8000/api/config');
  });

  it('derives websocket URLs from HTTP URLs', () => {
    expect(websocketUrl('http://localhost:8000', 'abc123')).toBe('ws://localhost:8000/ws/chat/abc123');
    expect(websocketUrl('https://example.com/base', 'abc123')).toBe('wss://example.com/ws/chat/abc123');
  });

  it('validates 32-character session IDs', () => {
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef')).toBe(true);
    expect(isValidSessionId('too-short')).toBe(false);
    expect(isValidSessionId('ABCDEFGHIJKLMNOPQRSTUVWXYZabcde!')).toBe(false);
  });
});
