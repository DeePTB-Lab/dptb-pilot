export function normalizeBaseUrl(input: string): string {
  const trimmed = input.trim().replace(/\/+$/, '');
  if (!trimmed) {
    return 'http://localhost:8000';
  }

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  const url = new URL(withProtocol);
  return url.toString().replace(/\/+$/, '');
}

export function apiUrl(baseUrl: string, path: string): string {
  const normalized = normalizeBaseUrl(baseUrl);
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${normalized}${cleanPath}`;
}

export function websocketUrl(baseUrl: string, sessionId: string): string {
  const normalized = normalizeBaseUrl(baseUrl);
  const url = new URL(normalized);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = `/ws/chat/${encodeURIComponent(sessionId)}`;
  url.search = '';
  url.hash = '';
  return url.toString();
}

export function generateSessionId(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const cryptoApi = globalThis.crypto;
  const bytes = new Uint8Array(length);
  cryptoApi?.getRandomValues?.(bytes);

  let result = '';
  for (let i = 0; i < length; i += 1) {
    const value = bytes[i] || Math.floor(Math.random() * chars.length);
    result += chars[value % chars.length];
  }
  return result;
}

export function isValidSessionId(sessionId: string): boolean {
  return /^[a-zA-Z0-9]{32}$/.test(sessionId);
}
