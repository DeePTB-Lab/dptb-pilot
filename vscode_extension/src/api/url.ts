export const DEFAULT_SERVER_URL = 'http://localhost:50003';

export function normalizeBaseUrl(input: string): string {
  const trimmed = input.trim().replace(/\/+$/, '');
  if (!trimmed) {
    throw new Error('Server URL is required.');
  }

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  let url: URL;
  try {
    url = new URL(withProtocol);
  } catch {
    throw new Error(`Enter a valid HTTP or HTTPS server URL, for example ${DEFAULT_SERVER_URL}.`);
  }
  if (!['http:', 'https:'].includes(url.protocol) || !url.hostname) {
    throw new Error(`Enter a valid HTTP or HTTPS server URL, for example ${DEFAULT_SERVER_URL}.`);
  }
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/+$/, '');
}

export function apiUrl(baseUrl: string, path: string): string {
  const url = new URL(normalizeBaseUrl(baseUrl));
  const basePath = url.pathname.replace(/\/+$/, '');
  const [pathOnly, query = ''] = path.split('?', 2);
  const cleanPath = `/${pathOnly.replace(/^\/+/, '')}`;
  url.pathname = basePath.endsWith('/api') && cleanPath.startsWith('/api/')
    ? `${basePath}${cleanPath.slice(4)}`
    : `${basePath}${cleanPath}`;
  url.search = query;
  return url.toString();
}

export function resolveServerResourceUrl(baseUrl: string, resourceUrl: string): string {
  const trimmed = resourceUrl.trim();
  if (!trimmed || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(trimmed)) {
    return trimmed;
  }

  return apiUrl(baseUrl, trimmed);
}

export function websocketUrl(baseUrl: string, sessionId: string, chatId?: string): string {
  const normalized = normalizeBaseUrl(baseUrl);
  const url = new URL(normalized);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  const basePath = url.pathname.replace(/\/+$/, '').replace(/\/api$/, '');
  url.pathname = `${basePath}/ws/chat/${encodeURIComponent(sessionId)}`;
  url.search = chatId ? `?chat_id=${encodeURIComponent(chatId)}` : '';
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
