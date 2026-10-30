import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const css = readFileSync(fileURLToPath(new URL('./index.css', import.meta.url)), 'utf8');

function rule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = css.match(new RegExp(`${escaped}\\s*\\{([^}]+)\\}`));
  if (!match) throw new Error(`Missing CSS rule for ${selector}`);
  return match[1].replace(/\s+/g, ' ');
}

describe('chat layout scrolling', () => {
  it('keeps the app constrained to the webview viewport', () => {
    expect(rule('html, body, #root')).toContain('height: 100%');
    expect(rule('.app-shell')).toContain('height: 100%');
    expect(rule('.app-shell')).toContain('overflow: hidden');
  });

  it('gives long messages an independent vertical scroll container', () => {
    expect(rule('.chat-main')).toContain('min-height: 0');
    expect(rule('.chat-main')).toContain('overflow: hidden');
    expect(rule('.message-list')).toContain('min-height: 0');
    expect(rule('.message-list')).toContain('overflow-y: auto');
    expect(rule('.composer')).toContain('flex: 0 0 auto');
  });

  it('allows rendered chat content to be selected', () => {
    const selectionRule = rule('.message-content, .message-content *, .tool-result-body, .tool-result-body *, .modal, .modal *');
    expect(selectionRule).toContain('user-select: text');
    expect(selectionRule).toContain('-webkit-user-select: text');
  });
});
