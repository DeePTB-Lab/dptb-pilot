import { describe, expect, it } from 'vitest';
import { isCopyShortcut, selectedText } from './clipboard';

describe('webview clipboard support', () => {
  it('reads the selected portion of an input control', () => {
    const input = {
      tagName: 'TEXTAREA',
      value: 'before selected after',
      selectionStart: 7,
      selectionEnd: 15,
    } as unknown as Element;

    expect(selectedText(input, null)).toBe('selected');
  });

  it('reads the browser selection for rendered messages', () => {
    const selection = { toString: () => 'assistant response' } as Selection;

    expect(selectedText(null, selection)).toBe('assistant response');
  });

  it('recognizes Windows, Linux, and macOS copy shortcuts', () => {
    expect(isCopyShortcut({ key: 'c', ctrlKey: true, metaKey: false, altKey: false })).toBe(true);
    expect(isCopyShortcut({ key: 'C', ctrlKey: false, metaKey: true, altKey: false })).toBe(true);
    expect(isCopyShortcut({ key: 'c', ctrlKey: true, metaKey: false, altKey: true })).toBe(false);
  });
});
