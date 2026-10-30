type SelectionControl = Element & {
  value?: unknown;
  selectionStart?: number | null;
  selectionEnd?: number | null;
};

export function selectedText(activeElement: Element | null, pageSelection: Selection | null): string {
  const control = activeElement as SelectionControl | null;
  const tagName = control?.tagName?.toLowerCase();
  if ((tagName === 'input' || tagName === 'textarea') && typeof control?.value === 'string') {
    const start = control.selectionStart;
    const end = control.selectionEnd;
    if (typeof start === 'number' && typeof end === 'number' && end > start) {
      return control.value.slice(start, end);
    }
  }
  return pageSelection?.toString() || '';
}

export function isCopyShortcut(event: Pick<KeyboardEvent, 'altKey' | 'ctrlKey' | 'key' | 'metaKey'>): boolean {
  return !event.altKey && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'c';
}

export function installSelectionCopyBridge(writeClipboard: (text: string) => void): () => void {
  const readSelection = () => selectedText(document.activeElement, window.getSelection());

  const handleCopy = (event: ClipboardEvent) => {
    const text = readSelection();
    if (!text) return;

    if (event.clipboardData) {
      event.clipboardData.setData('text/plain', text);
      event.preventDefault();
    } else {
      writeClipboard(text);
    }
  };

  const handleKeyDown = (event: KeyboardEvent) => {
    if (!isCopyShortcut(event)) return;
    const text = readSelection();
    if (text) writeClipboard(text);
  };

  document.addEventListener('copy', handleCopy);
  window.addEventListener('keydown', handleKeyDown, true);
  return () => {
    document.removeEventListener('copy', handleCopy);
    window.removeEventListener('keydown', handleKeyDown, true);
  };
}
