import * as path from 'node:path';
import * as vscode from 'vscode';
import type { HostToWebviewMessage, WebviewToHostMessage } from './shared/types';

const CONFIG_SECTION = 'dptbPilot';
const VIEW_TYPE = 'dptbPilot.chat';

class DptbPilotPanel {
  private static currentPanel: DptbPilotPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly context: vscode.ExtensionContext;
  private disposables: vscode.Disposable[] = [];

  static createOrShow(context: vscode.ExtensionContext): DptbPilotPanel {
    const column = vscode.window.activeTextEditor?.viewColumn || vscode.ViewColumn.One;

    if (DptbPilotPanel.currentPanel) {
      DptbPilotPanel.currentPanel.panel.reveal(column);
      DptbPilotPanel.currentPanel.postBootstrap();
      return DptbPilotPanel.currentPanel;
    }

    const panel = vscode.window.createWebviewPanel(VIEW_TYPE, 'DPTB Pilot', column, {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')]
    });

    DptbPilotPanel.currentPanel = new DptbPilotPanel(panel, context);
    return DptbPilotPanel.currentPanel;
  }

  static active(): DptbPilotPanel | undefined {
    return DptbPilotPanel.currentPanel;
  }

  private constructor(panel: vscode.WebviewPanel, context: vscode.ExtensionContext) {
    this.panel = panel;
    this.context = context;
    this.panel.webview.html = this.getHtmlForWebview();

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage((message: WebviewToHostMessage) => this.handleMessage(message), null, this.disposables);
  }

  post(message: HostToWebviewMessage): void {
    void this.panel.webview.postMessage(message);
  }

  postBootstrap(): void {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
    this.post({
      type: 'bootstrap',
      payload: {
        serverUrl: config.get<string>('serverUrl') || 'http://localhost:8000',
        sessionId: config.get<string>('sessionId') || '',
        extensionUri: this.context.extensionUri.toString()
      }
    });
  }

  private async handleMessage(message: WebviewToHostMessage): Promise<void> {
    switch (message.type) {
      case 'ready':
        this.postBootstrap();
        break;
      case 'setServerUrl':
        await vscode.workspace.getConfiguration(CONFIG_SECTION).update('serverUrl', message.payload.serverUrl, vscode.ConfigurationTarget.Global);
        this.post({ type: 'serverUrlChanged', payload: { serverUrl: message.payload.serverUrl } });
        break;
      case 'setSessionId':
        await vscode.workspace.getConfiguration(CONFIG_SECTION).update('sessionId', message.payload.sessionId, vscode.ConfigurationTarget.Global);
        this.post({ type: 'sessionChanged', payload: { sessionId: message.payload.sessionId } });
        break;
      case 'pickFiles':
        await this.pickAndSendFiles();
        break;
      case 'uploadCurrentFile':
        await this.sendCurrentFile();
        break;
      case 'saveSecret':
        await this.context.secrets.store(message.payload.key, message.payload.value);
        break;
      case 'getSecret': {
        const value = await this.context.secrets.get(message.payload.key);
        this.panel.webview.postMessage({ type: 'secret', payload: { requestId: message.payload.requestId, value } });
        break;
      }
    }
  }

  async uploadCurrentFile(): Promise<void> {
    await this.sendCurrentFile();
  }

  private async pickAndSendFiles(): Promise<void> {
    const uris = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: false, canSelectMany: true });
    if (!uris?.length) return;

    for (const uri of uris) {
      await this.sendFile(uri);
    }
  }

  private async sendCurrentFile(): Promise<void> {
    const uri = vscode.window.activeTextEditor?.document.uri;
    if (!uri || uri.scheme !== 'file') {
      void vscode.window.showWarningMessage('No file-backed editor is active.');
      return;
    }
    await this.sendFile(uri);
  }

  private async sendFile(uri: vscode.Uri): Promise<void> {
    const bytes = await vscode.workspace.fs.readFile(uri);
    this.post({
      type: 'uploadFile',
      payload: {
        name: path.basename(uri.fsPath),
        bytes: Array.from(bytes)
      }
    });
  }

  private getHtmlForWebview(): string {
    const webview = this.panel.webview;
    const mediaUri = vscode.Uri.joinPath(this.context.extensionUri, 'media');
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, 'assets', 'index.js'));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, 'assets', 'index.css'));
    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https: http: data: blob:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; connect-src http: https: ws: wss:;" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${styleUri}" rel="stylesheet" />
  <title>DPTB Pilot</title>
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" type="module" src="${scriptUri}"></script>
</body>
</html>`;
  }

  private dispose(): void {
    DptbPilotPanel.currentPanel = undefined;
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
  }
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('dptbPilot.openChat', () => {
      DptbPilotPanel.createOrShow(context);
    }),
    vscode.commands.registerCommand('dptbPilot.connectServer', async () => {
      const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
      const current = config.get<string>('serverUrl') || 'http://localhost:8000';
      const serverUrl = await vscode.window.showInputBox({
        title: 'Connect DPTB Pilot Server',
        prompt: 'Enter the dptb-pilot FastAPI server URL',
        value: current,
        ignoreFocusOut: true
      });
      if (!serverUrl) return;
      await config.update('serverUrl', serverUrl, vscode.ConfigurationTarget.Global);
      DptbPilotPanel.createOrShow(context).post({ type: 'serverUrlChanged', payload: { serverUrl } });
    }),
    vscode.commands.registerCommand('dptbPilot.newSession', () => {
      DptbPilotPanel.createOrShow(context).post({ type: 'openNewSession' });
    }),
    vscode.commands.registerCommand('dptbPilot.uploadCurrentFile', () => {
      const panel = DptbPilotPanel.createOrShow(context);
      void panel.uploadCurrentFile();
    }),
    vscode.window.registerUriHandler({
      handleUri: async (uri: vscode.Uri) => {
        const params = new URLSearchParams(uri.query);
        const serverUrl = params.get('url');
        const sessionId = params.get('session');
        const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

        if (serverUrl) {
          await config.update('serverUrl', serverUrl, vscode.ConfigurationTarget.Global);
        }
        if (sessionId) {
          await config.update('sessionId', sessionId, vscode.ConfigurationTarget.Global);
        }

        if (config.get<boolean>('autoOpenOnUrl') !== false) {
          const panel = DptbPilotPanel.createOrShow(context);
          if (serverUrl) panel.post({ type: 'serverUrlChanged', payload: { serverUrl } });
          if (sessionId) panel.post({ type: 'sessionChanged', payload: { sessionId } });
        }
      }
    })
  );
}

export function deactivate(): void {}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let text = '';
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
