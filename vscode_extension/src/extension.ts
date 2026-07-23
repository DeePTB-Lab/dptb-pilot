import * as path from 'node:path';
import * as vscode from 'vscode';
import { DEFAULT_SERVER_URL, normalizeBaseUrl } from './api/url';
import type { HostToWebviewMessage, VisualizationPayload, WebviewToHostMessage } from './shared/types';

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
        serverUrl: config.get<string>('serverUrl') || DEFAULT_SERVER_URL,
        sessionId: config.get<string>('sessionId') || '',
        extensionUri: this.context.extensionUri.toString(),
        matterViz: {
          commandId: config.get<string>('matterViz.commandId') || '',
          uriTemplate: config.get<string>('matterViz.uriTemplate') || '',
          autoOpen: config.get<boolean>('matterViz.autoOpen') || false
        }
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
      case 'openInMatterViz':
        await openVisualizationInMatterViz(message.payload.visualization);
        break;
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
      const current = config.get<string>('serverUrl') || DEFAULT_SERVER_URL;
      const serverUrl = await vscode.window.showInputBox({
        title: 'Connect DPTB Pilot Server',
        prompt: 'Enter the dptb-pilot FastAPI server URL',
        value: current,
        ignoreFocusOut: true,
        validateInput: (value) => {
          try {
            normalizeBaseUrl(value);
            return undefined;
          } catch (error) {
            return errorMessage(error);
          }
        }
      });
      if (!serverUrl) return;
      const normalizedUrl = normalizeBaseUrl(serverUrl);
      await config.update('serverUrl', normalizedUrl, vscode.ConfigurationTarget.Global);
      DptbPilotPanel.createOrShow(context).post({ type: 'serverUrlChanged', payload: { serverUrl: normalizedUrl } });
    }),
    vscode.commands.registerCommand('dptbPilot.newSession', () => {
      DptbPilotPanel.createOrShow(context).post({ type: 'openNewSession' });
    }),
    vscode.commands.registerCommand('dptbPilot.uploadCurrentFile', () => {
      const panel = DptbPilotPanel.createOrShow(context);
      void panel.uploadCurrentFile();
    }),
    vscode.commands.registerCommand('dptbPilot.openVisualizationInMatterViz', async (visualization?: VisualizationPayload) => {
      await openVisualizationInMatterViz(visualization);
    }),
    vscode.window.registerUriHandler({
      handleUri: async (uri: vscode.Uri) => {
        const params = new URLSearchParams(uri.query);
        const serverUrl = params.get('url');
        const sessionId = params.get('session');
        const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

        let normalizedServerUrl: string | null = null;
        if (serverUrl) {
          try {
            normalizedServerUrl = normalizeBaseUrl(serverUrl);
            await config.update('serverUrl', normalizedServerUrl, vscode.ConfigurationTarget.Global);
          } catch (error) {
            void vscode.window.showErrorMessage(`Invalid DPTB Pilot server URL: ${errorMessage(error)}`);
            return;
          }
        }
        if (sessionId) {
          await config.update('sessionId', sessionId, vscode.ConfigurationTarget.Global);
        }

        if (config.get<boolean>('autoOpenOnUrl') !== false) {
          const panel = DptbPilotPanel.createOrShow(context);
          if (normalizedServerUrl) panel.post({ type: 'serverUrlChanged', payload: { serverUrl: normalizedServerUrl } });
          if (sessionId) panel.post({ type: 'sessionChanged', payload: { sessionId } });
        }
      }
    })
  );
}

export function deactivate(): void {}

async function openVisualizationInMatterViz(visualization?: VisualizationPayload): Promise<void> {
  if (!visualization) {
    void vscode.window.showWarningMessage('No DPTB visualization payload is available to open in MatterViz.');
    return;
  }

  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  const commandId = config.get<string>('matterViz.commandId')?.trim();
  const uriTemplate = config.get<string>('matterViz.uriTemplate')?.trim();

  if (commandId) {
    try {
      await vscode.commands.executeCommand(commandId, visualization);
      return;
    } catch (error) {
      void vscode.window.showWarningMessage(`MatterViz command failed: ${errorMessage(error)}`);
    }
  }

  if (uriTemplate) {
    const payload = encodeURIComponent(JSON.stringify(visualization));
    const uri = vscode.Uri.parse(uriTemplate.replace('{payload}', payload));
    await vscode.env.openExternal(uri);
    return;
  }

  void vscode.window.showInformationMessage('Configure dptbPilot.matterViz.commandId or dptbPilot.matterViz.uriTemplate to enable MatterViz handoff.');
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let text = '';
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
