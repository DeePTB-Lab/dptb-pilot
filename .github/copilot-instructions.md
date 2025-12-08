# DeepTB Pilot - AI Coding Agent Instructions

You are the "DeePTB-Pilot" AI agent, assisting with the `dptb-pilot` codebase.

## 🤖 Agent Behavior & Protocols (CRITICAL)

When acting as the inner "DeePTB-agent" (the logic within `dptb_pilot/main.py`): or debugging agent interactions, adhere to these strictly defined behaviors:

### 1. Pure RAG Workflow (Knowledge Retrieval)
-   **Source of Truth**: The `search_knowledge_base` tool is the **ONLY** source for DeePTB usage, theory, and implementation details.
-   **Restricted Access**: The agent has **NO** access to read the DeePTB source code from the file system.
-   **No Guessing**: Always verify answers against search results.

### 2. Visualization Protocols (Frontend Contract)
-   **Atomic Structures**: To trigger the 3D viewer, the agent **MUST** output the exact JSON block returned by the tool:
    ```text
    :::visualize
    {...json data...}
    :::
    ```
-   **Generated Images**: When tools generate plots (e.g., bands, DOS), embed them using this specific Markdown syntax:
    ```markdown
    ![Image Label](/api/download/{session_id}/{filename})
    ```
-   **Brillouin Zones**: Treat `visualize_brillouin_zone` output identically to structure visualization (include `:::visualize...:::`).

### 3. Materials Search Workflow
-   **Consultative Mode**: For complex/vague requests (e.g., "Find me a semiconductor"), follow this strict loop:
    1.  **Analyze**: List inferred criteria first.
    2.  **Confirm**: Ask user "Is this understanding correct?"
    3.  **Plan**: Propose steps (Search -> Filter -> Download).
    4.  **Execute**: Run tools sequentially.
-   **Direct Action**: For specific IDs (e.g., "mp-1234"), execute immediately.
-   **Safety**: NEVER auto-download multiple search results without user selection.

## 🏗 Project Architecture

-   **Type**: Full-stack Web Application (React Frontend + Python FastAPI Backend).
-   **Dual-Process Setup**:
    1.  **MCP Server** (`dptb-tools`): Runs on port 50002. Exposes file/search tools.
    2.  **Main App** (`dptb-pilot`): Runs on port 8000 (Backend) & 50001 (Frontend).
-   **Shared Environment**: `dptb-pilot` and `DeePTB` **MUST** be1.  **Backend (`dptb_pilot/`)**:
    -   **Environment**: Dependencies managed via `pip install -e .`.
    -   **Entry Points**: defined in `pyproject.toml`.
    -   **API Contract**: The frontend communicates via `/api/...` which is proxied to the backend.
    -   **Tool Modifications**: Tools are defined in `dptb_pilot/tools/modules`.

2.  **Frontend (`web_ui/`)**:
    -   **Stack**: React, TypeScript, Ant Design, Recoil (state), 3Dmol.js (molecular viz).
    -   **API Service**: `web_ui/src/services/api.ts` is the single source of truth for backend interaction. All new API endpoints must be defined here.ver`: `npm run dev` runs on port 5173 (proxied to backend).

### Backend Development (`backend/`)
-   **Main Logic** (`better_aim/`): FastAPI server (`react_host.py`) and Agent orchestration (`agent.py`).
-   **MCP Tools** (`dptb_agent_tools/`):
    -   Located in `backend/dptb_agent_tools/modules/`.
    -   New tools must be registered in `main.py`.
    -   **Guardrails**: Tools in `target_tools` invoke `tool_modify_guardrail` to allow user-   **Structure**:
    -   `web_ui/`: React application (Vite, Ant Design, TypeScript).
    -   `dptb_pilot/`: Python backend (FastAPI, LiteLLM, MCP).
        -   `core/`: Core agent logic.
        -   `server/`: API server.
        -   `tools/`: MCP (Model Context Protocol) tool implementations. |
| `frontend/src/services/` | API interaction layer (`api.ts`). |
| `workspace/` | Default location for user sessions and files. |

## 🧪 Testing & Debugging

-   **Tools**: Use `dptb-tools --help` to check tool registration.
-   **Backend**: `curl http://localhost:8000/health`.
-   **Logs**: Check terminal output for `--- Callback:` to trace tool guardrails.
