<div align="center">
  <img src="frontend/public/pilot_logo_black.png" alt="DeePTB Pilot Logo" width="500"/>
  <h3>AI Agent for DeePTB</h3>
  <p>An intelligent copilot for the DeePTB software package, integrating LLM capabilities with material science tools.</p>
</div>

---

## 📖 Introduction

**DeePTB Pilot** is a sophisticated AI agent designed to assist researchers in using the DeePTB software. It combines a React-based frontend with a Python backend powered by LLMs and the Model Context Protocol (MCP).

### ✨ Key Features

*   **Interactive Chat**: Natural language interface to query DeePTB knowledge and perform tasks.
*   **RAG System**: Retrieval-Augmented Generation for accurate answers based on DeePTB documentation.
*   **Structure Visualization**: Interactive 3D visualization of crystal structures and Brillouin zones.
*   **MCP Tools**: Extensible tool system for file operations, materials search (Materials Project), and DeePTB calculations.
*   **Session Management**: Persistent chat sessions and history.

## 🛠️ Architecture

*   **Frontend**: React, Ant Design, 3Dmol.js (located in `frontend/`)
*   **Backend**: Python, LiteLLM, MCP, Bohr Agent SDK (located in `backend/`)
    *   `better_aim/`: Main application logic and API server.
    *   `dptb_agent_tools/`: Collection of MCP tools.

## 🚀 Getting Started

### Prerequisites

*   **Python 3.10+**
*   **Node.js & npm**
*   **Git**
*   **DeePTB**: This pilot agent is designed to work with the [DeePTB](https://github.com/deepmodeling/DeePTB) package.
    > **Important**: `dptb-pilot` and `DeePTB` MUST be installed in the **same Python environment** to ensure the agent can correctly invoke DeePTB tools.

### 1. Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/DeePTB-Lab/dptb-pilot.git
cd dptb-pilot

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configuration

Copy the example environment file and configure your settings:

```bash
cp env.example .env
```

Edit `.env` with your API keys and preferences:

```env
# LLM Configuration
LLM_MODEL=openai/gpt-4o  # or your preferred model
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=your_actual_api_key_here

# Materials Project API (Optional)
MP_API_KEY=your_mp_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=50001
BACKEND_HOST=localhost
MCP_TOOLS_PORT=50002
MCP_TOOLS_URL=http://localhost:${MCP_TOOLS_PORT}/sse

# Workspace Configuration
WORK_ROOT=./workspace
```

### 3. Running the Application

You need to start two services: the **MCP Tool Server** and the **Main Pilot Application**.

**Terminal 1: Start MCP Tools**

```bash
dptb-tools
```

**Terminal 2: Start Pilot App**

```bash
dptb-pilot
```

The application should automatically open in your browser at `http://localhost:50001`.

## 📚 Documentation

*   [**Usage Guide**](docs/USAGE.md): Detailed instructions on how to use the chat, file management, and visualization features.

## ⚠️ Troubleshooting

### Proxy Issues

If you are behind a proxy or VPN and cannot connect to `localhost`, set the `NO_PROXY` environment variable:

```bash
export NO_PROXY="localhost,127.0.0.1"
```

Or run the commands with the variable prepended:

```bash
NO_PROXY="localhost,127.0.0.1" dptb-pilot
```

## 📄 License

[License Information]
