# GEMINI.md

## Project Overview

This is a full-stack web application designed as an intelligent copilot for the DeePTB software package. It provides a chat-based interface for researchers to interact with DeePTB, leveraging the power of Large Language Models (LLMs) and Retrieval-Awgmented Generation (RAG) to answer questions and perform tasks.

The project is composed of:

*   **Frontend**: A [React](https://react.dev/) application built with [Vite](https://vitejs.dev/) and using the [Ant Design](https://ant.design/) component library. It features a chat interface, 3D structure visualization with [3Dmol.js](https://3dmol.csb.pitt.edu/), and file management capabilities. The source code is in the `web_ui/` directory.

*   **Backend**: A [Python](https://www.python.org/) backend powered by [FastAPI](https://fastapi.tiangolo.com/) and [LiteLLM](https://github.com/BerriAI/litellm). It uses the Model Context Protocol (MCP) and the Bohr Agent SDK to manage AI agent interactions. The backend serves a REST API and a WebSocket connection for real-time communication with the frontend. The source code is in the `dptb_pilot/` directory.

The application is designed to work with the [DeePTB](https://github.com/deepmodeling/DeePTB) package, which must be installed in the same Python environment.

## Building and Running

### Prerequisites

*   **uv** (Required):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Node.js & npm** (Required for Frontend)
*   **Git**

### 1. Installation

First, clone the repository.

```bash
git clone https://github.com/DeePTB-Lab/dptb-pilot.git
cd dptb-pilot
```

Run the installation script. This will set up the environment using `uv` and install the correct version of PyTorch and related dependencies.

```bash
# For CPU (default)
./install.sh

# For CUDA 11.8
./install.sh cu118

# For CUDA 12.1
./install.sh cu121

# For CUDA 12.4
./install.sh cu124
```

### 2. Configuration

Create a `.env` file from the example and fill in your API keys.

```bash
cp env.example .env
# Edit .env with your API keys (e.g., LLM_API_KEY)
```

### 3. Running the Application

The application requires two separate processes to be run in parallel.

**Terminal 1: Start the MCP Tool Server**

This server exposes the agent's tools (e.g., file operations, materials search).

```bash
dptb-tools
```

**Terminal 2: Start the Main Pilot Application**

This command starts the backend server and the frontend development server.

```bash
dptb-pilot
```

The application will be available at `http://localhost:50001`.

### 4. Running from Anywhere (Recommended)

To run the full application (Pilot + Tools) from any directory with a single command, use the `start.sh` script.

The `./install.sh` script will suggest an alias for you. Add it to your `~/.zshrc` or `~/.bashrc`:

```bash
# Replace with your actual path
alias dptb='/absolute/path/to/dptb-pilot/activate.sh'
```

Then you can simply run:

```bash
dptb
```

This will launch both services in parallel and handle clean shutdown when you press `Ctrl+C`.

The application logic automatically finds the configuration (`.env`) from your current working directory.

## Development Conventions

*   **Backend**: The backend follows standard Python packaging practices. Code is located in the `dptb_pilot/` directory and organized into `core` (main application) and `tools` (MCP tools). Tests are in the `tests/` directory and can be run with `pytest`.

*   **Frontend**: The frontend is a standard Vite-based React application. Code is located in `web_ui/src/`. It uses TypeScript for static typing and ESLint for code linting (`npm run lint`).

*   **Entry Points**: The main application entry points are defined as scripts in `pyproject.toml`:
    *   `dptb-pilot`: Starts the main application (`dptb_pilot/main.py`).
    *   `dptb-tools`: Starts the MCP tool server (`dptb_pilot/tools/server.py`).
