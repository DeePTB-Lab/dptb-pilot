# 📘 DeePTB Pilot Usage Guide

Welcome to the DeePTB Pilot! This guide will help you navigate the interface and make the most of the AI agent's capabilities.

## 🖥️ Interface Overview

The interface is divided into three main areas:

1.  **Left Sidebar (History)**: View and manage your past chat sessions.
2.  **Center (Chat)**: The main interaction area where you converse with the AI.
3.  **Right Sidebar (Tools & Files)**:
    *   **Parameters**: View and edit calculation parameters (if applicable).
    *   **Files**: Manage uploaded files and workspace content.

---

## 💬 Chatting with the Pilot

### Basic Interaction
Simply type your question or command in the input box at the bottom and press **Enter** or click the **Send** button.

*   **Ask Questions**: "How do I calculate the band structure of Silicon?"
*   **Request Actions**: "Search for MoS2 on Materials Project."
*   **Context**: The agent remembers the conversation context, so you can ask follow-up questions.

### Shortcut Cards
On a new chat screen, you will see shortcut cards for common tasks:
*   **Analyze Structures**: Upload a CIF/POSCAR file and ask for analysis.
*   **Calculate Bands**: Request a band structure calculation workflow.
*   **Generate Configs**: Ask for help creating `dptb` input files.
*   **Search Materials**: Query the Materials Project database.

---

## 📂 File Management

### Uploading Files
You can upload structure files (CIF, POSCAR, XYZ) or other data files to the workspace.
*   **Drag & Drop**: Drag files directly into the chat window.
*   **Paperclip Icon**: Click the paperclip icon in the input bar to select files.

### Managing Files
Switch to the **Files** tab in the Right Sidebar to see your workspace content.
*   **View**: Click on a file to preview its content (text files).
*   **Delete**: Use the trash icon to remove files.
*   **Copy Path**: Easily copy the file path for use in commands.

---

## ⚛️ Structure Visualization

When you upload a crystal structure or search for one, the Pilot can visualize it interactively.

*   **Rotate**: Click and drag to rotate the structure.
*   **Zoom**: Scroll to zoom in/out.
*   **Pan**: Right-click and drag to pan.
*   **Cell**: The unit cell box is displayed automatically.

---

## 🛠️ Using Tools (MCP)

The Pilot is equipped with "Tools" that allow it to interact with the outside world.

### Materials Project Search
You can ask the Pilot to find materials:
> "Find all stable cubic perovskites with a band gap > 1.5 eV."

The agent will query the Materials Project API and present the results.

### DeePTB Calculations
The agent can run DeePTB commands for you (requires DeePTB installation):
> "Run a bond analysis on `structure.cif`."
> "Generate a config template for a tight-binding model."

---

## ⌨️ Keyboard Shortcuts

*   **Enter**: Send message.
*   **Shift + Enter**: Insert a new line in the input box.

---

## ❓ FAQ

**Q: The agent says "I don't have that tool".**
A: Ensure the `dptb-tools` server is running in a separate terminal.

**Q: I can't see the visualization.**
A: Ensure you have uploaded a valid structure file (CIF/POSCAR) and the agent has recognized it.
