# langgraph-HITL-streamlit-app

A minimal Streamlit application demonstrating Human-in-the-Loop (HITL) decision control for tool execution using LangGraph. The agent drafts emails and proposes sending them, but the actual tool execution is paused until a human approves, edits, or rejects the action.

## Overview

This project shows how to integrate:

- LangChain and LangGraph agent execution
- A custom tool (`send_email`)
- Human-in-the-loop middleware that interrupts before a tool is executed
- A simple Streamlit UI to review and act on pending tool calls

Use case: The assistant drafts an email based on user instructions and prepares a call to the `send_email` tool. Instead of executing immediately, the application pauses and waits for human approval or modification.

## Features

- Single text input for natural language instructions
- Automatic email drafting by the agent
- Automatic detection of tool execution and interruption before side effects
- Human review interface with the following actions:
  - Approve tool execution
  - Edit tool arguments before execution
  - Reject tool execution entirely
- Clear resume flow based on user decisions

## Requirements

- Python 3.9+
- A valid OpenAI API key set in your environment or `.env` file

## Installation

```bash
git clone https://github.com/justinwkUKM/langgraph-HITL-streamlit-app.git
cd langgraph-HITL-streamlit-app
pip install -r requirements.txt
```

Create a `.env` file if needed:

```
OPENAI_API_KEY=your_key_here
```

## Running the Application

```bash
streamlit run streamlit_hitl_minimal.py
```

(Or use `streamlit_hitl_email_app.py` if using the full UI version.)

Then open your browser to the URL Streamlit provides.

## Usage

1. Enter a natural language instruction, for example:  
   `Send an email to bob@example.com with subject Follow up and body Let me know your availability`
2. Click **Run**.
3. If the agent proposes a `send_email` call, an interrupt will be displayed.
4. Review the drafted message and proposed tool arguments.
5. Choose:
   - **Approve**: executes the tool.
   - **Edit**: modify arguments before sending.
   - **Reject**: skip sending entirely.

## Architecture

- The agent is built using `create_agent` and guided via system prompt.
- `HumanInTheLoopMiddleware` halts execution when a tool is about to run.
- `InMemorySaver` provides checkpointing tied to a session thread ID.
- Streamlit handles:
  - Capturing user instructions
  - Displaying interrupts
  - Resuming execution with user decisions (`approve`, `edit`, or `reject`)

## Customization

You can replace the `send_email` function with actual email-sending logic (SMTP, Gmail API, SendGrid, etc.).

For persistent or multi-user environments, replace `InMemorySaver` with a database-backed checkpointer.

## License

MIT License. See `LICENSE` for details.

## Author

Waqas Khalid
