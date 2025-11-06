import uuid
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# =====================
# Setup
# =====================
load_dotenv()
st.set_page_config(page_title="HITL Email", page_icon="📧", layout="centered")
st.title("📧 Minimal HITL Email Assistant")
st.caption("Single input. Interrupts captured. Approve / Edit / Reject working.")

# =====================
# Agent factory
# =====================

def build_agent():
    @tool(parse_docstring=True)
    def send_email(recipient: str, subject: str, body: str) -> str:
        """Send an email to a recipient.

        Args:
            recipient (str): Email address of the recipient.
            subject (str): Subject line of the email.
            body (str): Body content of the email.

        Returns:
            str: Confirmation message.
        """
        # Demo stub: replace with real email integration (SMTP, Gmail API, etc.)
        return f"Email sent successfully to {recipient} (subject: {subject})"

    SYSTEM_PROMPT = (
        "You are a helpful assistant for Sydney that can send emails. "
        "When responding to email tasks, draft the reply AND propose calling the send_email tool. "
        "Prefer using the tool rather than answering directly. Do not execute side-effecting tools without approval."
    )

    agent = create_agent(
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        tools=[send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": "Review this outgoing email before it’s sent.",
                    }
                },
                description_prefix="Tool execution pending approval",
            )
        ],
        checkpointer=InMemorySaver(),
    )
    return agent

# =====================
# Session state
# =====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None  # entire interrupt value
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None     # first action_request dict
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# =====================
# Helpers
# =====================

def run_agent(user_text: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return st.session_state.agent.invoke({"messages": [{"role": "user", "content": user_text}]}, config=config)


def resume(decision: Dict[str, Any]) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return st.session_state.agent.invoke(Command(resume={"decisions": [decision]}), config=config)


def extract_interrupt(result: Dict[str, Any]):
    intr = result.get("__interrupt__")
    if not intr:
        return None, None
    value = intr[0].value if isinstance(intr, list) else intr.value
    action_requests: List[Dict[str, Any]] = value.get("action_requests", [])
    return value, (action_requests[0] if action_requests else None)


def render_assistant_draft(result: Dict[str, Any]):
    msgs = result.get("messages", [])
    if not msgs:
        return
    st.write("**Assistant draft message:**")
    # Handle both LC message objects and dict messages
    try:
        ai_msgs = [m for m in msgs if getattr(m, "type", None) == "ai"]
        if ai_msgs:
            st.code(getattr(ai_msgs[-1], "content", "") or "", language="markdown")
            return
    except Exception:
        pass
    try:
        ai_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        if ai_msgs:
            st.code(ai_msgs[-1].get("content", "") or "", language="markdown")
            return
    except Exception:
        pass
    st.write(msgs)

# =====================
# Single input UI
# =====================
user_text = st.text_area(
    "Type your instruction (e.g. 'Send an email to jane@acme.com with subject Hello and body Hi there')",
    height=120,
)
colA, colB = st.columns([1,1])
with colA:
    run_clicked = st.button("Run")
with colB:
    new_thread = st.button("New thread")

if new_thread:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_interrupt = None
    st.session_state.pending_action = None
    st.session_state.edit_mode = False
    st.session_state.agent = build_agent()
    st.success("Started a new thread.")

# Step 1: Run agent to get interrupt
if run_clicked and user_text.strip():
    with st.spinner("Thinking..."):
        result = run_agent(user_text.strip())
    render_assistant_draft(result)

    value, ar = extract_interrupt(result)
    if ar:
        st.session_state.pending_interrupt = value
        st.session_state.pending_action = ar
        st.info("Interrupt captured. Review the proposed tool call below.")
    else:
        st.session_state.pending_interrupt = None
        st.session_state.pending_action = None
        st.success("No tool call proposed or nothing requiring approval.")

# Step 2: If we have an interrupt, show Approve / Edit / Reject
if st.session_state.pending_action:
    ar = st.session_state.pending_action
    st.subheader("Proposed tool call")
    st.code(f"Tool: {ar.get('name')}\nArgs: {ar.get('args')}", language="text")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Approve & Send"):
            with st.spinner("Sending..."):
                final = resume({"type": "approve"})
            st.success("Approved and sent.")
            st.json(final)
            # clear pending state
            st.session_state.pending_interrupt = None
            st.session_state.pending_action = None
            st.session_state.edit_mode = False

    with c2:
        if st.button("✏️ Edit"):
            st.session_state.edit_mode = True

    with c3:
        if st.button("🛑 Reject"):
            with st.spinner("Continuing without sending..."):
                final = resume({
                    "type": "reject",
                    "action_name": ar.get("name"),
                    "reason": "User rejected sending this email.",
                })
            st.warning("Rejected send.")
            st.json(final)
            st.session_state.pending_interrupt = None
            st.session_state.pending_action = None
            st.session_state.edit_mode = False

    # Inline edit form (only when Edit pressed)
    if st.session_state.edit_mode:
        st.markdown("---")
        st.subheader("Edit tool arguments")
        args = ar.get("args", {})
        r = st.text_input("Recipient", value=args.get("recipient", ""))
        s = st.text_input("Subject", value=args.get("subject", ""))
        b = st.text_area("Body", value=args.get("body", ""), height=140)
        if st.button("Save edits & Send"):
            with st.spinner("Sending with edits..."):
                final = resume({
                    "type": "edit",
                    "action_name": ar.get("name"),
                    "args": {"recipient": r, "subject": s, "body": b},
                })
            st.success("Edited and sent.")
            st.json(final)
            st.session_state.pending_interrupt = None
            st.session_state.pending_action = None
            st.session_state.edit_mode = False
