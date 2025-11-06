from time import sleep
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command

import time

load_dotenv()

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
    # Demo stub: replace with your real mailer integration.
    return f"Email sent successfully to {recipient} (subject: {subject})"

# ---------------- Agent ----------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for Sydney that can send emails. "
    "When responding to email tasks, draft the reply AND propose calling the send_email tool. "
    "Prefer using the tool rather than answering directly."
)

print("creating agent now...")
agent = create_agent(
    model="gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware( 
            interrupt_on={
                "send_email": True,  # All decisions (approve, edit, reject) allowed
            },
            # Prefix for interrupt messages - combined with tool name and args to form the full message
            # e.g., "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            # Individual tools can override this by specifying a "description" in their interrupt config
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop requires checkpointing to handle interrupts.
    # In production, use a persistent checkpointer like AsyncPostgresSaver.
    checkpointer=InMemorySaver(),  
)

print("agent defined" )

# ---------------- Command ----------------

# Human-in-the-loop leverages LangGraph's persistence layer.
# You must provide a thread ID to associate the execution with a conversation thread,
# so the conversation can be paused and resumed (as is needed for human review).
config = {"configurable": {"thread_id": "some_id"}} 
# Run the graph until the interrupt is hit.
time.sleep(5)

print("invoking agent now...")


low_stakes_email = """
Respond to the following email:
From: alice@example.com
Subject: Coffee?
Body: Hey, would you like to grab coffee next week?
"""

# Consequential email
consequential_email = """
Respond to the following email:
From: partner@startup.com
Subject: Budget proposal for Q1 2026
Body: Hey Sydney, we need your sign-off on the $1M engineering budget for Q1. Can you approve and reply by EOD? This is critical for our timeline.
"""

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": low_stakes_email,
            }
        ]
    },
    config=config 
)


# The interrupt contains the full HITL request with action_requests and review_configs
print("interrupt___-: ", result['__interrupt__'])
# interrupt___-:  [Interrupt(value={'action_requests': [{'name': 'send_email', 'args': {'recipient': 'jdoe@gmail.com', 'subject': 'Hello', 'body': 'How are you?'}, 'description': "Tool execution pending approval\n\nTool: send_email\nArgs: {'recipient': 'jdoe@gmail.com', 'subject': 'Hello', 'body': 'How are you?'}"}], 'review_configs': [{'action_name': 'send_email', 'allowed_decisions': ['approve', 'edit', 'reject']}]}, id='d820e8dfd1962754ee954b83aadf9786')]

print("resuming agent now with approval decision...")
time.sleep(5)
# Resume with approval decision
invocation = agent.invoke(
    Command( 
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ), 
    config=config # Same thread ID to resume the paused conversation
)

print("invocation___-: ", invocation)
# invocation___-:  {'messages': [HumanMessage(content="Send an email to John Doe jdoe@gmail.com with the subject 'Hello' and the body 'How are you?'", additional_kwargs={}, response_metadata={}, 
# id='36c31c12-0ff0-45db-a500-292a3f9f347a'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 134, 'total_tokens': 163,
#  'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 
# 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_cbf1785567', 'id': 'chatcmpl-CYwV2MvmfMW40mqMlQlokVVI5S6SN', 'service_tier': 'default', 'finish_reason': 'tool_calls', 
# 'logprobs': None}, id='lc_run--2b28fb94-4cbb-4fa1-a9fb-e7b7aa77f8b9-0', tool_calls=[{'name': 'send_email', 'args': {'recipient': 'jdoe@gmail.com', 'subject': 'Hello', 'body': 'How are you?'}, 'id': 'call_bvPU315bb6Zk9TgYhx5Qqqp9', 
# 'type': 'tool_call'}], usage_metadata={'input_tokens': 134, 'output_tokens': 29, 'total_tokens': 163, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), 
# ToolMessage(content='Email sent successfully to jdoe@gmail.com (subject: Hello)', name='send_email', id='071be03e-c64e-4408-bd29-c616ad9a40da', tool_call_id='call_bvPU315bb6Zk9TgYhx5Qqqp9'), 
# AIMessage(content='The email has been sent to John Doe with the subject "Hello" and the body "How are you?".',
# additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 185, 'total_tokens': 208, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 
# 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_cbf1785567', 'id': 'chatcmpl-CYwV3bYId7HXMTcgYojnorzo7rotr', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 
# id='lc_run--e929c53e-57d8-45c5-b015-419400c01a79-0', usage_metadata={'input_tokens': 185, 'output_tokens': 23, 'total_tokens': 208, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}