import time
import os
from typing import Any, Dict, Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.callbacks.base import BaseCallbackHandler
from langchain_arcade import ArcadeToolManager
from langgraph.errors import NodeInterrupt
from langsmith import Client


# Set up Streamlit page configuration
st.set_page_config(
    page_title="Arcade Tools Chat",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check authentication
if st.experimental_user.get("email") is None:
    st.login()

# Add sidebar profile section
with st.sidebar:
    if st.experimental_user.get("email"):
        st.title("Profile")
        col1, col2 = st.columns([1, 3])
        with col1:
            # Display profile image if available, otherwise show a placeholder
            profile_url = st.experimental_user.get("picture")
            if profile_url:
                st.image(str(profile_url), width=100)
            else:
                st.markdown("👤")

        with col2:
            # Display name if available, otherwise show email
            if hasattr(st.experimental_user, "name") and st.experimental_user.name:
                st.markdown(f"**{st.experimental_user.name}**")
            st.markdown(f"_{st.experimental_user.email}_")

        # User details expander
        with st.expander("User Details"):
            st.write(st.experimental_user)

        st.divider()
        if st.button("Logout", use_container_width=True, icon=":material/logout:"):
            st.logout()
    else:
        st.write("Please log in to continue")
        st.login()


manager = ArcadeToolManager()


def authorize_tool(tool_name: str):
    """Authorize a tool."""

    # If it does not require authorization, return true
    if manager.requires_auth(tool_name) is False:
        return True

    # If the user is not logged in, login
    if not st.experimental_user.get("email"):
        st.login()

    # Request authorization
    try:
        auth_response = manager.authorize(
            tool_name=tool_name, user_id=str(st.experimental_user.get("email"))
        )

        # If the authorization is completed, return true
        if auth_response.status == "completed":
            return True

        # If there is a url, give the user a button to authorize the tool
        if auth_response.url and auth_response.id:
            if st.link_button("Authorize", auth_response.url):
                wait_response = manager.wait_for_auth(auth_response.id)
                if wait_response.status == "completed":
                    return True
    except Exception as e:
        st.error(f"Failed to authorize tool: {str(e)}")
        return False

    return False


class TokenStreamHandler(BaseCallbackHandler):
    """Handler for streaming tokens to a Streamlit container."""

    def __init__(self, container):
        self.container = container
        self.text = ""
        self.run_id = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens to the container."""
        self.text += token
        self.container.markdown(self.text)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Capture run ID when chain starts."""
        if "run_id" in kwargs:
            self.run_id = kwargs["run_id"]


def handle_message_edit(idx: int, new_content: Optional[str]) -> None:
    """Handle editing a message and restarting from that point."""
    if idx < len(st.session_state.messages) and new_content is not None:
        # Update the message
        st.session_state.messages[idx]["content"] = new_content
        # Remove all subsequent messages
        st.session_state.messages = st.session_state.messages[: idx + 1]
        # Update checkpoint ID to force a restart from this point
        st.session_state.checkpoint_id = f"chat_{idx}_{int(time.time())}"
        # Set flag to trigger re-run
        st.session_state.should_rerun = True
        # Clear editing state
        st.session_state.editing_message_idx = None
        st.rerun()


def submit_feedback(run_id: Optional[str], score: int, comment: str = "") -> None:
    """Submit feedback to LangSmith."""
    feedback_client = None
    try:
        feedback_client = Client()
    except Exception as e:
        st.error(f"Failed to initialize feedback client: {str(e)}")
    if run_id is not None and feedback_client is not None:
        try:
            feedback_client.create_feedback(
                run_id, "user_score", score=score, comment=comment
            )
            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Failed to submit feedback: {str(e)}")
    else:
        if feedback_client is None:
            st.warning("Feedback submission is disabled - LangSmith is not configured.")
        else:
            st.warning("Cannot submit feedback - no run ID available.")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = {
        "1": {
            "messages": [],
            "checkpoint_id": "chat_1",
            "name": "Thread 1",
            "created_at": time.time(),
        }
    }
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = "1"
if "editing_message_idx" not in st.session_state:
    st.session_state.editing_message_idx = None
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None
if "should_rerun" not in st.session_state:
    st.session_state.should_rerun = False
if "checkpoint_id" not in st.session_state:
    st.session_state.checkpoint_id = "chat_1"


def create_new_thread():
    """Create a new thread and switch to it."""
    thread_id = str(int(time.time()))
    st.session_state.threads[thread_id] = {
        "messages": [],
        "checkpoint_id": f"chat_{thread_id}",
        "name": f"Thread {len(st.session_state.threads) + 1}",
        "created_at": time.time(),
    }
    st.session_state.current_thread_id = thread_id
    st.session_state.messages = []
    st.rerun()


def switch_thread(thread_id: str):
    """Switch to a different thread."""
    if thread_id in st.session_state.threads:
        st.session_state.current_thread_id = thread_id
        st.session_state.messages = st.session_state.threads[thread_id]["messages"]
        st.rerun()


def update_thread_messages():
    """Update the current thread's messages."""
    st.session_state.threads[st.session_state.current_thread_id][
        "messages"
    ] = st.session_state.messages


def init_agent(callbacks=None) -> Any:
    """Initialize the agent with tools and model"""

    # Get Google toolkit, GitHub toolkit, and Web toolkit
    tools = manager.get_tools(toolkits=["Google", "GitHub", "Web"])

    # Set up the language model with callbacks for final response only
    model = ChatOpenAI(
        model=st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4")),
        streaming=True,
        callbacks=callbacks,  # Pass callbacks directly to allow token streaming
    )

    # Set up memory with checkpointing
    memory = MemorySaver()

    # Create the graph with model and tools
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
    )

    return agent, memory


# Display chat header
st.title("🛠️ Arcade Tools Chat")
st.markdown("Chat with an AI assistant powered by Arcade's productivity tools!")

# Display chat messages with edit controls
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Check if this message is being edited
        if st.session_state.editing_message_idx == idx:
            # Only show edit interface for user messages
            if message["role"] == "user":
                edit_col, button_col = st.columns([4, 1])
                with edit_col:
                    edited_content = st.text_area(
                        "Edit message", message["content"], key=f"edit_{idx}"
                    )
                with button_col:
                    if st.button("Save", key=f"save_{idx}"):
                        handle_message_edit(idx, edited_content)
                    if st.button("Cancel", key=f"cancel_{idx}"):
                        st.session_state.editing_message_idx = None
                        st.rerun()
            else:
                st.markdown(message["content"])
                st.session_state.editing_message_idx = None
                st.rerun()
        else:
            # Show message with edit button only for user messages
            if message["role"] == "user":
                message_col, edit_col = st.columns([20, 1])
                with message_col:
                    st.markdown(message["content"])
                with edit_col:
                    if st.button("✏️", key=f"edit_{idx}", help="Edit message"):
                        st.session_state.editing_message_idx = idx
                        st.rerun()
            else:
                st.markdown(message["content"])
                # Add feedback interface for assistant messages
                if message["role"] == "assistant" and st.session_state.last_run_id:
                    with st.expander("Provide Feedback"):
                        st.markdown("#### Was this response helpful?")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                "👍 Helpful",
                                key=f"helpful_{st.session_state.last_run_id}_{idx}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 1)
                                st.success("Thank you for your feedback!")
                        with col2:
                            if st.button(
                                "👎 Not Helpful",
                                key=f"not_helpful_{st.session_state.last_run_id}_{idx}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 0)
                                st.error(
                                    "Sorry to hear that. Your feedback helps us improve!"
                                )

                        # Add comment section
                        st.markdown("---")
                        comment = st.text_area(
                            "What could be improved?",
                            key=f"comment_{st.session_state.last_run_id}_{idx}",
                        )
                        if st.button(
                            "Submit Detailed Feedback",
                            key=f"submit_{st.session_state.last_run_id}_{idx}",
                        ):
                            submit_feedback(st.session_state.last_run_id, -1, comment)
                            st.success("Thank you for your detailed feedback!")

# After displaying messages, check if we need to re-run from an edit
if st.session_state.should_rerun:
    st.session_state.should_rerun = False  # Reset the flag

    with st.chat_message("assistant"):
        try:
            # Initialize containers
            response_container = st.empty()

            # Create callback handler for final response only
            token_callback = TokenStreamHandler(response_container)

            # Initialize agent with streaming callback for final response
            agent_tuple = init_agent(callbacks=[token_callback])
            agent = agent_tuple[0]  # Unpack the agent from the tuple

            # Get the last user message
            last_message = st.session_state.messages[-1]
            if last_message["role"] != "user":
                st.error("Expected last message to be from user")
                st.stop()

            # Prepare input with all messages up to the edit point
            message_history = [
                (msg["role"], msg["content"]) for msg in st.session_state.messages
            ]

            user_input = {"messages": message_history}

            config = {
                "configurable": {
                    "thread_id": st.session_state.current_thread_id,
                    "checkpoint_id": st.session_state.checkpoint_id,
                    "checkpoint_ns": "default",
                    "user_id": st.experimental_user.get("email", None),
                }
            }

            # Get the final response from the token stream handler
            final_content = token_callback.text
            if final_content and final_content.strip():
                # Display the response
                response_container.markdown(final_content)

                # Add to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_content}
                )
                # Store run ID for feedback
                st.session_state.last_run_id = token_callback.run_id

                # Add feedback interface in a container below the response
                feedback_container = st.container()
                with feedback_container:
                    with st.expander("Provide Feedback"):
                        st.markdown("#### Was this response helpful?")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                "👍 Helpful",
                                key=f"helpful_{st.session_state.last_run_id}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 1)
                                st.success("Thank you for your feedback!")
                        with col2:
                            if st.button(
                                "👎 Not Helpful",
                                key=f"not_helpful_{st.session_state.last_run_id}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 0)
                                st.error(
                                    "Sorry to hear that. Your feedback helps us improve!"
                                )

                        # Add comment section
                        st.markdown("---")
                        comment = st.text_area(
                            "What could be improved?",
                            key=f"comment_{st.session_state.last_run_id}",
                        )
                        if st.button(
                            "Submit Detailed Feedback",
                            key=f"submit_{st.session_state.last_run_id}",
                        ):
                            submit_feedback(st.session_state.last_run_id, -1, comment)
                            st.success("Thank you for your detailed feedback!")

        except NodeInterrupt as e:
            display_interrupt_details(e)

        except Exception as e:
            st.error(f"An error occurred while re-running: {str(e)}")

# Chat input
if prompt := st.chat_input("Ask me to analyze any webpage!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response using the agent
    with st.chat_message("assistant"):
        try:
            # Initialize containers
            response_container = st.empty()

            # Create callback handler for final response only
            token_callback = TokenStreamHandler(response_container)

            # Initialize agent with streaming callback for final response
            agent_tuple = init_agent(callbacks=[token_callback])
            agent = agent_tuple[0]  # Unpack the agent from the tuple

            # Prepare input according to LangGraph specs
            message_history = [
                (msg["role"], msg["content"]) for msg in st.session_state.messages
            ]

            user_input = {"messages": message_history}

            config = {
                "configurable": {
                    "thread_id": st.session_state.current_thread_id,
                    "checkpoint_id": st.session_state.checkpoint_id,
                    "checkpoint_ns": "default",
                    "user_id": st.experimental_user.get("email", None),
                }
            }

            # Get the final response from the token stream handler
            final_content = token_callback.text
            if final_content and final_content.strip():
                # Display the response
                response_container.markdown(final_content)

                # Add to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_content}
                )
                # Store run ID for feedback
                st.session_state.last_run_id = token_callback.run_id

                # Add feedback interface in a container below the response
                feedback_container = st.container()
                with feedback_container:
                    with st.expander("Provide Feedback"):
                        st.markdown("#### Was this response helpful?")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                "👍 Helpful",
                                key=f"helpful_{st.session_state.last_run_id}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 1)
                                st.success("Thank you for your feedback!")
                        with col2:
                            if st.button(
                                "👎 Not Helpful",
                                key=f"not_helpful_{st.session_state.last_run_id}",
                                use_container_width=True,
                            ):
                                submit_feedback(st.session_state.last_run_id, 0)
                                st.error(
                                    "Sorry to hear that. Your feedback helps us improve!"
                                )

                        # Add comment section
                        st.markdown("---")
                        comment = st.text_area(
                            "What could be improved?",
                            key=f"comment_{st.session_state.last_run_id}",
                        )
                        if st.button(
                            "Submit Detailed Feedback",
                            key=f"submit_{st.session_state.last_run_id}",
                        ):
                            submit_feedback(st.session_state.last_run_id, -1, comment)
                            st.success("Thank you for your detailed feedback!")

        except NodeInterrupt as e:
            display_interrupt_details(e)

        except Exception as e:
            st.error(f"An error occurred sending the request to the agent: {str(e)}")


def display_interrupt_details(e: NodeInterrupt) -> None:
    """Display interrupt details using st.write."""
    interrupt_details = {
        "Type": type(e).__name__,
        "Message": str(e),
        "Context": e.context if hasattr(e, "context") else "N/A",
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "Actions": e.actions if hasattr(e, "actions") else "N/A",
    }
    st.write("Interrupt details:", interrupt_details)
