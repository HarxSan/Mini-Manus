# app.py
import os
import streamlit as st
import uuid
import sys
import platform
from typing import List, Any, Annotated
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
import nest_asyncio
import traceback
from browser_client import BrowserAutomationClient
import asyncio

# Enable nested asyncio to run async functions inside Streamlit
nest_asyncio.apply()

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# Load all required API keys
GROQ_KEY = os.getenv("GROQ")
TAVILY_KEY = os.getenv("TAVILY")
LANGCHAIN_KEY = os.getenv("LANGCHAIN")

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TAVILY_API_KEY"] = TAVILY_KEY or ""

# === Fix for Windows asyncio subprocess issue ===
if platform.system() == "Windows":
    import asyncio
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
        # For Python 3.8+
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Create event loop explicitly for Windows
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        # For older Python versions
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]

class AIAssistant:
    def __init__(self, temperature=0.3):
        if not GROQ_KEY:
            st.error("❌ Missing GROQ API key for conversational mode. Check your .env file.")
            return
            
        self.llm = ChatGroq(
            temperature=temperature,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=GROQ_KEY
        )
        
        if not TAVILY_KEY:
            st.warning("⚠️ Missing TAVILY API key. Search functionality will be limited.")
            self.search_tool = None
            self.tools = []
        else:
            self.search_tool = TavilySearchResults(max_results=5)
            self.tools = [self.search_tool]
            
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(State)
        self.setup_graph()

    def setup_graph(self):
        self.graph_builder.add_node("chatbot", self.chatbot)
        
        if self.tools:
            tool_node = ToolNode(tools=self.tools)
            self.graph_builder.add_node("tools", tool_node)
            self.graph_builder.add_conditional_edges(
                "chatbot",
                tools_condition,
            )
            self.graph_builder.add_edge("tools", "chatbot")
            
        self.graph_builder.add_edge(START, "chatbot")
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def chatbot(self, state: State):
        """Core chatbot logic with tool integration"""
        messages = state["messages"]
        try:
            message = self.llm_with_tools.invoke(messages)
            if hasattr(message, 'tool_calls'):
                assert len(message.tool_calls) <= 1
            return {"messages": [message]}
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            return {"messages": [AIMessage(content=error_msg)]}
    
    def run_conversation(self, user_input: str, thread_id: str):
        """Execute conversation flow"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            events = self.graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values"
            )
            
            final_response = None
            for event in events:
                if "messages" in event:
                    final_response = event["messages"][-1]
            
            return final_response
        except Exception as e:
            error_msg = f"Conversation error: {str(e)}"
            st.error(error_msg)
            return AIMessage(content=error_msg)

def init_session_state():
    """Initialize all session state variables"""
    if "browser_client" not in st.session_state:
        st.session_state.browser_client = None
    if "awaiting_input" not in st.session_state:
        st.session_state.awaiting_input = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "browser_response" not in st.session_state:
        st.session_state.browser_response = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "browser_initialized" not in st.session_state:
        st.session_state.browser_initialized = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

def main():
    st.set_page_config(
        page_title="AI Assistant", 
        page_icon="🧠", 
        layout="wide"
    )
    
    # Initialize session state variables
    init_session_state()
    
    with st.sidebar:
        st.title("🧠 AI Assistant Controls")
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            # Properly close browser if it's open
            if st.session_state.browser_client:
                with st.spinner("Closing browser..."):
                    st.session_state.browser_client.close_browser()
                    
            # Reset all state
            for key in list(st.session_state.keys()):
                if key != "mode_toggle":  # Preserve mode toggle setting
                    del st.session_state[key]
                    
            init_session_state()
            st.rerun()
       
        st.header("Model Configuration")
        model_choice = st.selectbox(
            "Select Model", 
            ["LLaMA 3.3", "GPT-4"], 
            index=0
        )
        
        # Settings
        st.header("Advanced Settings")
        temperature = st.slider(
            "Creativity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,
            step=0.1
        )
        
        st.header("Mode Selection")
        
        mode_switch = st.toggle(
            "Enable Browser Automation Mode", 
            key="mode_toggle",
            help="Switch between conversational AI and browser automation"
        )
        
        if mode_switch:
            st.warning("🌐 Browser Automation Mode Activated")
            st.info("Inputs will now trigger browser actions")
            
            # Initialize browser client if not already done
            if not st.session_state.browser_client:
                st.session_state.browser_client = BrowserAutomationClient()
                
            # Display browser initialization button
            browser_service_url = st.text_input("Browser Service URL", "http://localhost:8000")
            chrome_path = st.text_input("Chrome Path (optional)", "")
            
            if not st.session_state.browser_initialized:
                if st.button("Initialize Browser", use_container_width=True):
                    with st.spinner("Initializing browser..."):
                        try:
                            # Update the service URL
                            st.session_state.browser_client.base_url = browser_service_url
                            
                            # Initialize the browser
                            success = st.session_state.browser_client.initialize_browser(
                                chrome_path=chrome_path if chrome_path else None,
                                headless=False
                            )
                            
                            if success:
                                st.session_state.browser_initialized = True
                                st.success("Browser initialized successfully!")
                            else:
                                st.error("Failed to initialize browser")
                                st.info("Make sure the browser service is running at " + browser_service_url)
                        except Exception as e:
                            st.error(f"Error initializing browser: {str(e)}")
                            traceback.print_exc()
            
            if st.session_state.browser_initialized:
                st.success("✅ Browser is initialized and ready!")
                
                # Allow checking the status
                if st.button("Check Browser Status"):
                    status = st.session_state.browser_client.get_status()
                    st.json(status)
        else:
            st.success("💬 Conversational AI Mode Active")
            # Close browser if switching away from browser mode
            if st.session_state.browser_client and st.session_state.browser_initialized:
                if st.button("Close Browser"):
                    with st.spinner("Closing browser..."):
                        success = st.session_state.browser_client.close_browser()
                        if success:
                            st.session_state.browser_initialized = False
                            st.success("Browser closed successfully!")
        
        st.markdown("---")        
        max_tokens = st.number_input(
            "Max Response Length", 
            min_value=50, 
            max_value=1000, 
            value=300
        )
        
        st.markdown("---")
        if mode_switch:
            st.info("Powered by LangGraph, Gemini & Browser-Use Microservice")
        else:
            st.info("Powered by LangGraph & Groq")

    st.title("🤖 AI Assistant")
    
    if st.session_state.mode_toggle:
        st.caption("Browser Automation Mode Active")
    else:
        st.caption("Intelligent Conversational AI")

    # Initialize conversational assistant
    assistant = AIAssistant(temperature=temperature/10)
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("human", avatar="👤"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)

    # If we're waiting for a response to a browser question
    if st.session_state.awaiting_input and st.session_state.pending_question:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Browser requires information:** {st.session_state.pending_question}")
        
        browser_input = st.text_input("Your response:", key="browser_input")
        
        if st.button("Submit Response"):
            # Store response in a place our callback can access
            st.session_state.browser_response = browser_input
            st.session_state.awaiting_input = False
            st.session_state.pending_question = None
            
            # Provide the input to the browser service
            if st.session_state.browser_client:
                success = st.session_state.browser_client.provide_input(browser_input)
                if not success:
                    st.error("Failed to send input to browser service")
            
            st.rerun()
    
    # Regular chat input
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Browser automation mode
        if st.session_state.mode_toggle:
            # Check if browser is initialized
            if not st.session_state.browser_initialized:
                st.session_state.chat_history.append(
                    AIMessage(content="⚠️ Browser not initialized. Please initialize the browser first.")
                )
                st.rerun()
                
            with st.spinner("Processing browser automation..."):
                if not st.session_state.browser_client:
                    st.session_state.browser_client = BrowserAutomationClient()
                
                # Define the callback for handling user input requests during browser automation
                def input_callback(question):
                    # Set the state to indicate we're waiting for input
                    st.session_state.awaiting_input = True
                    st.session_state.pending_question = question
                    st.session_state.browser_response = None
                    
                    # This will cause a rerun and pause execution here
                    st.rerun()
                    
                    # When we return here after the user provides input, we should have the response
                    response = st.session_state.get("browser_response", "No response provided")
                    return response
                
                try:
                    # Run the browser task
                    result = st.session_state.browser_client.run_task(
                        user_input, 
                        input_callback=input_callback
                    )
                    
                    # Check for errors
                    if "error" in result:
                        st.session_state.chat_history.append(
                            AIMessage(content=f"⚠️ Browser automation error: {result['error']}")
                        )
                    else:
                        # Format the successful result
                        response_text = f"Browser automation completed successfully.\n\n{result.get('result', 'Task completed')}"
                        st.session_state.chat_history.append(
                    AIMessage(content=response_text)
                    )
                
                except Exception as e:
                    error_msg = f"Browser automation error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        AIMessage(content=f"⚠️ {error_msg}")
                    )
                
                st.rerun()
        
        # Conversational AI mode
        else:
            with st.spinner("Analyzing your query..."):
                try:
                    response = assistant.run_conversation(
                        user_input, 
                        st.session_state.thread_id
                    )
                    
                    if response:
                        st.session_state.chat_history.append(response)
                    
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        AIMessage(content=f"⚠️ {error_msg}")
                    )
                
                st.rerun()

if __name__ == "__main__":
    main()
