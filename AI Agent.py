import os
import streamlit as st
import uuid
from typing import List, Dict, Any

# Langchain and LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from typing_extensions import TypedDict

# Environment Setup
from dotenv import load_dotenv
load_dotenv()

# Configuration and API Key Validation
GROQ_KEY = os.getenv("GROQ")
TAVILY_KEY = os.getenv("TAVILY")
LANGCHAIN_KEY = os.getenv("LANGCHAIN")

if not all([GROQ_KEY, TAVILY_KEY, LANGCHAIN_KEY]):
    st.error("❌ Missing required API keys. Check your .env file.")
    st.stop()

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# State Definition
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]

class AIAssistant:
    def __init__(self, temperature=0.3):
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=temperature,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=GROQ_KEY
        )
        
        # Initialize Search Tool
        self.search_tool = TavilySearchResults(max_results=3)
        self.tools = [self.search_tool]
        
        # Prepare LLM with tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize Memory and Graph
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(State)
        self.setup_graph()
    
    def setup_graph(self):
        # Add chatbot node
        self.graph_builder.add_node("chatbot", self.chatbot)
        
        # Add tool node
        tool_node = ToolNode(tools=self.tools)
        self.graph_builder.add_node("tools", tool_node)
        
        # Set up conditional edges
        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge(START, "chatbot")
        
        # Compile graph
        self.graph = self.graph_builder.compile(checkpointer=self.memory)
    
    def chatbot(self, state: State):
        """Core chatbot logic with tool integration"""
        messages = state["messages"]
        
        try:
            # Invoke LLM with current conversation context
            message = self.llm_with_tools.invoke(messages)
            
            # Ensure only one tool call at a time
            assert len(message.tool_calls) <= 1
            
            return {"messages": [message]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error: {str(e)}")]}
    
    def run_conversation(self, user_input: str, thread_id: str):
        """Execute conversation flow"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            events = self.graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values"
            )
            
            # Collect final response
            final_response = None
            for event in events:
                if "messages" in event:
                    final_response = event["messages"][-1]
            
            return final_response
        except Exception as e:
            return AIMessage(content=f"Conversation error: {str(e)}")

def main():
    # Configure Page
    st.set_page_config(
        page_title="AI Assistant", 
        page_icon="🧠", 
        layout="wide"
    )

    # Sidebar Configuration
    with st.sidebar:
        st.title("🧠 AI Assistant Controls")
        
        # Conversation Management
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Model Settings
        st.header("Model Configuration")
        model_choice = st.selectbox(
            "Select Model", 
            ["LLaMA 3.3", "GPT-4"], 
            index=0
        )
        
        # Advanced Settings
        st.header("Advanced Settings")
        temperature = st.slider(
            "Creativity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,
            step=0.1
        )
        
        # Additional Controls
        max_tokens = st.number_input(
            "Max Response Length", 
            min_value=50, 
            max_value=1000, 
            value=300
        )
        
        # Info Section
        st.markdown("---")
        st.info("Powered by LangGraph & Groq")

    # Main Content Area
    st.title("🤖 AI Assistant")
    st.caption("Intelligent Conversational AI")

    # Create AI Assistant
    assistant = AIAssistant(temperature=temperature/10)
    
    # Unique conversation thread
    thread_id = str(uuid.uuid4())
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat Container
    chat_container = st.container()
    
    # Display Chat History
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("human", avatar="👤"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
    
    # Fixed Chat Input at Bottom
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.spinner("Analyzing your query..."):
            try:
                # Process input through AI assistant
                response = assistant.run_conversation(user_input, thread_id)
                
                # Add AI response to history
                if response:
                    st.session_state.chat_history.append(response)
                
                # Rerun to update the chat container
                st.rerun()
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()