# browser_service.py
import asyncio
import os
import json
import traceback
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks # type: ignore
from pydantic import BaseModel, SecretStr

from dotenv import load_dotenv
import uvicorn # type: ignore

# Browser automation imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserConfig, Controller, ActionResult
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

app = FastAPI(title="Browser Automation Service")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Store active browser sessions
active_sessions = {}

class BrowserTask(BaseModel):
    task: str
    session_id: Optional[str] = None
    chrome_path: Optional[str] = None
    headless: bool = False
    max_steps: int = 50
    viewport_expansion: int = 0

class UserInputRequest(BaseModel):
    session_id: str
    question: str
    
class UserInputResponse(BaseModel):
    answer: str

# Store pending user input requests
pending_inputs = {}

@app.post("/initialize")
async def initialize_browser(config: BrowserTask):
    try:
        # Generate a session ID if none was provided
        session_id = config.session_id or f"session_{len(active_sessions) + 1}"
        
        # Check API keys
        if not GEMINI_API_KEY or not OPENROUTER_API_KEY:
            raise HTTPException(status_code=400, detail="Missing API keys for browser automation")
        
        # Set up the browser config
        browser_config = BrowserConfig(
            headless=config.headless,
            new_context_config=BrowserContextConfig(
                viewport_expansion=config.viewport_expansion,
            )
        )
        
        chrome_process = None
        # Handle Chrome launch based on OS
        import platform
        system_type = platform.system()
        
        if config.chrome_path:
            import subprocess
            import time
            
            # Kill existing Chrome processes
            try:
                print("Killing existing Chrome processes...")
                if system_type == "Windows":
                    subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                elif system_type == "Linux":
                    # For WSL or regular Linux
                    if "/mnt/" in config.chrome_path:  # WSL path
                        subprocess.run(["powershell.exe", "taskkill", "/F", "/IM", "chrome.exe"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        subprocess.run(["pkill", "-f", "chrome"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(2)
            except Exception as e:
                print(f"Failed to kill Chrome processes: {e}")
            
            # Launch Chrome with debugging port
            try:
                print(f"Launching Chrome from {config.chrome_path} with remote debugging...")
                user_data_dir = os.path.join(os.path.expanduser("~"), ".chrome-automation-data")
                os.makedirs(user_data_dir, exist_ok=True)
                
                chrome_args = [
                    config.chrome_path,
                    f"--remote-debugging-port=9222",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={user_data_dir}",
                    "about:blank"
                ]
                
                # Use appropriate subprocess creation based on OS
                if system_type == "Windows":
                    # Use CREATE_NO_WINDOW flag to prevent console window
                    from subprocess import CREATE_NO_WINDOW
                    chrome_process = subprocess.Popen(
                        chrome_args, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        creationflags=CREATE_NO_WINDOW
                    )
                else:
                    chrome_process = subprocess.Popen(
                        chrome_args, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                
                # Give Chrome time to start up
                time.sleep(5)
                
                # Configure Browser to connect to the already running instance
                browser_config.chrome_instance_path = None  # Don't start a new Chrome
                browser_config.playwright_cdp_url = "http://localhost:9222"
            except Exception as e:
                print(f"Failed to manually launch Chrome: {e}")
                traceback.print_exc()
        
        # Initialize controller with improved error handling
        controller = Controller()
        
        # Register the action for user input with error handling
        @controller.action('Ask user for information')
        def ask_human(question: str) -> str:
            try:
                pending_inputs[session_id] = question
                return ActionResult(extracted_content="Waiting for user input...")
            except Exception as e:
                print(f"Error in ask_human action: {e}")
                return ActionResult(extracted_content=f"Error asking for user input: {str(e)}")
        
        # Initialize browser with proper error handling
        try:
            browser = Browser(config=browser_config)
        except Exception as e:
            error_msg = f"Browser initialization failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Store session info
        active_sessions[session_id] = {
            "browser": browser,
            "controller": controller,
            "pending_task": None,
            "is_running": False,
            "chrome_process": chrome_process
        }
        
        return {"session_id": session_id, "status": "initialized"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Browser initialization error: {str(e)}")
        
@app.post("/run_task")
async def run_task(task: BrowserTask, background_tasks: BackgroundTasks):
    """Start a browser automation task"""
    try:
        session_id = task.session_id
        if not session_id or session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found. Initialize a browser first.")
        
        session = active_sessions[session_id]
        
        # Check if a task is already running
        if session["is_running"]:
            raise HTTPException(status_code=400, detail="A task is already running in this session")
        
        # Store task in session for asynchronous execution
        session["pending_task"] = task.task
        session["is_running"] = True
        
        # Run task in background
        background_tasks.add_task(execute_browser_task, session_id, task.task, task.max_steps)
        
        return {
            "session_id": session_id,
            "status": "task_started",
            "message": "Browser automation task started. Check status endpoint for updates."
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error starting task: {str(e)}")

async def execute_browser_task(session_id: str, task: str, max_steps: int = 50):
    """Execute a browser automation task with improved error handling"""
    try:
        if session_id not in active_sessions:
            print(f"Session {session_id} not found in active_sessions")
            return
            
        session = active_sessions[session_id]
        browser = session["browser"]
        controller = session["controller"]
        
        # Initialize LLMs with proper error handling
        try:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp', 
                api_key=SecretStr(GEMINI_API_KEY)
            )
            
            planner_llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model="google/gemma-3-27b-it:free",
                api_key=SecretStr(OPENROUTER_API_KEY),
            )
            
            # planner_llm=ChatGoogleGenerativeAI(
            #     model='gemini-2.0-flash-lite',
            #     api_key=SecretStr(GEMINI_API_KEY)
            # )

        except Exception as e:
            print(f"Error initializing LLMs: {e}")
            traceback.print_exc()
            session["is_running"] = False
            session["last_error"] = f"LLM initialization error: {str(e)}"
            return
        
        # Create the agent with timeout handling
        try:
            agent = Agent(
                task=task,
                llm=llm,
                max_actions_per_step=4,
                browser=browser,
                use_vision=True,
                planner_llm=planner_llm,
                controller=controller
            )
            
            # Run the agent with timeout protection
            import asyncio

            result = await asyncio.wait_for(
                agent.run(max_steps=max_steps), 
                timeout=600  # 10 minute timeout
            )
            
            # Update session status
            session["is_running"] = False
            session["last_result"] = result
            
            return result
        except asyncio.TimeoutError:
            error_msg = "Browser automation task timed out after 10 minutes"
            session["is_running"] = False
            session["last_error"] = error_msg
            print(error_msg)
            return {"error": error_msg}
        except Exception as e:
            session["is_running"] = False
            session["last_error"] = str(e)
            traceback.print_exc()
            return {"error": str(e)}
    except Exception as e:
        # Update session status even if there's an error
        if session_id in active_sessions:
            active_sessions[session_id]["is_running"] = False
            active_sessions[session_id]["last_error"] = str(e)
        traceback.print_exc()
        raise e

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get the status of a browser session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Check if there's a pending input question
    pending_question = pending_inputs.get(session_id)
    
    return {
        "session_id": session_id,
        "is_running": session["is_running"],
        "pending_task": session["pending_task"],
        "needs_input": pending_question is not None,
        "pending_question": pending_question,
        "last_result": session.get("last_result"),
        "last_error": session.get("last_error")
    }

@app.post("/provide_input/{session_id}")
async def provide_input(session_id: str, response: UserInputResponse):
    """Provide user input for a pending question"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_id not in pending_inputs:
        raise HTTPException(status_code=400, detail="No pending input request for this session")
    
    # Update the controller's action to return the provided answer
    controller = active_sessions[session_id]["controller"]
    
    @controller.action('Ask user for information')
    def ask_human(question: str) -> str:
        return ActionResult(extracted_content=response.answer)
    
    # Clear the pending input
    del pending_inputs[session_id]
    
    return {"status": "input_provided", "session_id": session_id}

@app.post("/close/{session_id}")
async def close_session(session_id: str):
    """Close a browser session with improved cleanup"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = active_sessions[session_id]
        browser = session["browser"]
        
        # Close the browser
        await browser.close()
        
        # Terminate Chrome process with proper error handling
        if session.get("chrome_process"):
            try:
                import platform
                system_type = platform.system()
                
                # Use appropriate termination method based on OS
                if system_type == "Windows":
                    import ctypes
                    # Use handle to terminate process more reliably on Windows
                    process = session["chrome_process"]
                    if process.poll() is None:  # Process is still running
                        process.terminate()
                        # Give it a moment to terminate gracefully
                        import time
                        time.sleep(1)
                        # If still running, kill forcefully
                        if process.poll() is None:
                            process.kill()
                else:
                    session["chrome_process"].terminate()
            except Exception as e:
                print(f"Error terminating Chrome process: {e}")
        
        # Remove the session
        del active_sessions[session_id]
        
        # Clean up any pending inputs
        if session_id in pending_inputs:
            del pending_inputs[session_id]
        
        return {"status": "closed", "session_id": session_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error closing session: {str(e)}")
    
    
@app.on_event("shutdown")
async def shutdown_event():
    """Close all browser sessions when shutting down"""
    for session_id in list(active_sessions.keys()):
        try:
            await active_sessions[session_id]["browser"].close()
            # Terminate Chrome process if we launched it manually
            if active_sessions[session_id].get("chrome_process"):
                try:
                    active_sessions[session_id]["chrome_process"].terminate()
                except:
                    pass
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("browser_service:app", host="0.0.0.0", port=8000, reload=False)
