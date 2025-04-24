# browser_client.py
import requests
import time
from typing import Optional, Dict, Any, Callable

class BrowserAutomationClient:
    """Client for the Browser Automation Microservice"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.polling_interval = 1  # seconds
    
    def initialize_browser(self, chrome_path=None, headless=False, viewport_expansion=0):
        """Initialize a new browser session with better error handling"""
        try:
            response = requests.post(
                f"{self.base_url}/initialize",
                json={
                    "task": "",  # Empty task for initialization
                    "chrome_path": chrome_path,
                    "headless": headless,
                    "viewport_expansion": viewport_expansion
                },
                timeout=60  # Longer timeout for initialization
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                return True
            else:
                try:
                    error_msg = response.json().get("detail", "Unknown error")
                except:
                    error_msg = f"HTTP error {response.status_code}"
                print(f"Error initializing browser: {error_msg}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Connection error: Unable to connect to browser service at {self.base_url}")
            print("Make sure the browser service is running")
            return False
        except Exception as e:
            print(f"Exception during browser initialization: {str(e)}")
            return False
    
    def run_task(self, task, max_steps=50, input_callback: Optional[Callable[[str], str]] = None):
        """Run a browser automation task and handle any user input requests"""
        if not self.session_id:
            print("No active browser session. Call initialize_browser() first.")
            return {"error": "No active browser session"}
            
        try:
            # Start the task
            response = requests.post(
                f"{self.base_url}/run_task",
                json={
                    "session_id": self.session_id,
                    "task": task,
                    "max_steps": max_steps
                }
            )
            
            if response.status_code != 200:
                error_msg = response.json().get("detail", "Unknown error")
                return {"error": f"Failed to start task: {error_msg}"}
            
            # Poll for status and handle input requests
            while True:
                time.sleep(self.polling_interval)
                
                status = self.get_status()
                
                # If task is not running and there's a result or error, we're done
                if not status.get("is_running"):
                    if "last_error" in status and status["last_error"]:
                        return {"error": status["last_error"]}
                    else:
                        # Clean up the result before returning
                        raw_result = status.get("last_result", "Task completed")
                        # Process the result to remove large binary data and format it nicely
                        clean_result = self._clean_result(raw_result)
                        return {"result": clean_result}
                
                # If input is needed and we have a callback
                if status.get("needs_input") and input_callback:
                    question = status.get("pending_question", "Input needed:")
                    user_answer = input_callback(question)
                    
                    # Provide the input back to the service
                    self.provide_input(user_answer)
        except Exception as e:
            return {"error": f"Error running task: {str(e)}"}

    def _clean_result(self, result):
        """Clean the result data by removing large binary content and formatting it for display"""
        if isinstance(result, dict):
            # If it's a dictionary, extract key info without large binary data
            if 'history' in result:
                # Format the history in a more readable way
                summary = []
                for step in result['history']:
                    # Extract actions performed
                    actions = []
                    if 'model_output' in step and 'action' in step['model_output']:
                        for action in step['model_output']['action']:
                            for action_type, details in action.items():
                                if action_type == 'go_to_url' and 'url' in details:
                                    actions.append(f"Navigated to {details['url']}")
                                elif action_type in ['click', 'type', 'submit', 'select']:
                                    actions.append(f"Performed {action_type} action")
                    
                    # Extract result information
                    results = []
                    if 'result' in step:
                        for res in step['result']:
                            if 'extracted_content' in res:
                                results.append(res['extracted_content'])
                    
                    if actions or results:
                        step_summary = "• " + ", ".join(actions)
                        if results:
                            step_summary += f" → {' '.join(results)}"
                        summary.append(step_summary)
                
                return "Task completed with the following steps:\n" + "\n".join(summary)
            
            # For simpler result formats
            return str(result)
        
        # If it's a string, just return it
        return str(result)
    
    def get_status(self):
        """Get the current status of the browser session"""
        if not self.session_id:
            return {"error": "No active browser session"}
            
        try:
            response = requests.get(f"{self.base_url}/status/{self.session_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.json().get("detail", "Unknown error")}
        except Exception as e:
            return {"error": f"Error getting status: {str(e)}"}
    
    def provide_input(self, answer):
        """Provide user input for a pending question"""
        if not self.session_id:
            return False
            
        try:
            response = requests.post(
                f"{self.base_url}/provide_input/{self.session_id}",
                json={"answer": answer}
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error providing input: {str(e)}")
            return False
    
    def close_browser(self):
        """Close the browser session"""
        if not self.session_id:
            return True
            
        try:
            response = requests.post(f"{self.base_url}/close/{self.session_id}")
            if response.status_code == 200:
                self.session_id = None
                return True
            else:
                return False
        except Exception as e:
            print(f"Error closing browser: {str(e)}")
            return False
