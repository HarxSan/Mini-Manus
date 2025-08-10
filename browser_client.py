# browser_client.py
import requests
import time
import logging
import json
import websocket
import threading
import queue
import backoff
from typing import Optional, Dict, Any, Callable, Union

class BrowserAutomationClient:
    """
    Enhanced client for the Browser Automation Microservice with:
    - Retry logic for API calls
    - Improved error handling and reporting
    - WebSocket support for real-time updates
    - Proper logging system
    """
    
    def __init__(self, base_url="http://localhost:8000", log_level=logging.INFO):
        """
        Initialize the browser automation client.
        
        Args:
            base_url: Base URL for the browser service API
            log_level: Logging level (default: INFO)
        """
        self.base_url = base_url
        self.ws_base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.session_id = None
        self.polling_interval = 1  # seconds
        self.websocket = None
        self.ws_connected = False
        self.ws_message_queue = queue.Queue()
        self.ws_thread = None
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Requests session with retry capability
        self.session = requests.Session()
        
        self.logger.info("Browser Automation Client initialized")
    
    def _setup_logger(self, log_level):
        """Set up a logger with the specified log level."""
        logger = logging.getLogger("BrowserAutomationClient")
        if not logger.handlers:  # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)
        return logger
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=3,
        factor=2
    )
    def _make_api_request(self, method, endpoint, json_data=None, timeout=30):
        """
        Make API request with retry logic using backoff decorator.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            json_data: JSON data to send
            timeout: Request timeout in seconds
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint}"
        self.logger.debug(f"Making {method} request to {url}")
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response
        except requests.exceptions.HTTPError as e:
            # For HTTP errors, try to extract error message from response
            try:
                error_detail = e.response.json().get("detail", str(e))
                self.logger.error(f"HTTP error: {error_detail}")
                raise requests.exceptions.HTTPError(f"HTTP error: {error_detail}") from e
            except (ValueError, KeyError):
                self.logger.error(f"HTTP error: {str(e)}")
                raise
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Request timeout: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request exception: {str(e)}")
            raise
    
    def initialize_browser(self, chrome_path=None, headless=False, viewport_expansion=0):
        """
        Initialize a new browser session with better error handling and retry logic.
        
        Args:
            chrome_path: Path to Chrome executable
            headless: Whether to run Chrome in headless mode
            viewport_expansion: Amount to expand the viewport by
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Initializing browser session")
        
        try:
            request_data = {
                "task": "",  # Empty task for initialization
                "chrome_path": chrome_path,
                "headless": headless,
                "viewport_expansion": viewport_expansion
            }
            
            self.logger.debug(f"Initialization parameters: {request_data}")
            response = self._make_api_request(
                "POST",
                "initialize",
                json_data=request_data,
                timeout=60  # Longer timeout for initialization
            )
            
            data = response.json()
            self.session_id = data["session_id"]
            self.logger.info(f"Browser initialized successfully with session ID: {self.session_id}")
            
            # Start WebSocket connection after successful initialization
            self._setup_websocket_connection()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {str(e)}")
            return False
    
    def _setup_websocket_connection(self):
        """Set up WebSocket connection for real-time updates."""
        if not self.session_id:
            self.logger.warning("Cannot setup WebSocket: No active session ID")
            return
        
        # Close existing connection if any
        self._close_websocket()
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(
            target=self._websocket_thread,
            daemon=True
        )
        self.ws_thread.start()
    
    def _websocket_thread(self):
        """WebSocket connection thread."""
        ws_url = f"{self.ws_base_url}/ws/{self.session_id}"
        self.logger.info(f"Connecting to WebSocket: {ws_url}")
        
        try:
            # Define WebSocket callbacks
            def on_message(ws, message):
                try:
                    msg_data = json.loads(message)
                    self.logger.debug(f"WebSocket message received: {msg_data}")
                    self.ws_message_queue.put(msg_data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON: {message}")
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
                self.ws_connected = False
            
            def on_open(ws):
                self.logger.info("WebSocket connection established")
                self.ws_connected = True
            
            # Create WebSocket connection
            self.websocket = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection (this will block until closed)
            self.websocket.run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket thread error: {str(e)}")
            self.ws_connected = False
    
    def _close_websocket(self):
        """Close the WebSocket connection."""
        if self.websocket:
            self.logger.info("Closing existing WebSocket connection")
            self.websocket.close()
            self.websocket = None
            self.ws_connected = False
            
            # Wait for thread to finish if it exists
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
                self.ws_thread = None
    
    def run_task(self, task, max_steps=50, input_callback: Optional[Callable[[str], str]] = None):
        """
        Run a browser automation task and handle any user input requests.
        
        Args:
            task: Task description
            max_steps: Maximum number of steps to run
            input_callback: Callback function for user input requests
            
        Returns:
            Dict with result or error
        """
        if not self.session_id:
            self.logger.error("No active browser session. Call initialize_browser() first.")
            return {"error": "No active browser session"}
            
        self.logger.info(f"Running task: {task}")
        
        try:
            # Start the task
            response = self._make_api_request(
                "POST",
                "run_task",
                json_data={
                    "session_id": self.session_id,
                    "task": task,
                    "max_steps": max_steps
                }
            )
            
            task_info = response.json()
            self.logger.debug(f"Task started: {task_info}")
            
            # Check if WebSocket is available for status updates
            if self.ws_connected:
                return self._monitor_task_websocket(input_callback)
            else:
                # Fall back to polling if WebSocket is not available
                self.logger.info("WebSocket not connected, using polling for task status")
                return self._monitor_task_polling(input_callback)
        except Exception as e:
            error_message = f"Error running task: {str(e)}"
            self.logger.error(error_message)
            return {"error": error_message}
    
    def _monitor_task_websocket(self, input_callback):
        """Monitor task progress using WebSocket connection."""
        self.logger.info("Monitoring task via WebSocket")
        
        try:
            # Process messages from WebSocket until task completes
            while True:
                try:
                    # Get message with timeout
                    message = self.ws_message_queue.get(timeout=60)
                    
                    # Process message based on its type
                    if message.get("type") == "status_update":
                        status = message.get("status", {})
                        
                        # Check if task has completed
                        if not status.get("is_running", True):
                            if "last_error" in status and status["last_error"]:
                                self.logger.error(f"Task error: {status['last_error']}")
                                return {"error": status["last_error"]}
                            else:
                                result = self._clean_result(status.get("last_result", "Task completed"))
                                self.logger.info("Task completed successfully")
                                return {"result": result}
                        
                        # Check if input is needed
                        if status.get("needs_input") and input_callback:
                            question = status.get("pending_question", "Input needed:")
                            self.logger.info(f"User input requested: {question}")
                            
                            user_answer = input_callback(question)
                            self.logger.debug(f"User provided input: {user_answer}")
                            
                            # Provide the input back to the service
                            self.provide_input(user_answer)
                    
                    elif message.get("type") == "task_complete":
                        result = self._clean_result(message.get("result", "Task completed"))
                        self.logger.info("Task completed (WebSocket notification)")
                        return {"result": result}
                    
                    elif message.get("type") == "task_error":
                        error_msg = message.get("error", "Unknown error")
                        self.logger.error(f"Task error (WebSocket): {error_msg}")
                        return {"error": error_msg}
                        
                except queue.Empty:
                    # If no messages for 60 seconds, check status via API
                    self.logger.warning("No WebSocket messages received for 60 seconds, checking status via API")
                    status = self.get_status()
                    
                    if not status.get("is_running", False):
                        if "last_error" in status and status["last_error"]:
                            return {"error": status["last_error"]}
                        else:
                            result = self._clean_result(status.get("last_result", "Task completed"))
                            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error monitoring task via WebSocket: {str(e)}")
            # Fall back to polling on WebSocket failure
            return self._monitor_task_polling(input_callback)
    
    def _monitor_task_polling(self, input_callback):
        """Monitor task progress using polling."""
        self.logger.info("Monitoring task via polling")
        
        try:
            poll_count = 0
            max_polls = 600  # 10 minutes with 1-second interval
            
            # Poll for status and handle input requests
            while poll_count < max_polls:
                time.sleep(self.polling_interval)
                poll_count += 1
                
                status = self.get_status()
                if isinstance(status, dict) and "error" in status:
                    self.logger.error(f"Error getting status: {status['error']}")
                    continue
                
                # If task is not running and there's a result or error, we're done
                if not status.get("is_running"):
                    if "last_error" in status and status["last_error"]:
                        self.logger.error(f"Task failed: {status['last_error']}")
                        return {"error": status["last_error"]}
                    else:
                        # Clean up the result before returning
                        raw_result = status.get("last_result", "Task completed")
                        # Process the result to remove large binary data and format it nicely
                        clean_result = self._clean_result(raw_result)
                        self.logger.info("Task completed successfully")
                        return {"result": clean_result}
                
                # If input is needed and we have a callback
                if status.get("needs_input") and input_callback:
                    question = status.get("pending_question", "Input needed:")
                    self.logger.info(f"User input requested: {question}")
                    
                    user_answer = input_callback(question)
                    self.logger.debug(f"User provided input: {user_answer}")
                    
                    # Provide the input back to the service
                    self.provide_input(user_answer)
            
            # If we reach this point, we've exceeded the maximum polling attempts
            error_msg = "Task monitoring timed out after 10 minutes"
            self.logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error monitoring task: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def _clean_result(self, result):
        """
        Clean the result data by removing large binary content and formatting it for display.
        
        Args:
            result: Raw result data
            
        Returns:
            Cleaned result data
        """
        try:
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
                                        element_desc = ""
                                        if 'element' in details:
                                            if 'text' in details['element']:
                                                element_desc = f" on '{details['element']['text']}'"
                                            elif 'selector' in details['element']:
                                                element_desc = f" on {details['element']['selector']}"
                                        actions.append(f"Performed {action_type}{element_desc}")
                        
                        # Extract result information
                        results = []
                        if 'result' in step:
                            for res in step['result']:
                                if 'extracted_content' in res:
                                    results.append(res['extracted_content'])
                        
                        if actions or results:
                            step_summary = "• " + ", ".join(actions) if actions else "• Action performed"
                            if results:
                                step_summary += f" → {' '.join(results)}"
                            summary.append(step_summary)
                    
                    return "Task completed with the following steps:\n" + "\n".join(summary)
                
                # For simpler result formats, convert to string without large data
                clean_dict = {}
                for key, value in result.items():
                    # Skip large binary data fields
                    if key in ['screenshot', 'html', 'full_html']:
                        clean_dict[key] = "[large data omitted]"
                    else:
                        clean_dict[key] = value
                return json.dumps(clean_dict, indent=2)
            
            # If it's a string, just return it
            return str(result)
        except Exception as e:
            self.logger.error(f"Error cleaning result: {str(e)}")
            return str(result)
    
    def get_status(self):
        """
        Get the current status of the browser session.
        
        Returns:
            Dict containing session status
        """
        if not self.session_id:
            self.logger.warning("Cannot get status: No active browser session")
            return {"error": "No active browser session"}
            
        try:
            self.logger.debug(f"Getting status for session {self.session_id}")
            response = self._make_api_request("GET", f"status/{self.session_id}")
            return response.json()
        except Exception as e:
            error_msg = f"Error getting status: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def provide_input(self, answer):
        """
        Provide user input for a pending question.
        
        Args:
            answer: User's answer to the pending question
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.session_id:
            self.logger.warning("Cannot provide input: No active browser session")
            return False
            
        try:
            self.logger.info(f"Providing user input for session {self.session_id}")
            response = self._make_api_request(
                "POST",
                f"provide_input/{self.session_id}",
                json_data={"answer": answer}
            )
            
            self.logger.debug("Input provided successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error providing input: {str(e)}")
            return False
    
    def close_browser(self):
        """
        Close the browser session.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.session_id:
            self.logger.info("No active browser session to close")
            return True
            
        try:
            self.logger.info(f"Closing browser session {self.session_id}")
            
            # Close WebSocket connection first
            self._close_websocket()
            
            # Close browser session
            response = self._make_api_request("POST", f"close/{self.session_id}")
            
            if response.status_code == 200:
                self.logger.info("Browser session closed successfully")
                self.session_id = None
                return True
            else:
                self.logger.error("Failed to close browser session")
                return False
        except Exception as e:
            self.logger.error(f"Error closing browser: {str(e)}")
            return False