import asyncio
import os

from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI  
from pydantic import SecretStr 

from browser_use import Agent, BrowserConfig, Controller, ActionResult
from browser_use.browser.browser import Browser 
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
planner_llm = ChatOpenAI(model='o3-mini')

# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)

browser = Browser(
	config=BrowserConfig(
		chrome_instance_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
		)
	)
)


async def run_search():
	agent = Agent(
		task="""
  				Give me a summary of the last five gmails.
      		""",
		llm=llm,
		max_actions_per_step=4,
		browser=browser,
		use_vision=True,
		planner_llm=planner_llm,
  		controller=controller
	)

	await agent.run(max_steps=50);     await browser.close() 
    

if __name__ == '__main__':
	asyncio.run(run_search())