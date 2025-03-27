import asyncio
import os

from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_openai import ChatOpenAI
from pydantic import SecretStr 

from browser_use import Agent, BrowserConfig, Controller, ActionResult
from browser_use.browser.browser import Browser 
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
gemma_key = os.getenv('OPENROUTER_API_KEY')

if not api_key and not gemma_key:
	raise ValueError('API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

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
  				Goto gmail and propose a interview confirmation to hariisankar.s2022ai-ds@sece.ac.in. Ensure the mail is sent by verifying it in the sent box.
      		""",
		llm=llm,
		max_actions_per_step=4,
		browser=browser,
		use_vision=True,
		planner_llm = ChatOpenAI(
			base_url="https://openrouter.ai/api/v1",
			model="google/gemma-3-27b-it:free",
			api_key=SecretStr(gemma_key),
		)
,
  		controller=controller
	)

	await agent.run(max_steps=50);     
 	await browser.close() 
    

if __name__ == '__main__':
	asyncio.run(run_search())