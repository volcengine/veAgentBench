from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from uuid import uuid4
import json
import requests
import time
from pydantic import BaseModel, Field
from veadk.utils.logger import get_logger
import traceback
from veagentbench.agents.base_agent import BaseAgent, AgentOutPut
from veagentbench.utils.tool_result_success import is_tool_execution_success
# Set up logging
logger = get_logger(__name__)

# Try to import Event from google.adk.events, with fallback

from google.adk.events import Event





class AdkAgent(BaseAgent):
    
    def __init__(
        self,
        end_point: str,
        api_key: str,
        api_server: Optional[str] = None,
        agent_name: Optional[str] = 'agent',
                 
    ):
        self.agent_name = agent_name
        self.end_point = end_point
        self.api_key = api_key
        self.api_server = api_server

    def get_session(self):
        if self.api_server == None:
            return str(uuid4())

    async def generate_output(
        self,
        prompt: str,
        user_id: str,
        stream: bool=True,
        **kwargs
    ):
        """Generate output with optional additional parameters from kwargs."""
        session_id = self.get_session()
        
        # Handle additional parameters from kwargs
        custom_headers = kwargs.get('headers', {})
        custom_data = kwargs.get('data', {})
        
        HEADERS={
            "user_id": user_id,
            "session_id": session_id,
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json; charset=UTF-8",
            **custom_headers
        }
        DATA = {
            "prompt": prompt,
            **custom_data
        }


        tool_called = {}
        full_response_text = ''
        final_text = ''
        first_token_duration = 0.0
        end2end_duration = 0.0

        time_start = time.time()
        try:
            with requests.post(
                self.end_point, data=json.dumps(DATA), headers=HEADERS, stream=stream
            ) as r:
                for chunk in r.iter_lines():  
                    if not chunk:
                        continue
                    if first_token_duration == 0.0:
                        first_token_duration = (time.time() - time_start)
                    json_string = chunk.decode("utf-8").removeprefix("data: ").strip()
                    json_string = json.loads(json_string)
                    # Handle different event formats
                    if json_string.startswith("data: "):
                        event_data = json.loads(json_string.removeprefix("data: "))
                    else:
                        event_data = json.loads(json_string)

                    event = Event(**event_data)

                    calls = event.get_function_calls()
                    if calls:
                        for call in calls:
                            tool_name = call.name
                            arguments = call.args # This is usually a dictionary
                            id = call.id
                            tool_called[id] = {
                                "name": tool_name,
                                "input_parameters": arguments,
                            }
                    
                    responses = event.get_function_responses()
                    if responses:
                        for response in responses:
                            tool_name = response.name
                            result_dict = response.response # The dictionary returned by the tool
                            id = response.id
                            if id in tool_called:
                                tool_called[id]["output"] = result_dict
                                tool_called[id]["success"] = is_tool_execution_success(result_dict)

                    if event.partial and event.content and event.content.parts and event.content.parts[0].text:
                        full_response_text += event.content.parts[0].text
                    
                    if event.is_final_response():
                        # print("\n--- Final Output Detected ---")
                        end2end_duration = (time.time() - time_start)
                    
                        if event.content and event.content.parts and event.content.parts[0].text:
                            # If it's the final part of a stream, use accumulated text
                            final_text = full_response_text + (event.content.parts[0].text if not event.partial else "")
                            # print(f"Display to user: {final_text.strip()}")
                            full_response_text = "" # Reset accumulator
                        elif event.actions and event.actions.skip_summarization and event.get_function_responses():
                            # Handle displaying the raw tool result if needed
                            response_data = event.get_function_responses()[0].response
                            print(f"Display raw tool result: {response_data}")
                        elif hasattr(event, 'long_running_tool_ids') and event.long_running_tool_ids:
                            
                            print("Display message: Tool is running in background...")
                        else:
                            
                            # Handle other types of final responses if applicable
                            print("Display: Final non-textual response or signal.")
            
            return AgentOutPut(
                first_token_duration = first_token_duration,
                end2end_duration = end2end_duration,
                final_response = final_text,
                tool_called = tool_called,
                success = True
            )
        
        except Exception as err:
            logger.error(f"Error in generate_output: {err}")
            traceback.print_exc()
            return AgentOutPut(
                success = False
            )



if __name__ == "__main__":
    # 测试AdkAgent
    agent = AdkAgent(
        end_point="http://127.0.0.1:8000/invoke",
        api_key= "IRa2yiORMmZqoCBmn_bn5hZJ_LVtcLSYdvBT"

    )
    import asyncio
    res =  asyncio.run(agent.generate_output(
        prompt="查询今日股票市场",
        user_id="test01",
    ))
    print(res.model_dump_json(indent=2))
    
    # 测试LocalAdkAgent
    print("\n" + "="*50 + "\n")
    print("测试LocalAdkAgent...")
    # local_agent = LocalAdkAgent(agent_file_path="agents/financial_analysis_agent.py")
    # res = asyncio.run(local_agent.generate_output(
    #     prompt="查询今日股票市场",
    #     user_id="test01",
    # ))
    # print(res.model_dump_json(indent=2))
