from .tracer import VeOpentelemetryTracer as Tracer
from veadk import Agent, Runner

from veadk.runner import RunnerMessage
import asyncio
from pydantic import BaseModel
from veadk.utils.logger import get_logger
from typing import List, Optional
from veagentbench.agents.tracer import VeOpentelemetryTracer
from veagentbench.agents.consts import DEFAULT_TRACE_DIR
logger = get_logger(__name__)
class TestResut(BaseModel):
    success: bool
    response: str
    trace_file_path: Optional[str]=''
    trace_data: List[dict]

class TestAgent:

    # The agent entity to be tested, support veadk.Agent
    agent_entity: Agent = None

    # tracer to be used, default to VeOpentelemetryTracer. If you want to customize the tracer path, please set it before pre_test_init
    tracer: Tracer = None

    #trace folder path, default to ./trace. If you want to customize the trace folder path, please set it before pre_test_init
    trace_dir: str = DEFAULT_TRACE_DIR

    # session id base, will be used to generate session id for each test case
    session_id_base: str = "test_agent"

    # tools used by the agent, will be set in pre_test_init
    tools: list = []

    # agent runtime, will be set in pre_test_init
    agent_runtime: Runner = None

    def __init__(self,    
            agent_entity: Agent = None,
            tracer: Tracer = None,
            trace_dir: str = DEFAULT_TRACE_DIR,
            session_id_base: str = "test_agent",
        ):
        self.agent_entity = agent_entity
        self.trace_dir = trace_dir
        if tracer == None:
            self.tracer = VeOpentelemetryTracer(trace_folder=trace_dir)
        else:
            self.tracer = tracer
        self.session_id_base = session_id_base
        self.agent_runtime = None
        self.pre_test_init()
        
    def get_agent_id(self):
        
        
        return 
    
    def validate(self):
        '''
            Validate if the agent is properly initialized.
        '''
        if not self.agent_entity:
            raise ValueError("Agent entity is not initialized.")
    
    def pre_test_init(self):
        '''
            Pre-test initialization, validate the agent and set up the tracer and runner.
        '''
        self.validate()
        self.tracer.set_trace_folder(self.trace_dir)
        self.tools = self.agent_entity.tools if self.agent_entity else []
        self.trace_dir = self.tracer.get_trace_folder() if self.tracer else DEFAULT_TRACE_DIR
        self.agent_entity.tracers = [self.tracer] if self.tracer else DEFAULT_TRACER

        self.agent_runtime = Runner(
            agent=self.agent_entity
        )

    async def generate(self, messages: RunnerMessage):
        """
            Asynchronously generate a response from the agent through the agent runtime.

            Args:
                messages (RunnerMessage): The input messages for the agent.
        """
        def get_session_id():
            import uuid
            return f"{self.session_id_base}_{str(uuid.uuid4())[:8]}"
        try:
            session_id = get_session_id()
            logger.info('strat generate response: %s'%session_id)
            response = await self.agent_runtime.run(messages=messages, session_id=session_id, save_tracing_data=True)
            print(f"Tracing file path: {self.tracer._trace_file_path}")
            
            trace_data = self.tracer.get_spans(session_id=session_id)
            # logger.debug(trace_data)
            logger.info('complete generate: %s'%session_id)
            return TestResut(success=True, response=response, trace_file_path="", trace_data=trace_data)
        except Exception as e:
            return TestResut(success=False, response=str(e), trace_file_path="")

