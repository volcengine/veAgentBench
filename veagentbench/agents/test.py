import asyncio
import time

from veadk import Agent, Runner
from veadk.tools.builtin_tools.vesearch import vesearch
from veadk.knowledgebase.knowledgebase import KnowledgeBase
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer
from utils.data_loader import load_law_prompts
from veagentbench.agents.tracer import VeOpentelemetryTracer
from veagentbench.agents.base_agent import TestAgent
knowledgebase = KnowledgeBase( backend="viking")
exporters = [APMPlusExporter()]
# tracer = VeOpentelemetryTracer(trace_folder="./trace")
# tracer = OpentelemetryTracer()
agent = Agent(instruction="你是一个法律援助的客服agent，当收到问题时请优先从知识库中检索答案,如果没有检索到请联网搜索",
              tools=[vesearch],
              knowledgebase=knowledgebase,
             )

testagent = TestAgent(
    session_id_base="legal_aid_agent_test",
    agent_entity=agent,
)
if __name__ == '__main__':
    prompts = load_law_prompts()
    result = asyncio.run(testagent.generate(messages=prompts[0]))
    print(result.response, result.trace_file_path)