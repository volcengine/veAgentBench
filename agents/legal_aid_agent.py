import asyncio
import time

from veadk import Agent, Runner
from veadk.tools.builtin_tools.vesearch import vesearch
from veadk.knowledgebase.knowledgebase import KnowledgeBase
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer
from utils.data_loader import load_law_prompts
from veagentbench.agents.tracer import VeOpentelemetryTracer

knowledgebase = KnowledgeBase( backend="viking")
exporters = [APMPlusExporter()]
tracer = VeOpentelemetryTracer(trace_folder="./trace")
# tracer = OpentelemetryTracer()
agent = Agent(instruction="你是一个法律援助的客服agent，当收到问题时请优先从知识库中检索答案,如果没有检索到请联网搜索",
              tools=[vesearch],
              knowledgebase=knowledgebase,
              tracers=[tracer])

if __name__ == '__main__':
    session_id_base = "knowledgebase_test2"
    prompts = load_law_prompts()
    runner = Runner(
        agent=agent
    )
    for i in range(1):
        time.sleep(3)
        session_id = f"{session_id_base}_{i}"
        response = asyncio.run(runner.run(messages=prompts[i], session_id=session_id, save_tracing_data=True))
        print(response)
        print(f"Tracing file path: {tracer._trace_file_path}")
        dump_path = asyncio.run(runner.save_eval_set(session_id=session_id))
        print(dump_path)