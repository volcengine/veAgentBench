import os
from veadk import Agent
from veadk.memory.long_term_memory import LongTermMemory
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer


app_name = "long_memeval_app"
long_term_memory = LongTermMemory(
    backend="viking",
    app_name=app_name,
    index=os.getenv("DATABASE_VIKINGMEM_COLLECTION"),
    top_k=10,
)
exporters = [APMPlusExporter()]
tracer = OpentelemetryTracer()
agent = Agent(
    name="longmemeval_test_agent",
    instruction="""
                          Your task is to briefly answer the question. If you don't know how to answer the question, abstain from answering.Befor answer question below, you must use load_memory tool to search relevant memory and profile, the seary query should be a combination of  <Current Date> and <Question> content, like: query: "Current Date: xxxx\nQuestion: xxxx". 
                          """,
    long_term_memory=long_term_memory,
    tracers=[tracer],
)
