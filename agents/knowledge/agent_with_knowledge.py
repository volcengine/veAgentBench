import logging

# from readline import backend

from veadk import Agent, Runner
from veadk.knowledgebase.knowledgebase import KnowledgeBase

import os

from agentkit.apps import AgentkitSimpleApp

logger = logging.getLogger(__name__)


app = AgentkitSimpleApp()

# Knowledgebase usage
# Required env vars for viking knowledgebase
# Local VOLCENGINE_ACCESS_KEY, VOLCENGINE_ACCESS_KEY,
# Clooud: ServiceRole with VikingdbFullAccess permission
my_knowledge_type = os.getenv("DATABASE_TYPE", "viking")
my_knowledge_collection = os.getenv("DATABASE_COLLECTION", "")
if not my_knowledge_collection:
    raise ValueError("DATABASE_COLLECTION environment variable is required")

knowledgebase = KnowledgeBase(backend=my_knowledge_type, index=my_knowledge_collection, top_k=5)


instruction = "你是一个知识渊博的助手，请利用知识库回答问题。如果知识库中无相关内容，回答“未找到相关内容”"

# switch LTM backend
agent = Agent(
    model_name="doubao-seed-1-6-251015",
    instruction=instruction,
    knowledgebase=knowledgebase,
)
runner = Runner(agent=agent)


@app.entrypoint
async def run(payload: dict, headers: dict) -> str:
    prompt = payload.get("prompt", "")
    user_id = headers["user_id"]
    session_id = headers["session_id"]

    logger.info(
        f"Running agent with prompt: {prompt}, user_id: {user_id}, session_id: {session_id}"
    )

    response = await runner.run(messages=prompt, user_id=user_id, session_id=session_id)

    logger.info(f"Run response: {response}")
    return response


@app.ping
def ping() -> str:
    return "pong!"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
