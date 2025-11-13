
## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http:##www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

from veagentbench.task.base import BaseTask
from veagentbench.metrics.mcp_bench import MCPToolMetric
from veagentbench.metrics.answer_correctness._answer_correctness import AnswerCorrectnessMetric
import pandas as pd
from veagentbench.evals.deepeval.test_case import LLMTestCase, MCPServer, ToolCall
from veagentbench.utils.analyze_trace import *
from typing import List, Union
from veagentbench.utils.eval_set import get_output_from_eval_case
from veagentbench.models.models import VolceOpenAI
from veagentbench.evals.deepeval import evaluate
import os
from veagentbench.evals.deepeval.evaluate import AsyncConfig, ErrorConfig, CacheConfig
from veagentbench.models.models import VolceOpenAI
from veagentbench.test_case import AgentTestCase
from veagentbench.metrics.mcp_bench.schema import ToolExecutionResult, MCPExecutionData
from veagentbench.utils.extract_expected_tool_calls import parse_expected_tool_calls, turn_tool_dict2toolcall
from veagentbench.utils.tool_result_success import is_tool_execution_success
from veadk import Agent, Runner
from veagentbench.agents.base_agent import TestAgent
from veadk.utils.logger import get_logger
import asyncio
from tqdm import tqdm
from veagentbench.agents.base_agent import TestResut
from pathlib import Path
import hashlib
import time
import json
logger = get_logger(__name__)

INDEX_COLUMN = 'index'
INPUT_COLUMN = 'input'
EXPECTED_TOOLS_RESULT_COLUMN = 'expect_tools_result'
EXPECTED_TOOLS_COLUMN = 'expect_tools_calls'
EXPECTED_OUTPUT_COLUMN = 'expect_output'
TRACE_ID_COLUMN = 'trace_id'
EVAL_SET_ID_COLUMN = 'eval_set_id'




def turn_toolcalls(tool_calls) ->  List[ToolCall]:
    tool_use = []
    for call in tool_calls:
        tool_name = call.get('tool_name', 'unknown_tool')
        parameters = call.get('input_params', {})
        output_result = call.get('output_result', {})
        if tool_name in ['load_knowledgebase', '(merged tools)']:
            continue
        tool_use.append(ToolExecutionResult(
            name=tool_name,
            input_parameters=parameters,
            output=output_result,
            description=call.get('description', ''),
            # 成功判断：基于结果内容与错误信号进行更稳健的判断
            server=call.get('server', 'default'),
            success=is_tool_execution_success(output_result)
        ))
        
    return tool_use


def turn_avalible_tools(tools_dict):
    avalible_tool_use = {}
    for avalible_tool in tools_dict:
        if isinstance(avalible_tool, dict):
            tool_name = avalible_tool.get('name', 'unknown_tool')
            if tool_name not in  ['load_knowledgebase', '(merged tools)']:
                avalible_tool_use[tool_name] = ToolCall(
                    name=avalible_tool.get('name', 'unknown_tool'),
                    input_parameters=avalible_tool.get('parameters', {}),
                    description=avalible_tool.get('description', ''),
                )

    return avalible_tool_use

class MCPTaskOffline(BaseTask):

    def __init__(self, task_name: str, agents: List[Agent]):
        super().__init__(task_name)
        self.metrics = []
        self.dataset = None
        self.output_dir = None
        self.agents = agents
        # self.llm_mode = VolceOpenAI(model="ep-20250618145117-ps5rq")
        self.llm_model = VolceOpenAI(
            # model="ep-20250618145117-ps5rq",
            model = os.environ.get("VOLCEMODEL"),
            temperature=0,
            base_url=os.environ.get("VOLCEBASEURL"),
            cost_per_input_token=0.000002,
            cost_per_output_token=0.000008,
            _openai_api_key=os.environ.get("ARK_API_KEY"),
            
            # api_key=os.environ.get("ARK_API_KEY"),
        )
    
    def add_metric(self, metric):
        self.metrics.append(metric)

    
    def evaluate(self, predictions, references):
        # Implement evaluation logic specific to RAG tasks
        pass

    
    def load_data(self, data_path: str):
        # Implement data loading logic specific to RAG tasks

        csv_reader = pd.read_csv(data_path)
        self.dataset = csv_reader.to_dict()
        self.data_dir = os.path.dirname(data_path)
        # print(self.dataset)
        
        

    
    def run(self):
        
        self.add_metric(MCPToolMetric(model=self.llm_model, enable_judge_stability=False))
        testcase = []
        for i in range(0, len(self.dataset[INDEX_COLUMN])):
            print('index: %s'%i)
            input_text = self.dataset[INPUT_COLUMN][i]
            reference = self.dataset.get(EXPECTED_OUTPUT_COLUMN, None)[i] if self.dataset.get(EXPECTED_OUTPUT_COLUMN, None) else ""
            index = self.dataset[INDEX_COLUMN][i]-1
            # prediction = self.agent.generate_response(input_text)
            trace_file = os.path.join(self.data_dir, self.dataset.get(TRACE_ID_COLUMN, None)[i])
            eva_set_file = os.path.join(self.data_dir, self.dataset.get(EVAL_SET_ID_COLUMN, None)[i])
            expected_tool_content = self.dataset.get(EXPECTED_TOOLS_COLUMN, '')[i]
            expected_tool_result_content = self.dataset.get(EXPECTED_TOOLS_RESULT_COLUMN, '')[i]
            expected_tool_calls = parse_expected_tool_calls(expected_tool_content, expected_tool_result_content)

            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
            tool_calls, avalible_tools = extract_tool_calls_from_trace(trace_data)

            tools_called = turn_toolcalls(tool_calls)
            # 覆写工具执行成功与否：基于通用工具函数判断输出结果是否有效/是否包含错误信号
            for exec_res in tools_called:
                try:
                    exec_res.success = is_tool_execution_success(exec_res.output)
                except Exception:
                    # 若判定异常，保守为原值
                    pass
            with open(eva_set_file, 'r', encoding='utf-8') as f:
                eval_case = json.load(f)
            # print(eval_case)
            output = get_output_from_eval_case(eval_case['eval_cases'][index])
            contexts = get_context_from_trace(trace_data)

            tool_called_dict, avalibale_tools = extract_tool_calls_from_trace(trace_data )
            tool_called_struct = turn_toolcalls(tool_called_dict)
            


            testcase.append(AgentTestCase(
                id=index,
                input=input_text,
                actual_output=output,
                expected_output=reference,
                tools_called=tool_called_struct,
                available_tools=turn_avalible_tools(avalibale_tools),
                retrieval_context=contexts,
                expected_tools=turn_tool_dict2toolcall(expected_tool_calls)
            ))
        
        evaluate(testcase, self.metrics, async_config=AsyncConfig(max_concurrent=100), \
                 error_config=ErrorConfig(ignore_errors=False), cache_config=CacheConfig(use_cache=False, write_cache=True))

    def post_process(self, results):
        # 增加对badcase的一些分析处理，输出当前agent在该任务下的badcase以及存在的主要问题
        pass


class MCPTaskOnline(BaseTask):
    def __init__(self, 
                 task_name: str, 
                 agents: List[Agent],
                 testset_file: str = None,
                 sheet_name: str = None,
                 enable_chche: bool = False,
                 tracer: OpentelemetryTracer = None

                 ):
        super().__init__(task_name)
        self.testset_file = testset_file
        self.sheet_name = sheet_name    #If test_file is xlsx，you should specify the sheet_name
        self.output_dir = None
        self.agents: List[TestAgent] = []
        for agent in agents:
            self.agents.append(TestAgent(agent_entity=agent, tracer=tracer))
        self.llm_model = VolceOpenAI(
            model = os.environ.get("VOLCEMODEL"),
            temperature=0,
            base_url=os.environ.get("VOLCEBASEURL"),
            cost_per_input_token=0.000002,
            cost_per_output_token=0.000008,
            _openai_api_key=os.environ.get("ARK_API_KEY"),
            
        )
        self.metrics = [MCPToolMetric(model=self.llm_model, enable_judge_stability=False)]
        # 最大并发控制（可通过环境变量覆盖）
        try:
            self.max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "5"))
        except Exception:
            self.max_concurrency = 5
        # 缓存配置（可通过环境变量关闭/指定目录）
        self.enable_cache = enable_chche
        self.cache_dir = os.environ.get("VEAB_CACHE_DIR", ".cache/veagentbench")
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning(f"Failed to create cache dir: {self.cache_dir}")
       
    def validate(self):
        pass

    def add_metric(self, metric):
        self.metrics.append(metric)

    

    def generate_agent_response(self):
        async def _runner():
            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = []
            for agent in self.agents:
                for testcase in self.testcases[self._get_agent_id(agent)]:
                    tasks.append(asyncio.create_task(self._run_one(sem, agent, testcase)))
            total = len(tasks)
            if total == 0:
                return
            completed = 0
            # 使用 as_completed 驱动进度条
            with tqdm(total=total, desc="Evaluating testcases", unit="task") as pbar:
                for fut in asyncio.as_completed(tasks):
                    try:
                        await fut
                    finally:
                        completed += 1
                        pbar.update(1)
        try:
            asyncio.run(_runner())
        except RuntimeError as e:
            # 若在已有事件循环中调用（如Notebook环境），回退使用当前循环
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_runner())
    
    def get_agents_metrics(self):
        """_summary_
        """
        for agent in self.agents:
            testcases = self.testcases[self._get_agent_id(agent)]
            result = evaluate(testcases, self.metrics, async_config=AsyncConfig(max_concurrent=100), \
                 error_config=ErrorConfig(ignore_errors=False), cache_config=CacheConfig(use_cache=False, write_cache=True))
    
    def _get_metrics(self, testcases):
        """_summary_

        Args:
            testcases (_type_): _description_
        """
        pass


    def evaluate(self):
        self.load_data()
        # 并发执行所有 (testcase, agent) 组合，使用 asyncio.Semaphore 控制最大并发
        if not hasattr(self, "testcases") or not self.testcases:
            logger.warning("No testcases loaded. Did you call load_data()?")
            return
        self.generate_agent_response()
        self.get_agents_metrics()
          

    
    def load_data(self):
        # Implement data loading logic specific to RAG tasks
        logger.info('load data: %s'%self.testset_file)
        if os.path.splitext(self.testset_file)[1] == '.csv':
            csv_reader = pd.read_csv(self.testset_file)
            pd_dict = csv_reader.to_dict()
        elif os.path.splitext(self.testset_file)[1] == '.xlsx':
            excel_reader = pd.read_excel(self.testset_file, sheet_name=self.sheet_name)
            pd_dict = excel_reader.to_dict()           
        self.testcases = {}
        _testcases = self._parse_testcase(pd_dict)
        for agent in self.agents:
            self.testcases[self._get_agent_id(agent)] = _testcases
        


    def _parse_testcase(self, pd_dict: dict) -> List[AgentTestCase]:
        """_summary_

        Args:
            pd_dict (dict): _description_

        Returns:
            List[AgentTestCase]: _description_
        """
        testcases = []
        for i in range(0, len(pd_dict[INDEX_COLUMN])):
            logger.info('testcase_id: %s'%i)
            input_text = pd_dict[INPUT_COLUMN][i]
            reference = pd_dict.get(EXPECTED_OUTPUT_COLUMN, None)[i] if pd_dict.get(EXPECTED_OUTPUT_COLUMN, None) else ""
            
            expected_tool_content = pd_dict.get(EXPECTED_TOOLS_COLUMN, '')[i]
            expected_tool_result_content = pd_dict.get(EXPECTED_TOOLS_RESULT_COLUMN, '')[i]
            expected_tool_calls = parse_expected_tool_calls(expected_tool_content, expected_tool_result_content)

            testcases.append(AgentTestCase(
                input=input_text,
                id=i,
                expected_output=reference,
                expected_tools=turn_tool_dict2toolcall(expected_tool_calls),
                
            ))
        return testcases
         
    # ============ 缓存工具函数 ============
    def _get_agent_id(self, agent: TestAgent) -> str:
        try:
            if getattr(agent, "name", None):
                return str(agent.name)
            ent = getattr(agent, "agent_entity", None)
            if ent is not None and getattr(ent, "name", None):
                return str(ent.name)
        except Exception:
            pass
        return "agent"

    def _hash_text(self, text: str) -> str:
        try:
            return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]
        except Exception:
            return "unknown"

    def _cache_key(self, agent: TestAgent, testcase: AgentTestCase) -> str:
        agent_id = self._get_agent_id(agent)
        h = self._hash_text(getattr(testcase, "input", "") or "")
        task = getattr(self, "task_name", "task")
        return f"{task}__{agent_id}__{h}"

    def _cache_path(self, key: str) -> Path:
        return Path(self.cache_dir) / f"{key}.json"

    def _try_load_cache(self, agent: TestAgent, testcase: AgentTestCase):
        if not getattr(self, "enable_cache", False):
            return None
        key = self._cache_key(agent, testcase)
        p = self._cache_path(key)
        logger.info("cached path:%s"%p)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Read cache failed {p}: {e}")
            return None

    def _write_cache(self, agent: TestAgent, testcase: AgentTestCase, result: TestResut):
        if not getattr(self, "enable_cache", False):
            return
        try:
            key = self._cache_key(agent, testcase)
            p = self._cache_path(key)
            tmp = p.with_suffix(".json.tmp")
            payload = {
                "agent_id": self._get_agent_id(agent),
                "task_name": getattr(self, "task_name", "task"),
                "testcase_input_hash": self._hash_text(getattr(testcase, "input", "") or ""),
                "success": bool(getattr(result, "success", False)),
                "response": getattr(result, "response", None),
                "trace_data": getattr(result, "trace_data", None),
                "timestamp": int(time.time()),
            }
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)  # 原子替换
        except Exception as e:
            logger.warning(f"Write cache failed: {e}")

    async def _run_one(self, sem: asyncio.Semaphore, agent: TestAgent, testcase: AgentTestCase):
        # 使用信号量限制并发
        async with sem:
            # 先尝试读取缓存，命中则直接返回
            if self.enable_cache:
                try:
                    cached = await asyncio.to_thread(self._try_load_cache, agent, testcase)
                    if cached and cached.get("success") and (cached.get("response") is not None):
                        testcase.actual_output = cached.get("response")
                        testcase.trace_data = cached.get("trace_data")
                        tool_called_dict, avalibale_tools = extract_tool_calls_from_trace(testcase.trace_data )
                        tool_called_struct = turn_toolcalls(tool_called_dict)
                        testcase.tools_called = tool_called_struct
                        testcase.available_tools = turn_avalible_tools(avalibale_tools)
                        logger.info('response from cached: %s'%testcase.id)
                        return
                    else:

                        logger.info('load cached faied: %s'%testcase.id)
                except Exception:
                    pass
            try:
                # 找到可用的执行方法，按优先级选择
                fn = getattr(agent, 'generate')

                # 根据方法签名调用
                result: TestResut = None
                result = await fn(testcase.input)
               

                # 将结果写回用例，方便后续评估
                if result is not None and (result.success):
                    try:
                        testcase.actual_output = result.response
                        testcase.trace_data = result.trace_data
                        tool_called_dict, avalibale_tools = extract_tool_calls_from_trace(testcase.trace_data )
                        tool_called_struct = turn_toolcalls(tool_called_dict)
                        testcase.tools_called = tool_called_struct
                        testcase.available_tools = turn_avalible_tools(avalibale_tools)
                    except Exception:
                        pass
                    # 执行成功后写入缓存
                    try:
                        await asyncio.to_thread(self._write_cache, agent, testcase, result)
                    except Exception:
                        pass
                else:
                    raise Exception("testcase fail: %s"%result.response)
            except Exception as e:
                logger.exception(f"Execution failed for testcase with agent: {e}")
        
    

    def post_process(self, results):
        # 增加对badcase的一些分析处理，输出当前agent在该任务下的badcase以及存在的主要问题
        # from veagentbench.report.extract_metrics_corrected import 
        pass
