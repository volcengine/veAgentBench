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

from typing import Optional, List, Type, Union, Dict, Any
import asyncio
import json

from veagentbench.evals.deepeval.utils import get_or_create_event_loop
from veagentbench.evals.deepeval.metrics.utils import (
    check_llm_test_case_params,
    initialize_model,
)
from veagentbench.evals.deepeval.test_case import (
    LLMTestCaseParams,
)
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.models import DeepEvalBaseLLM
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator

from .template import LiveMCPBenchTemplate

from ...test_case import AgentTestCase



def safe_get(item, key, default=None):
    """Safely get a value from a dictionary"""
    if isinstance(item, dict):
        return item.get(key, default)
    else:
        return default


class LiveMcpBenchMetric(BaseMetric):
    """LiveMcpBench评估器
    """
    
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,

    ]

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[LiveMCPBenchTemplate] = LiveMCPBenchTemplate,
        enable_judge_stability: bool = False,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.evaluation_template = evaluation_template
        self.enable_judge_stability = enable_judge_stability
        
        
        # 其他属性
        self.success = False
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: AgentTestCase) -> float:
        """同步评估方法
        
        Args:
            test_case: 测试用例，支持LLMTestCase或AgentTestCase
        """
        # 处理不同类型的测试用例
        
        check_llm_test_case_params(test_case, self._required_params, self)
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(self.a_measure(test_case))
            else:
                self._measure(test_case)

        return self.score

    async def a_measure(self, test_case: AgentTestCase) -> float:
        """异步评估方法
        
        Args:
            test_case: 测试用例，支持LLMTestCase或AgentTestCase
        """
        check_llm_test_case_params(test_case, self._required_params, self)
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, async_mode=True):
            await self._a_measure(test_case)

        return self.score
    
    def _measure(self, test_case: AgentTestCase):
        """同步评估实现"""
        key_points = test_case.extra_fields.get('key_points')
        from pathlib import Path
        from collections import defaultdict

        current_dir_path = Path(__file__).resolve().parent
        tool_json = current_dir_path / 'tools.json'
        tool_map = defaultdict(dict)
        with open(tool_json, 'r') as f:
            toll_infos = json.loads(f.read())
            for tool_server in toll_infos:
                tools = tool_server["tools"]
                for tool in tools.values():
                    server_name = tool["server_name"]
                    for tl in tool["tools"]:
                        tool_map[server_name][tl["name"]] = {
                            "description": tl["description"],
                            "inputSchema": tl["inputSchema"],
                        }
        tool_call_info_list = []
        tools_called = test_case.tools_called
        tool_descriptions = ''
        for tool_call in tools_called:
            name = tool_call.name
            if name == 'mcp_use_tool':
                use_tools_infos = tool_call.input_parameters['use_tool_info']
                for info in use_tools_infos:
                    _tool_info = {
                        'server_name': info['server_name'],
                        'tool_name': info['tool_name'],
                        'params': info['argument']
                    }
                    tool_call_info_list.append(json.dumps(_tool_info))
                    tool_descriptions += self.format_tool_descriptions(
                                    tool_map,
                                    info.get("server_name", "not_given"),
                                    info.get("tool_name", "not_given"),
                                )
        
        # LLM评估指标
        llm_scores = self._evaluate_with_llm_judge(
            test_case.input,
            test_case.actual_output,
            key_points,
            tool_descriptions=tool_descriptions,
            tool_calls=tool_call_info_list
        )
        self.score = llm_scores.get('reward', 0)
        self.reason = llm_scores.get('judge_reason', '')
        self.success = True if self.score == 1 else False


    def format_tool_descriptions(self, tool_map, server_name, tool_name):
        if server_name not in tool_map or tool_name not in tool_map[server_name]:
            return f"Tool {tool_name} not found in server {server_name}."
        tool_descriptions = ""
        tool_descriptions += f"Server: {server_name}\n"
        tool_descriptions += f"Tool: {tool_name}\n"
        tool_info = tool_map[server_name][tool_name]
        tool_descriptions += f"Description: {tool_info['description']}\n"
        tool_descriptions += "\n"

        return tool_descriptions.strip()


    async def _a_measure(self, test_case: AgentTestCase):
        """异步评估实现"""   
        
        key_points = eval(test_case.extra_fields.get('Annotator Metadata'))['Steps']
        from collections import defaultdict

        tool_map = defaultdict(dict)
                        
        tool_call_info_list = []
        tools_called = test_case.tools_called
        tool_descriptions = ''
        
        for tool_call in tools_called:
            if tool_call.name == 'mcp_search_tool':
                text = tool_call.output['content'][0]['text']
                select_toolinfo = text.replace("### 1. Current available MCP Tools List is：", '').replace("### 2. After you have selected the MCP tool you need from the current available list, please call the mcp_use_tool tool to complete the task",'').strip()
                select_tools = json.loads(select_toolinfo)
                for tool in select_tools:
                    tool_map[tool['text']['server_name']][tool['tool_name']] = {
                        "description": tool['text']['tool_info']['description'],
                        "inputSchema": tool['text']['tool_info']["inputSchema"],
                    }
        for tool_call in tools_called:
            name = tool_call.name
            if name == 'mcp_use_tool':
                use_tools_infos = tool_call.input_parameters['use_tool_info']
                for info in use_tools_infos:
                    _tool_info = {
                        'server_name': info['server_name'],
                        'tool_name': info['tool_name'],
                        'params': info['argument']
                    }
                    tool_call_info_list.append(json.dumps(_tool_info))
                    tool_descriptions += self.format_tool_descriptions(
                                    tool_map,
                                    info.get("server_name", "not_given"),
                                    info.get("tool_name", "not_given"),
                                )
        
        # LLM评估指标
        llm_scores = await self._a_evaluate_with_llm_judge(
            test_case.input,
            test_case.actual_output,
            key_points,
            tool_descriptions=tool_descriptions,
            tool_calls=tool_call_info_list
        )
        self.score = llm_scores.get('reward', 0)
        self.reason = llm_scores.get('judge_reason', '')
        self.success = True if self.score == 1 else False


    def _evaluate_with_llm_judge(self, task: str, actual_output: str, key_points: str=None, tool_calls: str=None, tool_descriptions: str=None) -> Dict[str, Any]:
        """使用LLM评判进行评估（异步）"""
        
        import re
        prompt = self.evaluation_template.evaluate_llm_judge_dimensions(
            task=task,
            key_points=key_points,
            response=actual_output,
            tool_calls=tool_calls,
            tool_descriptions=tool_descriptions
        )

        
        # SON解析重试机制（指数退避 + 兜底）
        max_parse_retries = 3
        backoff = 0.5
        attempt = 0
        data = None
        res = None
        last_err = None
        while attempt <= max_parse_retries:
            try:
                # 首次已请求，其余重试重新请求并累计成本
                if attempt == 0:
                    res, cost =  self.model.generate(prompt)
                    self.evaluation_cost += cost
                else:
                    
                    res, cost =  self.model.generate(prompt)
                    self.evaluation_cost += cost
                    backoff *= 2
                # 解析
                judge_pattern = r"Status:\s*([\S]+)"
                thoughts_pattern=r"Thoughts:([\s\S]+)Status"
                judge_match = re.search(judge_pattern, res, re.DOTALL)
                thoughts_match = re.search(thoughts_pattern, res, re.DOTALL)
                if judge_match:
                    judge = judge_match.group(1).strip()
                else:
                    judge = res
                if thoughts_match:
                    thoughts = thoughts_match.group(1).strip()
                else:
                    thoughts = "Thoughts extract failed."
                reward = 1
                if "success" in judge.lower():
                    reward *= 1
                elif "failure" in judge.lower():
                    reward *= 0
                else:
                    reward *= 0
                data = {
                        "judge": judge,
                        "judge_reason": thoughts,
                        "reward": reward
                    }
            
                break
            except Exception as e:
                print('judge failed on attempt %d: %s' % (attempt, str(e)))
                if attempt > max_parse_retries:
                    # 兜底：尽量转为可用字典，避免任务中断
                    try:
                        if isinstance(res, dict):
                            data = res
                        else:
                            data = json.loads(res)
                    except Exception:
                        data = {}
                    break
        
        return data



    async def _a_evaluate_with_llm_judge(self, task: str, actual_output: str, key_points: str=None, tool_calls: str=None, tool_descriptions: str=None) -> Dict[str, Any]:
        """使用LLM评判进行评估（异步）"""
        
        import re
        prompt = self.evaluation_template.evaluate_llm_judge_dimensions(
            task=task,
            key_points=key_points,
            response=actual_output,
            tool_calls=tool_calls,
            tool_descriptions=tool_descriptions
        )
        # SON解析重试机制（指数退避 + 兜底）
        max_parse_retries = 3
        backoff = 0.5
        attempt = 0
        data = None
        res = None
        last_err = None
        while attempt <= max_parse_retries:
            try:
                # 首次已请求，其余重试重新请求并累计成本
                if attempt == 0:
                    res, cost = await self.model.a_generate(prompt)
                    self.evaluation_cost += cost
                else:
                    await asyncio.sleep(backoff)
                    res, cost = await self.model.a_generate(prompt)
                    self.evaluation_cost += cost
                    backoff *= 2
                # 解析
                judge_pattern = r"Status:\s*([\S]+)"
                thoughts_pattern=r"Thoughts:([\s\S]+)Status"
                judge_match = re.search(judge_pattern, res, re.DOTALL)
                thoughts_match = re.search(thoughts_pattern, res, re.DOTALL)
                if judge_match:
                    judge = judge_match.group(1).strip()
                else:
                    judge = res
                if thoughts_match:
                    thoughts = thoughts_match.group(1).strip()
                else:
                    thoughts = "Thoughts extract failed."
                reward = 1
                if "success" in judge.lower():
                    reward *= 1
                elif "failure" in judge.lower():
                    reward *= 0
                else:
                    reward *= 0
                data = {
                        "judge": judge,
                        "judge_reason": thoughts,
                        "reward": reward
                    }
            
                break
            except Exception as e:
                print('judge failed on attempt %d: %s' % (attempt, str(e)))
                if attempt > max_parse_retries:
                    # 兜底：尽量转为可用字典，避免任务中断
                    try:
                        if isinstance(res, dict):
                            data = res
                        else:
                            data = json.loads(res)
                    except Exception:
                        data = {}
                    break
        
        return data


    def is_successful(self) -> bool:
        """返回评估是否成功"""
        return self.success

    @property
    def __name__(self):
        return "Live Mcp Bench Correctness"

       