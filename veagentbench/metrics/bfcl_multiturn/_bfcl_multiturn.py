## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import asyncio
from typing import List, Dict, Any
import traceback

from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.test_case.agent_test_case import AgentTestCase
from veagentbench.evals.deepeval.test_case import LLMTestCaseParams
from .multi_turn_eval.multi_turn_checker import multi_turn_checker



class BFCLMultiTurnMetric(BaseMetric):
    """基于MCP架构的BFCL多轮对话评测器
    
    适配bfcl_eval的multi_turn_checker，但使用MCP服务器进行工具调用
    而非本地类实例。
    """
    
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
    
    def __init__(
        self,
        threshold: float = 0.8,
        include_reason: bool = True,
        strict_mode: bool = False,
        model = None
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.success = False
        self.score = 0.0
        self.reason = None
        self.error = None
        self.verbose_logs = None
        
    def measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """评测AgentTestCase的多轮对话正确性"""
        try:
            # 运行多轮检查
            _ = asyncio.run(self.a_measure(test_case))

            return self.score
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            self.reason = f"评测过程出错: {str(e)}"
            self.verbose_logs = f"错误详情:\n{traceback.format_exc()}"
            return self.score
    
    async def a_measure(self, test_case: AgentTestCase) -> float:
        """异步评测AgentTestCase的多轮对话正确性"""
        # 提取测试数据
        multi_turn_model_results = self._extract_model_results(test_case)
        multi_turn_ground_truth = eval(test_case.expected_output)
        initial_config = eval(self._extract_initial_config(test_case))
        involved_classes = eval(self._extract_involved_classes(test_case))
        
        # 执行多轮检查
        return await self.multi_turn_checker(
            multi_turn_model_results,
            multi_turn_ground_truth,
            initial_config,
            involved_classes,
            test_case.name or "unknown_test"
        )
    
    def _extract_model_results(self, test_case: AgentTestCase) -> List[List[List[str]]]:
        """从AgentTestCase提取模型结果"""
        results = []
        
        for turn in test_case.turns:
            if turn.role != "assistant":   #跳过非助手角色的轮次
                continue
            if turn.tools_called:
                turn_results = []
                for tool_call in turn.tools_called:
                    # 将MCP工具调用转换为函数调用字符串
                    func_call = self._mcp_tool_to_func_call(tool_call)
                    turn_results.append([func_call])
                results.append(turn_results)
            else:
                results.append([[]])
                
        return results
    
    def _extract_ground_truth(self, test_case: AgentTestCase) -> List[List[str]]:
        """从AgentTestCase提取ground truth"""
        # 从expected_tools或自定义字段中提取期望的工具调用
        if test_case.expected_tools:
            return [[tool.tool_name for tool in test_case.expected_tools]]
        return [[]]
    
    def _extract_initial_config(self, test_case: AgentTestCase) -> Dict[str, Any]:
        """提取初始配置"""
        # 从test_case的additional_metadata或其他字段中提取
        if hasattr(test_case, 'extra_fields') and test_case.extra_fields:
            return test_case.extra_fields.get('initial_config', {})
        return {}
    
    def _extract_involved_classes(self, test_case: AgentTestCase) -> List[str]:
        """提取涉及的类/服务器"""
        # 从available_tools或MCP服务器信息中提取
        if hasattr(test_case, 'extra_fields') and test_case.extra_fields:
            return test_case.extra_fields.get('involved_classes', {})
        return {}
    
    def _mcp_tool_to_func_call(self, mcp_tool_call) -> str:
        """将MCP工具调用转换为函数调用字符串"""
        tool_name = mcp_tool_call.name
        arguments = mcp_tool_call.input_parameters if hasattr(mcp_tool_call, 'input_parameters') else {}
        
        # 构建函数调用字符串，如: tool_name(arg1="value1", arg2="value2")
        args_str = ", ".join([f'{k}="{v}"' for k, v in arguments.items()])
        return f"{tool_name}({args_str})"
    
    async def multi_turn_checker(
        self,
        multi_turn_model_result_list: List[List[List[str]]],
        multi_turn_ground_truth_list: List[List[str]],
        initial_config: Dict[str, Any],
        involved_classes: List[str],
        test_entry_id: str,
    ) -> float:
        """核心的多轮对话检查器"""
        
        result = multi_turn_checker(
            multi_turn_model_result_list_decoded = multi_turn_model_result_list,
            multi_turn_ground_truth_list = multi_turn_ground_truth_list,
            initial_config= initial_config,
            involved_classes = involved_classes,
            test_entry_id = test_entry_id,
            model_name="",
        )
        print(result)
        if result.get('valid') == True:
            self.score = 1
            self.success = True
        else:
            self.score = 0
            self.reason = result.get('details')
        
        return self.score
        
        
    

    def is_successful(self) -> bool:
        """检查是否评测成功"""
        return self.success and self.score >= self.threshold
    
    @property
    def __name__(self):
        return "BFCL Multi-Turn MCP"
