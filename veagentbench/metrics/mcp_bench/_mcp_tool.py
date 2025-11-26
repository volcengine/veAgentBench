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
import time
import json
import numpy as np
import jsonschema
from jsonschema import ValidationError
from collections import Counter
from veagentbench.utils.extract_expected_tool_calls import turn_tool_toolcall2dict

from veagentbench.evals.deepeval.utils import get_or_create_event_loop, prettify_list
from veagentbench.evals.deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from veagentbench.evals.deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.models import DeepEvalBaseLLM
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator

# Import local schema and template
from .schema import *
from .template import MCPToolTemplate

# Import AgentTestCase

from ...test_case import AgentTestCase



def safe_get(item, key, default=None):
    """Safely get a value from a dictionary"""
    if isinstance(item, dict):
        return item.get(key, default)
    else:
        return default


class MCPToolMetric(BaseMetric):
    """MCP工具评估指标
    
    基于mcp-bench的TaskEvaluator实现，包含两大类指标：
    1. LLM评估指标：任务完成、工具使用、规划效率
    2. 工具匹配指标：schema合规性、工具名称有效性、执行成功率等
    
    统一的输入结构：
    - 支持AgentTestCase（推荐）：集成了MCPExecutionData的统一测试用例
    - 支持LLMTestCase + execution_data：传统方式，向后兼容
    - 支持LLMTestCase自动解析：从actual_output解析工具调用信息
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
        strict_mode: bool = False,
        verbose_mode: bool = False,
        available_tools: List[Dict[str, ToolCall]] = None,
        evaluation_template: Type[MCPToolTemplate] = MCPToolTemplate,
        enable_judge_stability: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.available_tools = available_tools or {}
        self.evaluation_template = evaluation_template
        self.enable_judge_stability = enable_judge_stability

        # 评估结果 - LLM评估指标
        self.task_fulfillment = 0.0
        self.grounding = 0.0
        self.tool_appropriateness = 0.0
        self.parameter_accuracy = 0.0
        self.dependency_awareness = 0.0
        self.parallelism_and_efficiency = 0.0
        
        # 聚合分数
        self.task_completion_score = 0.0
        self.tool_selection_score = 0.0
        self.planning_effectiveness_score = 0.0
        
        # 工具匹配指标
        self.input_schema_compliance = 0.0
        self.valid_tool_name_rate = 0.0
        self.execution_success_rate = 0.0
        self.valid_call_failure_rate = 0.0
        # self.planning_json_compliance = 1.0
        
        # 服务器利用率指标
        self.server_count = 0
        self.cross_server_coordination = False
        self.server_distribution = {}
        
        # 其他属性
        self.success = False
        self.score = 0.0
        self.reason = ""
        self.execution_data = None

    def measure(self, test_case: AgentTestCase) -> float:
        """同步评估方法
        
        Args:
            test_case: 测试用例，支持LLMTestCase或AgentTestCase
            execution_data: 工具执行数据（当使用LLMTestCase时需要）
        """
        # 处理不同类型的测试用例
        
        self.available_tools = test_case.available_tools
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
            execution_data: 工具执行数据（当使用LLMTestCase时需要）
        """
        
        self.available_tools = test_case.available_tools
        check_llm_test_case_params(test_case, self._required_params, self)
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, async_mode=True):
            await self._a_measure(test_case)

        return self.score

    def _process_test_case(self, test_case: Union[LLMTestCase, 'AgentTestCase'], execution_data: Optional[MCPExecutionData] = None) -> tuple[LLMTestCase, Optional[MCPExecutionData]]:
        """处理不同类型的测试用例
        
        Args:
            test_case: 输入的测试用例
            execution_data: 可选的执行数据
            
        Returns:
            (LLMTestCase, MCPExecutionData) 元组
        """
        if AgentTestCase and isinstance(test_case, AgentTestCase):
            # 使用AgentTestCase（推荐方式）
            llm_test_case = test_case.to_llm_test_case()
            execution_data = test_case.execution_data
            
            # 如果AgentTestCase中有available_tools，使用它
            if test_case.available_tools and not self.available_tools:
                self.available_tools = test_case.available_tools
                
        elif isinstance(test_case, LLMTestCase):
            # 使用传统的LLMTestCase
            llm_test_case = test_case
            # execution_data保持传入的值
            
        else:
            raise ValueError(f"Unsupported test case type: {type(test_case)}")
        
        return llm_test_case, execution_data

    def _measure(self, test_case: AgentTestCase):
        """同步评估实现"""
        self.execution_data = MCPExecutionData(
                    expected_tool_calls=turn_tool_toolcall2dict(test_case.expected_tools),
                    tool_executions=test_case.tools_called, 
                    total_rounds=test_case.total_round, 
                    concrete_task_description=test_case.input, 
                    dependency_analysis=test_case.dependency_analysis)

        # 计算工具匹配指标
        tool_metrics = self._calculate_tool_accuracy_metrics(self.execution_data.tool_executions)
        self._update_tool_metrics(tool_metrics)
        
       
        # LLM评估指标
        llm_scores = self._evaluate_with_llm_judge(
            test_case.input,
            test_case.actual_output,
            getattr(test_case, 'expected_output', None)
        )
        self._update_llm_scores(llm_scores)
        
        # 计算最终分数
        self._calculate_final_score()

    async def _a_measure(self, test_case: AgentTestCase):
        """异步评估实现"""   
        self.execution_data = MCPExecutionData(
                expected_tool_calls=turn_tool_toolcall2dict(test_case.expected_tools),
                tool_executions=test_case.tools_called, 
                total_rounds=test_case.total_round, 
                concrete_task_description=test_case.input, 
                dependency_analysis=test_case.dependency_analysis)     
        # 计算工具匹配指标
        tool_metrics = self._calculate_tool_accuracy_metrics(self.execution_data.tool_executions)
        self._update_tool_metrics(tool_metrics)
        


        # LLM评估指标
        llm_scores = await self._a_evaluate_with_llm_judge(
            test_case.input,
            test_case.actual_output,
            getattr(test_case, 'expected_output', None)
        )
        self._update_llm_scores(llm_scores)
        
        # 计算最终分数
        self._calculate_final_score()


    def _calculate_tool_accuracy_metrics(self, execution_results: List[ToolExecutionResult]) -> Dict[str, Any]:
        """计算工具选择和执行准确性指标"""
        if not execution_results:
            return {
                'input_schema_compliance': None,
                'valid_tool_name_rate': None,
                'execution_success_rate': None,
                'valid_call_failure_rate': None,
                # 'planning_json_compliance': self.execution_data.planning_json_compliance if self.execution_data else 1.0
            }
        
        valid_tool_calls = 0
        schema_compliant_calls = 0
        successful_executions = 0
        valid_call_failures = 0
        
        for result in execution_results:
            tool_name = result.name
            success = result.success
            parameters = result.input_parameters
            
            # 检查工具名称是否有效
            is_valid_tool = tool_name in self.available_tools.keys()
            if is_valid_tool:
                valid_tool_calls += 1
                
                # 检查schema合规性
                if self._check_schema_compliance(tool_name, parameters):
                    schema_compliant_calls += 1
                
                # 跟踪有效工具调用的失败
                if not success:
                    valid_call_failures += 1
            
            # 跟踪成功执行
            if success:
                successful_executions += 1
        
        total_calls = len(execution_results)
        
        return {
            'input_schema_compliance': schema_compliant_calls / valid_tool_calls if valid_tool_calls > 0 else 0.0,
            'valid_tool_name_rate': valid_tool_calls / total_calls,
            'execution_success_rate': successful_executions / total_calls,
            'valid_call_failure_rate': valid_call_failures / valid_tool_calls if valid_tool_calls > 0 else 0.0,
            # 'planning_json_compliance': self.execution_data.planning_json_compliance if self.execution_data else 1.0
        }

    def _check_schema_compliance(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """检查参数是否符合工具schema"""
        try:
            tool_info = self.available_tools.get(tool_name, {})
            input_schema = tool_info.input_parameters
            
            if not input_schema:
                return True  # 没有schema可验证
            
            jsonschema.validate(parameters, input_schema)
            return True
            
        except (ValidationError, Exception):
            return False



    def generate_accumulated_info(self, task: str) -> str:
        """_summary_

        Args:
            task (str): _description_
            execution_data (MCPExecutionData): _description_

        Returns:
            str: _description_
        """
        execution_data = self.execution_data
        if not execution_data or not execution_data.tool_executions:
            return "无法完成任务：没有可用的工具执行结果。"
        
        # 参照mcp-bench的_update_state方法构建详细的累积信息
        accumulated_info = []
        
        # 按轮次组织工具执行信息（如果有轮次信息）
        rounds_info: Dict[int, List[ToolExecutionResult]] = {}
        for execution in execution_data.tool_executions:
            round_num = getattr(execution, 'round_num', 1)
            if round_num not in rounds_info:
                rounds_info[round_num] = []
            rounds_info[round_num].append(execution)
        
        # 生成每轮的详细摘要
        for round_num in sorted(rounds_info.keys()):
            round_executions = rounds_info[round_num]
            round_summary = f"\n\n--- Summary of Round {round_num} ---\n"
            
            for execution in round_executions:
                server = getattr(execution, 'server', 'unknown')
                tool_name = execution.name
                parameters = execution.input_parameters
                
                # 格式化参数显示
                params_str = f"{parameters}" if parameters else "{}"
                
                if execution.success:
                    content = execution.output
                    # 估算内容长度（简单的token估算）
                    content_length = len(str(content))
                    
                    if content_length <= 2000:  # 相当于约500 tokens
                        round_summary += f"Tool `{tool_name}` with Parameter {params_str} on {server} succeeded. Result: {content}\n"
                    else:
                        # 对于长内容，进行截断处理
                        truncated_content = str(content)[:1500] + "..." if len(str(content)) > 1500 else str(content)
                        round_summary += f"Tool `{tool_name}` with Parameter {params_str} on {server} succeeded. Result (truncated from {content_length} chars): {truncated_content}\n"
                else:
                    error_content = execution.error_message or "Unknown error"
                    error_length = len(str(error_content))
                    
                    if error_length <= 1000:
                        round_summary += f"Tool `{tool_name}` with Parameter {params_str} on {server} failed. Error: {error_content}\n"
                    else:
                        # 对于长错误信息，进行截断处理
                        truncated_error = str(error_content)[:800] + "..." if len(str(error_content)) > 800 else str(error_content)
                        round_summary += f"Tool `{tool_name}` with Parameter {params_str} on {server} failed. Error (truncated from {error_length} chars): {truncated_error}\n"
            
            accumulated_info.append(round_summary)
        
        # 如果没有轮次信息，使用简单的顺序格式
        if not rounds_info or len(rounds_info) == 1:
            accumulated_info = []
            for i, execution in enumerate(execution_data.tool_executions, 1):
                server = getattr(execution, 'server', 'unknown')
                tool_name = execution.name
                parameters = execution.input_parameters
                params_str = f"{parameters}" if parameters else "{}"
                
                if execution.success:
                    content = execution.output
                    content_length = len(str(content))
                    
                    if content_length <= 2000:
                        info = f"Tool `{tool_name}` with Parameter {params_str} on {server} succeeded. Result: {content}"
                    else:
                        truncated_content = str(content)[:1500] + "..." if len(str(content)) > 1500 else str(content)
                        info = f"Tool `{tool_name}` with Parameter {params_str} on {server} succeeded. Result (truncated): {truncated_content}"
                else:
                    error_content = execution.error_message or "Unknown error"
                    error_length = len(str(error_content))
                    
                    if error_length <= 1000:
                        info = f"Tool `{tool_name}` with Parameter {params_str} on {server} failed. Error: {error_content}"
                    else:
                        truncated_error = str(error_content)[:800] + "..." if len(str(error_content)) > 800 else str(error_content)
                        info = f"Tool `{tool_name}` with Parameter {params_str} on {server} failed. Error (truncated): {truncated_error}"
                
                accumulated_info.append(info)
        
        accumulated_information = "\n".join(accumulated_info)
        self.execution_data.accumulated_information = accumulated_information


    async def _synthesize_final_solution(self, task) -> str:
        """
        参照mcp-bench的_synthesize_final_solution方法，基于工具执行结果生成最终答案
        
        Args:
            task: 原始用户任务
            execution_data: 工具执行数据
            
        Returns:
            基于工具执行结果综合生成的最终答案
        """
        total_executions = len(self.execution_data.tool_executions)
        accumulated_information = self.execution_data.accumulated_information or ""

        # 构建综合提示词（参照mcp-bench的实现）
        system_prompt = """你是一个专业的解决方案综合器，负责为多工具AI代理执行生成最终答案。
你的任务是基于工具执行结果，为用户提供清晰、全面、结构化的最终答案。"""
        
        prompt = f"""原始任务: "{task}"

多轮执行过程已完成，共进行了 {total_executions} 次工具调用。

累积信息和结果:
{accumulated_information}

基于原始任务和从多个工具收集的所有信息，请提供一个最终的、全面的、结构良好的答案，直接回应用户的请求。

要求:
1. 综合关键发现并以清晰、有组织的方式呈现
2. 展示不同工具能力是如何结合使用的
3. 提供直接有用的最终答案，而不是技术执行细节
4. 确保答案完整回应用户的原始需求
5. 使用用户友好的语言，避免技术术语

最终答案:"""
        
        try:
            
            
            response, cost = await self.model.a_generate("%s\n%s"%(prompt, system_prompt))
            
            return response.strip()
            
        except Exception as e:
            # 如果LLM生成失败，提供基础的结果汇总
            successful_results = [exec for exec in execution_data.tool_executions if exec.success]
            if successful_results:
                summary = f"基于 {len(successful_results)} 个成功的工具调用结果："
                for exec in successful_results:
                    summary += f"\n- {exec.name}: {exec.output}"
                return summary
            else:
                return f"任务执行遇到问题，{total_executions} 次工具调用均未成功。"

    def _evaluate_with_llm_judge(self, task: str, actual_output: str, expected_output: str = None) -> Dict[str, Any]:
        """使用LLM评判进行评估（同步），支持judge stability增强评分可信度"""
        # 创建执行摘要
        self.generate_accumulated_info(task)
        # 根据mcp-bench的实现，如果没有提供actual_output或者需要生成final_solution
        # 我们基于execution_data生成final_solution
        if not actual_output or actual_output.strip() == "":
            # 同步版本：使用异步方法的同步包装
            # import asyncio
            # try:
            #     loop = asyncio.get_event_loop()
            #     final_solution = loop.run_until_complete(self._synthesize_final_solution(task, execution_data))
            # except RuntimeError:
            #     # 如果没有事件循环，创建一个新的
            #     final_solution = asyncio.run(self._synthesize_final_solution(task, execution_data))
            final_solution = ''
        else:
            final_solution = actual_output
        execution_summary = self._create_execution_summary(self.execution_data)

        # 标准单次评估
        prompt = self.evaluation_template.evaluate_llm_judge_dimensions(
            task=task,
            final_solution=final_solution,
            execution_summary=execution_summary,
            total_rounds=self.execution_data.total_rounds if self.execution_data else 1,
            available_tools=self.available_tools,
            concrete_task_description=self.execution_data.concrete_task_description if self.execution_data else None,
            dependency_analysis=self.execution_data.dependency_analysis if self.execution_data else None,
            expected_tool_calls=getattr(self.execution_data, 'expected_tool_calls', None) if self.execution_data else None
        )
        
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
        else:
            res, cost = self.model.generate(prompt)


        data = trimAndLoadJson(res, self)
        return data

    async def _a_evaluate_with_llm_judge(self, task: str, actual_output: str, expected_output: str = None) -> Dict[str, Any]:
        """使用LLM评判进行评估（异步）"""
        # 根据mcp-bench的实现，如果没有提供actual_output或者需要生成final_solution
        # 我们基于execution_data生成final_solution
        
        # _final_solution = await self._synthesize_final_solution(task, execution_data)
        self.generate_accumulated_info(task)
        _final_solution = ''
        if not actual_output or actual_output.strip() == "":
            final_solution = _final_solution
        else:
            final_solution = actual_output
        # 创建执行摘要
        execution_summary = self._create_execution_summary(self.execution_data)
        
        

        prompt = self.evaluation_template.evaluate_llm_judge_dimensions(
            task=task,
            final_solution=final_solution,
            execution_summary=execution_summary,
            total_rounds=self.execution_data.total_rounds if self.execution_data else 1,
            available_tools=self.available_tools,
            concrete_task_description=self.execution_data.concrete_task_description if self.execution_data else None,
            dependency_analysis=self.execution_data.dependency_analysis if self.execution_data else None,
            expected_tool_calls=self.execution_data.expected_tool_calls
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
                data = trimAndLoadJson(res, self)
                break
            except Exception as e:
                last_err = e
                attempt += 1
                print('JSON parsing failed on attempt %d: %s' % (attempt, str(e)))
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

    def _create_execution_summary(self, execution_data: MCPExecutionData = None) -> str:
        """创建执行摘要"""
        if not execution_data or not execution_data.tool_executions:
            return "No tools were executed."
        
        execution_results = execution_data.tool_executions
        successful_tools = [r for r in execution_results if r.success]
        failed_tools = [r for r in execution_results if not r.success]
        
        summary_parts = [
            f"Total rounds: {execution_data.total_rounds}",
            f"Tools executed: {len(execution_results)}",
            f"Successful: {len(successful_tools)}",
            f"Failed: {len(failed_tools)}"
        ]
        
        if successful_tools:
            successful_tool_names = [r.name for r in successful_tools]
            summary_parts.append(f"Successful tools: {', '.join(successful_tool_names)}")
        
        if failed_tools:
            failed_tool_names = [r.name for r in failed_tools]
            summary_parts.append(f"Failed tools: {', '.join(failed_tool_names)}")
        
        execution_stats = "; ".join(summary_parts)
        
        # 添加累积信息
        if execution_data.accumulated_information:
            return f"{execution_stats}\n\n--- ACCUMULATED INFORMATION FROM EXECUTION ---\n{execution_data.accumulated_information}"
        else:
            return execution_stats

    def _update_tool_metrics(self, tool_metrics: Dict[str, Any]):
        """更新工具匹配指标"""
        self.input_schema_compliance = tool_metrics.get('input_schema_compliance', 0.0) or 0.0
        self.valid_tool_name_rate = tool_metrics.get('valid_tool_name_rate', 0.0) or 0.0
        self.execution_success_rate = tool_metrics.get('execution_success_rate', 0.0) or 0.0
        self.valid_call_failure_rate = tool_metrics.get('valid_call_failure_rate', 0.0) or 0.0
        # self.planning_json_compliance = tool_metrics.get('planning_json_compliance', 1.0) or 1.0

    def _update_server_metrics(self, server_metrics: Dict[str, Any]):
        """更新服务器指标"""
        self.server_count = server_metrics.get('server_count', 0)
        self.cross_server_coordination = server_metrics.get('cross_server_coordination', False)
        self.server_distribution = server_metrics.get('server_distribution', {})

    def _update_llm_scores(self, llm_scores: Dict[str, Any]):
        """更新LLM评估分数"""
        # 直接使用mcp-bench的6个维度分数 (1-10分制)
        self.task_fulfillment = llm_scores.get('task_fulfillment', 0.0) if llm_scores.get('task_fulfillment') != None else 0.0 
        self.grounding = llm_scores.get('grounding', 0.0) if llm_scores.get('grounding') != None else 0.0 
        self.tool_appropriateness = llm_scores.get('tool_appropriateness', 0.0) if llm_scores.get('tool_appropriateness') != None else 0.0 
        self.parameter_accuracy = llm_scores.get('parameter_accuracy', 0.0) if llm_scores.get('parameter_accuracy') != None else 0.0 
        self.dependency_awareness = llm_scores.get('dependency_awareness', 0.0) if llm_scores.get('dependency_awareness') != None else 0.0 
        self.parallelism_and_efficiency = llm_scores.get('parallelism_and_efficiency', 0.0) if llm_scores.get('parallelism_and_efficiency') != None else 0.0 
        
        # 计算聚合分数
        self.task_completion_score = (self.task_fulfillment + self.grounding) / 2
        self.tool_selection_score = (self.tool_appropriateness + self.parameter_accuracy) / 2
        self.planning_effectiveness_score = (self.dependency_awareness + self.parallelism_and_efficiency) / 2
        
        self.score_breakdown = {
            'task_fulfillment': self.task_fulfillment,
            'grounding': self.grounding,
            'tool_appropriateness': self.tool_appropriateness,
            'parameter_accuracy': self.parameter_accuracy,
            'dependency_awareness': self.dependency_awareness,
            'parallelism_and_efficiency': self.parallelism_and_efficiency,
            # 'task_completion_score': self.task_completion_score,
            # 'tool_selection_score': self.tool_selection_score,
            # 'planning_effectiveness_score': self.planning_effectiveness_score,
            'input_schema_compliance': self.input_schema_compliance,
            'valid_tool_name_rate': self.valid_tool_name_rate,
            'execution_success_rate': self.execution_success_rate,
            # 'valid_call_failure_rate': self.valid_call_failure_rate,
            # 'planning_json_compliance': self.planning_json_compliance,
        }
        # 保存评估原因
        if self.include_reason:
            reasoning_parts = []
            if llm_scores.get('task_fulfillment_reasoning'):
                reasoning_parts.append(f"Task Fulfillment: {llm_scores['task_fulfillment_reasoning']}")
            if llm_scores.get('grounding_reasoning'):
                reasoning_parts.append(f"Grounding: {llm_scores['grounding_reasoning']}")
            if llm_scores.get('tool_appropriateness_reasoning'):
                reasoning_parts.append(f"Tool Appropriateness: {llm_scores['tool_appropriateness_reasoning']}")
            if llm_scores.get('parameter_accuracy_reasoning'):
                reasoning_parts.append(f"Parameter Accuracy: {llm_scores['parameter_accuracy_reasoning']}")
            if llm_scores.get('dependency_awareness_reasoning'):
                reasoning_parts.append(f"Dependency Awareness: {llm_scores['dependency_awareness_reasoning']}")
            if llm_scores.get('parallelism_efficiency_reasoning'):
                reasoning_parts.append(f"Parallelism & Efficiency: {llm_scores['parallelism_efficiency_reasoning']}")
            
            self.reason = "\n\n".join(reasoning_parts) if reasoning_parts else "No detailed reasoning provided"

    def _calculate_final_score(self):
        """计算最终综合分数"""
        # LLM评估指标权重 (60%)
        llm_score = (self.task_completion_score + self.tool_selection_score + self.planning_effectiveness_score) / 30  # 转换为0-1
        
        # 工具匹配指标权重 (40%)
        tool_metrics = [
            self.input_schema_compliance,
            self.valid_tool_name_rate,
            self.execution_success_rate,
            (1 - self.valid_call_failure_rate),  # 转换为正向指标
        ]
        
        # 过滤None值
        valid_tool_metrics = [m for m in tool_metrics if m is not None]
        tool_score = sum(valid_tool_metrics) / len(valid_tool_metrics) if valid_tool_metrics else 0.0
        
        # 综合分数
        self.score = 0.6 * llm_score + 0.4 * tool_score
        
        # 确保分数在0-1范围内
        self.score = max(0.0, min(1.0, self.score))
        
        # 判断是否成功
        self.success = self.score >= self.threshold

    def is_successful(self) -> bool:
        """返回评估是否成功"""
        return self.success

    @property
    def __name__(self):
        return "MCP Tool Correctness"
    
    def get_intermediate_metrics(self) -> Dict[str, Any]:
        """导出评估过程中的中间指标，便于对外透出或记录"""
        return {
            # LLM评估维度 (0-10 或按模板返回的区间)
            "task_fulfillment": self.task_fulfillment,
            "grounding": self.grounding,
            "tool_appropriateness": self.tool_appropriateness,
            "parameter_accuracy": self.parameter_accuracy,
            "dependency_awareness": self.dependency_awareness,
            "parallelism_and_efficiency": self.parallelism_and_efficiency,
            # 聚合分数
            "task_completion_score": self.task_completion_score,
            "tool_selection_score": self.tool_selection_score,
            "planning_effectiveness_score": self.planning_effectiveness_score,
            # 工具匹配指标
            "input_schema_compliance": self.input_schema_compliance,
            "valid_tool_name_rate": self.valid_tool_name_rate,
            "execution_success_rate": self.execution_success_rate,
            "valid_call_failure_rate": self.valid_call_failure_rate,
            # 规划JSON合规（若存在）
            "planning_json_compliance": getattr(self, "planning_json_compliance", None),
            # 服务器利用率
            "server_count": self.server_count,
            "cross_server_coordination": self.cross_server_coordination,
            "server_distribution": self.server_distribution,
            # 最终结果
            "final_score": self.score,
            "success": self.success,
            "reason": self.reason if self.include_reason else None,
        }
    def _evaluate_with_stability_testing(self, task: str, final_solution: str, 
                                       execution_summary: str, execution_data: MCPExecutionData) -> Dict[str, Any]:
        """使用stability testing进行多次评估以增强可信度"""
        import random
        
        scores = []
        reasons = []
        
        # 进行5次评估，每次使用随机化的prompt
        for i in range(5):
            try:
                # 生成随机化的prompt
                randomized_prompt = self._generate_randomized_prompt(task, final_solution, execution_summary, i, execution_data)
                
                # 执行评估
                if self.using_native_model:
                    response = self.model.generate(randomized_prompt)
                else:
                    response = self.model.generate(randomized_prompt)
                    self.evaluation_cost += self.model.cost
                
                # 解析结果
                data = trimAndLoadJson(response, self)
                if data and isinstance(data, dict):
                    scores.append(data)
                    reasons.append(f"Run {i+1}: {data.get('reason', 'No reason provided')}")
                    
            except Exception as e:
                reasons.append(f"Run {i+1} failed: {str(e)}")
                continue
        
        if not scores:
            return {
                'score': 0.0,
                'reason': "All stability testing runs failed: " + "; ".join(reasons),
                'success': False
            }
        
        # 使用mcp-bench的_calculate_average_scores方法计算平均分数
        averaged_result = self._calculate_average_scores_from_stability(scores)
        
        # 获取平均后的最终分数
        final_avg_score = averaged_result.get('score', 0.0)
        
        # 计算原始分数的标准差以评估稳定性
        original_scores = [result.get('score', 0.0) for result in scores if 'score' in result]
        if len(original_scores) > 1:
            original_avg = sum(original_scores) / len(original_scores)
            variance = sum((s - original_avg) ** 2 for s in original_scores) / (len(original_scores) - 1)
            std_dev = variance ** 0.5
            stability_info = f" (std_dev: {std_dev:.3f})"
        else:
            std_dev = 0.0
            stability_info = ""
        
        # 构建详细的原因说明
        individual_scores_info = [f"Run {i+1}: {scores[i].get('score', 0.0):.3f}" for i in range(len(scores))]
        combined_reason = f"Stability testing with {len(scores)}/5 successful runs. Final averaged score: {final_avg_score:.3f}{stability_info}. Individual scores: [{', '.join(individual_scores_info)}]. Reasoning from first run: {averaged_result.get('reason', 'No reason provided')}"
        
        # 返回各个维度的平均分数，与标准评估保持一致的结构
        result = {
            'score': final_avg_score,
            'reason': combined_reason,
            'success': True,
            'stability_scores': original_scores,
            'stability_std_dev': std_dev
        }
        
        # 添加各个维度的平均分数（如果存在）
        dimension_fields = [
            'task_fulfillment', 'grounding', 'tool_appropriateness', 
            'parameter_accuracy', 'dependency_awareness', 'parallelism_and_efficiency',
            'task_completion_score', 'tool_selection_score', 
            'planning_effectiveness_and_efficiency_score'
        ]
        
        for field in dimension_fields:
            if field in averaged_result:
                result[field] = averaged_result[field]
        
        # 添加分析文本（如果存在）
        analysis_fields = [
            'task_completion_analysis', 'tool_selection_analysis',
            'planning_effectiveness_and_efficiency_analysis'
        ]
        
        for field in analysis_fields:
            if field in averaged_result:
                result[field] = averaged_result[field]
        
        return result
    
    def _generate_randomized_prompt(self, task: str, final_solution: str, 
                                  execution_summary: str, run_index: int, execution_data: MCPExecutionData = None) -> str:
        """生成随机化的prompt以增强评估稳定性"""
        
        # 随机化prompt结构的不同变体
        prompt_variants = [
            # 变体1: 标准格式
            self.evaluation_template.evaluate_llm_judge_dimensions(
                task=task,
                final_solution=final_solution,
                execution_summary=execution_summary,
                total_rounds=1,
                available_tools=self.available_tools,
                concrete_task_description=None,
                dependency_analysis=None,
                expected_tool_calls=getattr(execution_data, 'expected_tool_calls', None) if execution_data else None
            ),
            
            # 变体2: 重新排列评估维度
            self._create_reordered_prompt(task, final_solution, execution_summary),
            
            # 变体3: 添加随机化的引导语
            self._create_guided_prompt(task, final_solution, execution_summary),
            
            # 变体4: 简化版本
            self._create_simplified_prompt(task, final_solution, execution_summary),
            
            # 变体5: 详细分析版本
            self._create_detailed_prompt(task, final_solution, execution_summary)
        ]
        
        # 根据run_index选择不同的变体，确保每次运行使用不同的prompt
        return prompt_variants[run_index % len(prompt_variants)]
    
    def _create_reordered_prompt(self, task: str, final_solution: str, execution_summary: str) -> str:
        """创建重新排列评估维度的prompt"""
        return f"""请评估以下AI代理的任务完成情况。

任务描述: {task}

代理的最终回答: {final_solution}

执行过程摘要: {execution_summary}

评估要求:
1. 首先评估答案的完整性和准确性
2. 然后评估逻辑推理的合理性
3. 最后评估整体的任务完成度

请给出0-1之间的分数，并详细说明评分理由。

输出格式:
{{"score": [0-1之间的数值], "reason": "[详细的评分理由]"}}"""

    def _create_guided_prompt(self, task: str, final_solution: str, execution_summary: str) -> str:
        """创建带引导语的prompt"""
        return f"""作为一个专业的AI评估专家，请仔细分析以下任务完成情况。

【任务要求】
{task}

【代理回答】
{final_solution}

【执行详情】
{execution_summary}

【评估指导】
请从以下角度进行综合评估：
- 答案是否直接回应了任务要求？
- 提供的信息是否准确可靠？
- 逻辑推理过程是否清晰合理？
- 是否充分利用了可用的工具和信息？

请提供0-1之间的评分，其中1表示完美完成任务。

格式要求:
{{"score": [数值], "reason": "[评估理由]"}}"""

    def _create_simplified_prompt(self, task: str, final_solution: str, execution_summary: str) -> str:
        """创建简化版本的prompt"""
        return f"""任务: {task}

回答: {final_solution}

执行信息: {execution_summary}

请评分(0-1)并说明理由，输出JSON格式:

{{"score": [评分], "reason": "[理由]"}}"""

    def _create_detailed_prompt(self, task: str, final_solution: str, execution_summary: str) -> str:
        """创建详细分析版本的prompt"""
        return f"""【深度评估任务】

请对以下AI代理的表现进行全面评估：

目标任务:
{task}

代理输出:
{final_solution}

执行轨迹:
{execution_summary}

【评估维度】
1. 任务理解度: 代理是否正确理解了任务要求？
2. 信息准确性: 提供的信息是否事实正确？
3. 完整性: 回答是否涵盖了任务的所有关键方面？
4. 工具使用效率: 是否合理有效地使用了可用工具？
5. 逻辑连贯性: 推理过程是否逻辑清晰？

请综合以上维度给出总体评分(0-1)，并提供详细的分析说明。

输出JSON格式:
{{"score": [评分], "reason": "[详细分析]"}}"""
    
    def _calculate_average_scores_from_stability(self, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """参照mcp-bench的方法计算stability testing的平均分数"""
        if not all_scores:
            raise ValueError("No scores to average")
        
        # 定义需要平均的分数字段（如果存在的话）
        score_fields = [
            "task_fulfillment", "grounding",
            "tool_appropriateness", "parameter_accuracy", 
            "dependency_awareness", "parallelism_and_efficiency",
            "score"  # 添加总分字段
        ]
        
        # 定义分析文本字段，从第一个评估结果中保留
        analysis_fields = [
            "task_completion_analysis",
            "tool_selection_analysis", 
            "planning_effectiveness_and_efficiency_analysis",
            "reason"
        ]
        
        # 计算平均值
        averaged_result = {}
        
        # 从第一个结果复制非分数字段（包括分析文本）
        first_result = all_scores[0]
        for key in first_result:
            if key not in score_fields:
                averaged_result[key] = first_result[key]
        
        # 确保分析字段被保留，即使缺失
        for field in analysis_fields:
            if field not in averaged_result and field in first_result:
                averaged_result[field] = first_result[field]
        
        # 计算各个分数字段的平均值
        for field in score_fields:
            valid_scores = []
            for result in all_scores:
                if field in result and isinstance(result[field], (int, float)):
                    valid_scores.append(result[field])
            
            if valid_scores:
                averaged_result[field] = sum(valid_scores) / len(valid_scores)
            else:
                # 如果没有有效分数，使用0（除了总分字段）
                if field == "score":
                    averaged_result[field] = 0.0
        
        # 如果有子维度分数，重新计算聚合分数
        if all(field in averaged_result for field in ["task_fulfillment", "grounding"]):
            task_completion_scores = [
                averaged_result['task_fulfillment'], 
                averaged_result['grounding']
            ]
            averaged_result['task_completion_score'] = sum(task_completion_scores) / len(task_completion_scores)
        
        if all(field in averaged_result for field in ["tool_appropriateness", "parameter_accuracy"]):
            tool_selection_scores = [
                averaged_result['tool_appropriateness'], 
                averaged_result['parameter_accuracy']
            ]
            averaged_result['tool_selection_score'] = sum(tool_selection_scores) / len(tool_selection_scores)
        
        if all(field in averaged_result for field in ["dependency_awareness", "parallelism_and_efficiency"]):
            planning_scores = [
                averaged_result['dependency_awareness'], 
                averaged_result['parallelism_and_efficiency']
            ]
            averaged_result['planning_effectiveness_and_efficiency_score'] = sum(planning_scores) / len(planning_scores)
        
        return averaged_result