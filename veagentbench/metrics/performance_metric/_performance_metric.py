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

from typing import Optional, List, Dict, Any, Type, Union
import asyncio
import json

from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCase, LLMTestCaseParams
from veagentbench.evals.deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator

from ..mcp_bench.schema import MCPExecutionData, ToolExecutionResult
from .schema import PerformanceSummary
from .template import PerformanceTemplate
from veagentbench.utils.extract_performance_data import extract_performance_data_from_trace
from veagentbench.test_case.agent_test_case import AgentTestCase

class PerformanceMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        model = None,
        model_name = None,
        evaluation_template: Type[PerformanceTemplate] = PerformanceTemplate,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.evaluation_template = evaluation_template

        # outputs
        self.success = False
        self.score = 0.0
        self.reason = None
        self.verbose_logs = None
        self.evaluation_cost = 0  # no model cost

        # breakdown - 核心时延指标
        self.end_to_end_duration = 0.0
        self.tool_call_duration = 0.0
        self.llm_call_duration = 0.0

    def measure(
        self,
        test_case: Union[LLMTestCase, AgentTestCase],
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = asyncio.get_event_loop()
                try:
                    loop.run_until_complete(
                        self.a_measure(
                            test_case,
                            _show_indicator=False,
                            _in_component=_in_component,
                        )
                    )
                except RuntimeError:
                    asyncio.run(
                        self.a_measure(
                            test_case,
                            _show_indicator=False,
                            _in_component=_in_component,
                        )
                    )
            else:
                self._compute_metrics(test_case)

            return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, AgentTestCase],
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            await self._compute_metrics(test_case)
            return self.score

    async def _compute_metrics(self, test_case: Union[LLMTestCase, AgentTestCase]):
        # 初始化基于trace_data的性能指标
        trace_performance_data = None
        end_to_end_duration = None
        tool_call_stats = None
        llm_call_stats = None
        
        # 优先使用trace_data进行性能分析（如果存在）
        trace_data_list = getattr(test_case, "trace_data", None)
        for trace_data in trace_data_list or []:
        
            try:
                # 解析trace_data（可能是JSON字符串）
                if isinstance(trace_data, str):
                    trace_data_obj = json.loads(trace_data)
                else:
                    trace_data_obj = trace_data
                
                # 使用前面实现的性能分析功能
                trace_performance_data = await extract_performance_data_from_trace(trace_data_obj)
                
                # 提取关键性能指标
                end_to_end_duration = trace_performance_data.get("end_to_end_duration")
                tool_call_stats = trace_performance_data.get("tool_call_stats")
                llm_call_stats = trace_performance_data.get("llm_call_stats")
                
            except Exception as e:
                # 如果trace_data分析失败，回退到传统的execution_data分析
                print(f"Warning: Failed to analyze trace_data: {e}")

        # Prefer execution_data on AgentTestCase-like objects
        execution_data: Optional[MCPExecutionData] = getattr(test_case, "execution_data", None)

        tool_execs: List[ToolExecutionResult] = []
        if execution_data and getattr(execution_data, "tool_executions", None):
            tool_execs = execution_data.tool_executions

        rounds = getattr(execution_data, "total_rounds", 1) if execution_data else 1

        # Aggregate statistics
        total_calls = len(tool_execs)
        success_calls = 0
        durations: List[float] = []

        for r in tool_execs:
            try:
                if getattr(r, "success", False):
                    success_calls += 1
                # duration field optional
                d = getattr(r, "duration", None)
                if d is None:
                    # fallback via timestamps if present
                    start_t = getattr(r, "start_time", None) or getattr(r, "start_ts", None)
                    end_t = getattr(r, "end_time", None) or getattr(r, "end_ts", None)
                    if start_t is not None and end_t is not None:
                        try:
                            durations.append(float(end_t) - float(start_t))
                        except Exception:
                            pass
                else:
                    durations.append(float(d))
            except Exception:
                continue

        # 计算核心时延指标
        if trace_performance_data and end_to_end_duration is not None:
            # 使用trace_data的端到端耗时
            self.end_to_end_duration = end_to_end_duration
        else:
            # 回退到传统计算
            self.end_to_end_duration = sum(durations) if durations else 0.0

        # 计算工具调用总耗时
        if tool_call_stats:
            self.tool_call_duration = sum(
                stats.get("total_duration", 0) 
                for stats in tool_call_stats.values()
            )
        else:
            self.tool_call_duration = sum(durations) if durations else 0.0

        # 计算LLM调用总耗时（使用真实的LLM耗时，排除工具调用）
        if llm_call_stats:
            # 优先使用纯LLM耗时（排除工具调用子span）
            self.llm_call_duration = llm_call_stats.get("pure_llm_duration", llm_call_stats.get("total_duration", 0))
        else:
            self.llm_call_duration = 0.0

        # 计算成功率
        success_rate = (success_calls / total_calls) if total_calls > 0 else 0.0
        avg_duration = (self.end_to_end_duration / max(total_calls, 1)) if total_calls > 0 else 0.0

        # 计算并发效率
        avg_calls_per_round = (total_calls / rounds) if rounds > 0 else total_calls
        if avg_duration > 0:
            parallel_efficiency = min(1.0, max(0.0, (avg_calls_per_round / (1.0 + avg_duration))))
        else:
            parallel_efficiency = success_rate

        # 评分算法：成功率（50%）+ 性能（30%）+ 效率（20%）
        performance_score = 1.0
        if self.end_to_end_duration > 0:
            # 性能评分（基于端到端耗时）
            if self.end_to_end_duration <= 30:
                performance_score = 1.0
            elif self.end_to_end_duration <= 120:
                performance_score = 0.8
            elif self.end_to_end_duration <= 300:
                performance_score = 0.6
            else:
                performance_score = max(0.2, 1.0 - (self.end_to_end_duration - 300) / 300)

        # 综合评分
        self.score = 1.0 * performance_score

        self.success = self.score >= self.threshold

        # 生成评价原因
        if self.include_reason:
            self.reason = self.evaluation_template.generate_reason(
                total_calls=total_calls,
                success_rate=success_rate,
                avg_duration=avg_duration,
                rounds=rounds,
                parallel_efficiency=parallel_efficiency,
                end_to_end_duration=self.end_to_end_duration,
                tool_call_duration=self.tool_call_duration,
                llm_call_duration=self.llm_call_duration,
            )

        # 详细日志
        log_steps = [
            f"Total Calls: {total_calls}",
            f"Success Calls: {success_calls}",
            f"Success Rate: {success_rate:.3f}",
            f"End-to-End Duration: {self.end_to_end_duration:.3f}s",
            f"Tool Call Duration: {self.tool_call_duration:.3f}s",
            f"LLM Call Duration: {self.llm_call_duration:.3f}s",
            f"Score: {self.score:.3f}, Threshold: {self.threshold}, Success: {self.success}",
        ]

        self.verbose_logs = construct_verbose_logs(self, steps=log_steps)
        self.score_breakdown = {
            'end_to_end_duration': self.end_to_end_duration,
            'tool_call_duration': self.tool_call_duration,
            'llm_call_duration': self.llm_call_duration,
        }

    def get_summary(self) -> PerformanceSummary:
        """获取性能摘要，包含核心时延指标"""
        return PerformanceSummary(
            total_calls=0,  # 为了兼容性保留
            success_calls=0,
            success_rate=0.0,
            fail_rate=0.0,
            total_duration=self.end_to_end_duration,
            avg_duration=0.0,
            rounds=0,
            parallel_efficiency=0.0,
            reason=self.reason,
            # 核心时延指标
            end_to_end_duration=self.end_to_end_duration,
            tool_call_duration=self.tool_call_duration,
            llm_call_duration=self.llm_call_duration,
        )

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Performance Metric"
