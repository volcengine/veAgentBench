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
    
    check_llm_test_case_params,
)
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator


from veagentbench.test_case.agent_test_case import AgentTestCase
from veagentbench.evals.deepeval.models import DeepEvalBaseLLM

class PerformanceMetricStream(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,        
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        async_mode: bool = True,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None
    ):
        self.threshold = threshold
        self.async_mode = async_mode

        # outputs
        self.success = False
        self.score = 0.0
        self.reason = None
        self.verbose_logs = None
        self.evaluation_cost = 0  # no model cost
        
        self.time_to_first_token = 0
        self.time_to_completion = 0
      

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
            self._compute_metrics(test_case)
            return self.score

    def _compute_metrics(self, test_case: Union[LLMTestCase, AgentTestCase]):
        # 初始化基于trace_data的性能指标

        ttft_list = test_case.time_to_first_token
        e2e_list = test_case.time_to_compeletion
        if len(ttft_list) > 0:
            self.time_to_first_token = sum(ttft_list) / len(ttft_list)
        if len(e2e_list) > 0:
            self.time_to_completion = sum(e2e_list) / len(e2e_list)
        self.score = 1.0
        self.success = True
        self.score_breakdown = {
            "time_to_first_token": self.time_to_first_token,
            "time_to_completion": self.time_to_completion,
        }
        
    def is_successful(self) -> bool:
        return bool(self.success)
    
    @property
    def __name__(self):
        return "Performance Metric Stream"
