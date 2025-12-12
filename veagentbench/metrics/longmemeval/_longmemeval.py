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

import os
import backoff
import openai
from openai import OpenAI
from typing import Optional, List, Type, Union
import numpy as np

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
from veagentbench.test_case.agent_test_case import AgentTestCase

# Import local schema and template
from .schema import *
from .template import LongMemEvalTemplate


# Model zoo for different evaluation models
MODEL_ZOO = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    """Make API call with backoff for rate limiting."""
    return client.chat.completions.create(**kwargs)


class LongMemEvalMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        model_name: str = 'gpt-4o-mini',
        strict_mode: bool = False,
        verbose_mode: bool = False,
        task_type: str = 'single-session-user',
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.task_type = task_type
        self.modle_name = model_name
        
        # Initialize model if provided, otherwise use evaluation_model
        if model:
            self.model, self.using_native_model = initialize_model(model)
        else:
            self.model = None
            self.using_native_model = False
            

    def measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                # Get task type from test case or use default
                task_type = test_case.extra_fields.get('task_type', self.task_type)
                
                # Evaluate the response
                self.evaluation_result = self._evaluate_response(
                    task_type,
                    test_case.input,
                    test_case.expected_output,
                    test_case.actual_output
                )
                
                # Calculate score based on evaluation result
                self.score = 1.0 if self.evaluation_result.is_correct else 0.0
                self.reason = self.evaluation_result.explanation
                self.success = self.score >= self.threshold
                
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Task Type: {task_type}",
                        f"Question: {test_case.input}",
                        f"Expected Answer: {test_case.expected_output}",
                        f"Model Response: {test_case.actual_output}",
                        f"Evaluation Result: {'Correct' if self.evaluation_result.is_correct else 'Incorrect'}",
                        f"Explanation: {self.evaluation_result.explanation}",
                        f"Final Score: {self.score}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            # Get task type from test case or use default
            task_type = test_case.extra_fields.get('task_type', self.task_type)
            
            # Evaluate the response asynchronously
            self.evaluation_result = await self._a_evaluate_response(
                task_type,
                test_case.input,
                test_case.expected_output,
                test_case.actual_output
            )
            
            # Calculate score based on evaluation result
            self.score = 1.0 if self.evaluation_result.is_correct else 0.0
            self.reason = self.evaluation_result.explanation
            self.success = self.score >= self.threshold
            
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Task Type: {task_type}",
                    f"Question: {test_case.input}",
                    f"Expected Answer: {test_case.expected_output}",
                    f"Model Response: {test_case.actual_output}",
                    f"Evaluation Result: {'Correct' if self.evaluation_result.is_correct else 'Incorrect'}",
                    f"Explanation: {self.evaluation_result.explanation}",
                    f"Final Score: {self.score}",
                ],
            )

            return self.score

    def _evaluate_response(
        self,
        task_type: str,
        question: str,
        correct_answer: str,
        model_response: str
    ) -> LongMemEvalEvaluationOutput:
        """Evaluate a single response using the new LLM scoring methodology."""
        # Use the new evaluation prompt
        prompt = LongMemEvalTemplate.get_evaluation_prompt(
            task_type, question, correct_answer, model_response
        )
        
        if self.evaluation_client:
            # Use OpenAI API for evaluation
            kwargs = {
                'model': self.evaluation_model_name,
                'messages': [
                    {"role": "user", "content": prompt}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': 100
            }
            
            completion = chat_completions_with_backoff(self.evaluation_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            
            # Parse the response to extract CORRECT/WRONG and explanation
            is_correct, explanation = self._parse_llm_evaluation(eval_response)
            
            return LongMemEvalEvaluationOutput(
                is_correct=is_correct,
                explanation=explanation
            )
        else:
            # Use the provided model for evaluation
            if self.using_native_model:
                res, cost = self.model.generate(
                    prompt, schema=LongMemEvalEvaluationOutput
                )
                self.evaluation_cost += cost
                return res
            else:
                try:
                    res: LongMemEvalEvaluationOutput = self.model.generate(
                        prompt, schema=LongMemEvalEvaluationOutput
                    )
                    return res
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return LongMemEvalEvaluationOutput(
                        is_correct=data.get('is_correct', False),
                        explanation=data.get('explanation', 'No explanation provided')
                    )

    def _parse_llm_evaluation(self, eval_response: str) -> tuple[bool, str]:
        """Parse LLM evaluation response to extract CORRECT/WRONG and explanation."""
        eval_response = eval_response.strip().lower()
        
        # Extract explanation (everything before CORRECT/WRONG)
        lines = eval_response.split('\n')
        explanation_lines = []
        final_label = None
        
        for line in lines:
            line = line.strip()
            if 'correct' in line and 'wrong' not in line:
                final_label = True
                break
            elif 'wrong' in line and 'correct' not in line:
                final_label = False
                break
            else:
                explanation_lines.append(line)
        
        explanation = ' '.join(explanation_lines).strip() or "No explanation provided"
        
        # If we couldn't find explicit CORRECT/WRONG, try to infer
        if final_label is None:
            if 'correct' in eval_response and 'wrong' not in eval_response:
                final_label = True
            elif 'wrong' in eval_response and 'correct' not in eval_response:
                final_label = False
            else:
                # Default to wrong if unclear
                final_label = False
                explanation = f"Could not parse evaluation response: {eval_response}"
        
        return final_label, explanation

    async def _a_evaluate_response(
        self,
        task_type: str,
        question: str,
        correct_answer: str,
        model_response: str
    ) -> LongMemEvalEvaluationOutput:
        """Asynchronously evaluate a single response using the evaluation model."""
        # Check if question is unanswerable
        abstention = '_abs' in question or task_type.endswith('-abs')

        # Use the provided model for evaluation
        prompt = LongMemEvalTemplate.generate_evaluation(
            task_type, question, correct_answer, model_response, abstention
        )
        
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=LongMemEvalEvaluationOutput
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: LongMemEvalEvaluationOutput = await self.model.a_generate(
                    prompt, schema=LongMemEvalEvaluationOutput
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return LongMemEvalEvaluationOutput(
                    is_correct=data.get('is_correct', False),
                    explanation=data.get('explanation', 'No explanation provided')
                )

    def is_successful(self) -> bool:
        """Check if the metric evaluation is successful."""
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return f"LongMemEval ({self.task_type})"
