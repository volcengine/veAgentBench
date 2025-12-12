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

import regex
import json
import string
import unicodedata
from typing import List, Optional, Union
import numpy as np
from collections import Counter
import os
from nltk.stem import PorterStemmer
import backoff
import openai
from openai import OpenAI

from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCaseParams
from veagentbench.test_case.agent_test_case import AgentTestCase
from veagentbench.evals.deepeval.metrics.utils import check_llm_test_case_params
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator
from veagentbench.evals.deepeval.utils import get_or_create_event_loop
from veagentbench.evals.deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from veagentbench.evals.deepeval.models import DeepEvalBaseLLM
import traceback
# Initialize stemmer
ps = PorterStemmer()

LENGTH_THRESHOLD = 5


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    """Make API call with backoff for rate limiting."""
    return client.chat.completions.create(**kwargs)


class LocomoLLMMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: str = "gpt-4o-mini",
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        model = None,
        model_name = None
    ):
        self.model_name = model_name
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.model = model
        self.reason = None
        self.score = None
        self.success = None
        self.error = None
        self.evaluation_cost = 0.0

    def measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

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
                # Use LLM-based evaluation instead of rule-based scoring
                self.score = self._evaluate_with_llm(
                    test_case.input,
                    test_case.expected_output,
                    test_case.actual_output
                )
                self.success = self.score >= self.threshold

            return self.score

    async def a_measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            # Use LLM-based evaluation instead of rule-based scoring
            self.score = await self._evaluate_with_llm(
                test_case.input,
                test_case.expected_output,
                test_case.actual_output
            )
            self.success = self.score >= self.threshold

            return self.score

    def _get_evaluation_prompt(self, question: str, gold_answer: str, response: str) -> str:
        """Generate evaluation prompt using the new scoring methodology."""
        return f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
(1) a question (posed by one user to another user), 
(2) a 'gold' (ground truth) answer, 
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""

    async def _evaluate_with_llm(self, question: str, gold_answer: str, response: str) -> float:
        """Evaluate using LLM instead of rule-based scoring."""
        try:
            # Generate evaluation prompt
            prompt = self._get_evaluation_prompt(question, gold_answer, response)
            
            evaluation_result, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            # Parse the result to extract CORRECT/WRONG
            if "CORRECT" in evaluation_result.upper():
                self.reason = self._generate_reason(question, gold_answer, response, True, evaluation_result)
                return 1.0
            elif "WRONG" in evaluation_result.upper():
                self.reason = self._generate_reason(question, gold_answer, response, False, evaluation_result)
                return 0.0
            else:
                # Fallback: try to parse JSON
                try:
                    # Extract JSON from the response
                    json_start = evaluation_result.find('{')
                    json_end = evaluation_result.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = evaluation_result[json_start:json_end]
                        result_json = json.loads(json_str)
                        if result_json.get("label", "").upper() == "CORRECT":
                            self.reason = self._generate_reason(question, gold_answer, response, True, evaluation_result)
                            return 1.0
                        else:
                            self.reason = self._generate_reason(question, gold_answer, response, False, evaluation_result)
                            return 0.0
                except:
                    # If parsing fails, use fallback scoring
                    return self._fallback_score(question, gold_answer, response)
                    
        except Exception as e:
            self.error = str(e)
            traceback.print_exc()
            # Fallback to rule-based scoring if LLM evaluation fails
            return self._fallback_score(question, gold_answer, response)

    def _fallback_score(self, question: str, gold_answer: str, response: str) -> float:
        """Fallback scoring method when LLM evaluation fails."""
        # Simple keyword matching as fallback
        gold_tokens = set(gold_answer.lower().split())
        response_tokens = set(response.lower().split())
        
        overlap = len(gold_tokens.intersection(response_tokens))
        if overlap > 0:
            self.reason = f"Fallback scoring: Found {overlap} overlapping keywords between gold answer and response."
            return 0.5  # Partial credit for overlap
        else:
            self.reason = "Fallback scoring: No overlapping keywords found."
            return 0.0

    def _generate_reason(self, question: str, gold_answer: str, response: str, is_correct: bool, evaluation_result: str) -> str:
        """Generate evaluation reason."""
        if not self.include_reason:
            return None
            
        status = "CORRECT" if is_correct else "WRONG"
        return f"""LocomoLLM Assessment:

Question: {question}
Gold Answer: {gold_answer}
Model Response: {response}

LLM Evaluation Result: {status}
Detailed Evaluation: {evaluation_result}

This assessment is based on LLM evaluation using the conversational memory criteria."""
    
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
        return "Locomo LLM"


def get_or_create_event_loop():
    """Get or create event loop for async operations."""
    try:
        import asyncio
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
