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
import json
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

# Import local schema and template
from .schema import *
from .template import CustomMetricTemplate


class CustomMetric(BaseMetric):
    """Custom LLM evaluation metric with user-defined templates"""

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Optional[CustomMetricTemplate] = None,
        template_params: Optional[Dict[str, Any]] = None,
        evaluation_mode: str = "default",  # "default", "qa", "summarization", or "custom"
        custom_template_string: Optional[str] = None,
    ):
        """
        Initialize custom metric
        
        Args:
            threshold: Minimum score for success
            model: LLM model to use for evaluation
            include_reason: Whether to include reasoning in results
            async_mode: Whether to use async evaluation
            strict_mode: Whether to use strict scoring (threshold = 1.0)
            verbose_mode: Whether to include verbose logs
            evaluation_template: Custom evaluation template
            template_params: Additional parameters for template
            evaluation_mode: Predefined evaluation mode
            custom_template_string: Custom template string
        """
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.template_params = template_params or {}
        self.evaluation_mode = evaluation_mode
        
        # Initialize template based on mode or custom template
        if evaluation_template:
            self.evaluation_template = evaluation_template
        elif custom_template_string:
            self.evaluation_template = CustomMetricTemplate.create_template_from_string(custom_template_string)
        elif evaluation_mode == "qa":
            self.evaluation_template = CustomMetricTemplate.create_qa_template()
        elif evaluation_mode == "summarization":
            self.evaluation_template = CustomMetricTemplate.create_summarization_template()
        else:
            self.evaluation_template = CustomMetricTemplate()
        
        # Store evaluation details for later use
        self.evaluation_details = None
        self.custom_params = {}

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        **kwargs
    ) -> float:
        """Measure the quality using custom evaluation"""
        
        # check_llm_test_case_params(test_case, self._required_params, self)

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
                        **kwargs
                    )
                )
            else:
                # Perform custom evaluation
                self.evaluation_details = self._perform_evaluation(
                    **test_case.additional_metadata if test_case.additional_metadata else {},
                )
                
                # Calculate score based on evaluation
                self.score = self._calculate_score(self.evaluation_details)
                
                # Generate reason if requested
                if self.include_reason:
                    self.reason = self._generate_reason(self.evaluation_details, self.score)
                else:
                    self.reason = None
                
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Evaluation Details: {self.evaluation_details}",
                        f"Final Score: {self.score}",
                        f"Reason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        **kwargs
    ) -> float:
        """Async measure the quality using custom evaluation"""
        
        # check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            # Perform custom evaluation
            self.evaluation_details = await self._a_perform_evaluation(
                test_case.input,
                test_case.actual_output,
                test_case.expected_output,
                test_case.additional_metadata
                **kwargs
            )
            
            # Calculate score based on evaluation
            self.score = self._calculate_score(self.evaluation_details)
            
            # Generate reason if requested
            if self.include_reason:
                self.reason = await self._a_generate_reason(self.evaluation_details, self.score)
            else:
                self.reason = None
            
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Evaluation Details: {self.evaluation_details}",
                    f"Final Score: {self.score}",
                    f"Reason: {self.reason}",
                ],
            )

            return self.score

    def _perform_evaluation(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform custom evaluation"""
        
        # Format evaluation prompt
        evaluation_prompt = self.evaluation_template.format_evaluation_prompt(**kwargs)
        
        # Generate evaluation using LLM
        if self.using_native_model:
            evaluation_result, cost = self.model.generate(evaluation_prompt)
            self.evaluation_cost += cost
        else:
            try:
                evaluation_result = self.model.generate(evaluation_prompt)
            except Exception as e:
                
                evaluation_result = {}
        
        # Parse evaluation result
        try:
            if isinstance(evaluation_result, str):
                evaluation_details = json.loads(evaluation_result)
            else:
                evaluation_details = evaluation_result
        except json.JSONDecodeError:
            evaluation_details = {"evaluation": str(evaluation_result), "raw_result": evaluation_result}
        
        return evaluation_details

    async def _a_perform_evaluation(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Async perform custom evaluation"""
        
        # Format evaluation prompt
        evaluation_prompt = self.evaluation_template.format_evaluation_prompt(**kwargs)
        
        # Generate evaluation using LLM
        if self.using_native_model:
            evaluation_result, cost = await self.model.a_generate(evaluation_prompt)
            self.evaluation_cost += cost
        else:
            try:
                evaluation_result = await self.model.a_generate(evaluation_prompt)
            except Exception as e:
                # Fallback to basic evaluation if LLM fails
                evaluation_result = {}
        
        # Parse evaluation result
        try:
            if isinstance(evaluation_result, str):
                evaluation_details = json.loads(evaluation_result)
            else:
                evaluation_details = evaluation_result
        except json.JSONDecodeError:
            evaluation_details = {"evaluation": str(evaluation_result), "raw_result": evaluation_result}
        
        return evaluation_details

    def _calculate_score(self, evaluation_details: Dict[str, Any]) -> float:
        """Calculate score based on evaluation details"""
        
        # Try to extract score from evaluation details
        if 'score' in evaluation_details:
            return float(evaluation_details['score'])
        
        # Try to extract from common score fields
        score_fields = ['overall_score', 'final_score', 'quality_score', 'accuracy_score']
        for field in score_fields:
            if field in evaluation_details:
                score = float(evaluation_details[field])
                # Normalize score if it's not in 0-1 range
                if score > 1.0:
                    score = score / 10.0  # Assume 0-10 scale
                return max(0.0, min(1.0, score))
        
        # Fallback: calculate score based on available metrics
        return self._calculate_fallback_score(evaluation_details)

    def _calculate_fallback_score(self, evaluation_details: Dict[str, Any]) -> float:
        """Calculate fallback score when no explicit score is provided"""
        
        # Look for numeric scores in evaluation details
        numeric_scores = []
        for key, value in evaluation_details.items():
            if isinstance(value, (int, float)) and 'score' in key.lower():
                score = float(value)
                if score > 1.0:
                    score = score / 10.0
                numeric_scores.append(score)
        
        if numeric_scores:
            return np.mean(numeric_scores)
        
        # Very basic fallback: check for positive/negative indicators
        evaluation_text = str(evaluation_details).lower()
        positive_words = ['good', 'excellent', 'accurate', 'complete', 'clear', 'well']
        negative_words = ['poor', 'bad', 'inaccurate', 'incomplete', 'unclear']
        
        positive_count = sum(1 for word in positive_words if word in evaluation_text)
        negative_count = sum(1 for word in negative_words if word in evaluation_text)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral score
        
        return positive_count / (positive_count + negative_count)

    def _generate_reason(self, evaluation_details: Dict[str, Any], score: float) -> str:
        """Generate reason for the score"""
        
        # Try to extract reason from evaluation details
        reason_fields = ['reason', 'explanation', 'summary', 'conclusion']
        for field in reason_fields:
            if field in evaluation_details:
                return str(evaluation_details[field])
        
        # Generate reason based on evaluation
        evaluation_text = str(evaluation_details.get('evaluation', ''))
        if evaluation_text:
            return f"Score: {score:.2f}. Evaluation: {evaluation_text[:200]}..."
        
        return f"Score: {score:.2f} based on custom evaluation criteria."

    async def _a_generate_reason(self, evaluation_details: Dict[str, Any], score: float) -> str:
        """Async generate reason for the score"""
        return self._generate_reason(evaluation_details, score)

    def _fallback_evaluation(self, input_text: str, actual_output: str, expected_output: str) -> Dict[str, Any]:
        """Fallback evaluation when LLM fails"""
        
        # Basic text similarity calculation
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, actual_output, expected_output).ratio()
        
        return {
            "evaluation": "Basic similarity evaluation due to LLM failure",
            "similarity_score": similarity,
            "fallback": True,
            "input_length": len(input_text),
            "output_length": len(actual_output),
            "expected_length": len(expected_output)
        }

    def is_successful(self) -> bool:
        """Check if evaluation is successful"""
        if hasattr(self, 'error') and self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Custom Metric"

    def set_custom_params(self, params: Dict[str, Any]) -> None:
        """Set custom parameters for evaluation"""
        self.custom_params.update(params)
        self.template_params.update(params)

    def get_evaluation_details(self) -> Dict[str, Any]:
        """Get detailed evaluation results"""
        return {
            "score": getattr(self, 'score', 0.0),
            "reason": getattr(self, 'reason', None),
            "evaluation_details": self.evaluation_details,
            "threshold": self.threshold,
            "success": self.is_successful(),
            "custom_params": self.custom_params
        }
