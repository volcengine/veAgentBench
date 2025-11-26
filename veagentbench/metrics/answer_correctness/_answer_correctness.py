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

# Import local schema and template
from .schema import *
from .template import AnswerCorrectnessTemplate


class AnswerCorrectnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.25,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        weights: List[float] = [0.75, 0.25],  # [factuality, semantic_similarity]
        beta: float = 1.0,
        evaluation_template: Type[
            AnswerCorrectnessTemplate
        ] = AnswerCorrectnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.weights = weights
        self.beta = beta
        self.evaluation_template = evaluation_template
        
        # Validate weights
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

    def measure(
        self,
        test_case: LLMTestCase,
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
                # Generate statements for both answer and ground truth
                self.answer_statements = self._generate_statements(
                    test_case.input, test_case.actual_output
                )
                self.ground_truth_statements = self._generate_statements(
                    test_case.input, test_case.expected_output
                )
                
                # Generate verdicts
                self.verdicts: List[StatementVerdict] = (
                    self._generate_verdicts(
                        test_case.input,
                        self.answer_statements,
                        self.ground_truth_statements,
                    )
                )
                
                # Calculate factuality score
                self.factuality_score = self._calculate_factuality_score()
                
                # Calculate semantic similarity score (simplified for now)
                self.semantic_similarity_score = self._calculate_semantic_similarity(
                    test_case.actual_output, test_case.expected_output
                )
                
                # Calculate final score
                self.score = self._calculate_final_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Answer Statements:\n{prettify_list(self.answer_statements)}",
                        f"Ground Truth Statements:\n{prettify_list(self.ground_truth_statements)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Factuality Score: {self.factuality_score}",
                        f"Semantic Similarity Score: {self.semantic_similarity_score}",
                        f"Final Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
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
            # Generate statements for both answer and ground truth
            self.answer_statements = await self._a_generate_statements(
                test_case.input, test_case.actual_output
            )
            self.ground_truth_statements = await self._a_generate_statements(
                test_case.input, test_case.expected_output
            )
            
            # Generate verdicts
            self.verdicts: List[StatementVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input,
                    self.answer_statements,
                    self.ground_truth_statements,
                )
            )
            
            # Calculate factuality score
            self.factuality_score = self._calculate_factuality_score()
            
            # Calculate semantic similarity score (simplified for now)
            self.semantic_similarity_score = self._calculate_semantic_similarity(
                test_case.actual_output, test_case.expected_output
            )
            
            # Calculate final score
            self.score = self._calculate_final_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Answer Statements:\n{prettify_list(self.answer_statements)}",
                    f"Ground Truth Statements:\n{prettify_list(self.ground_truth_statements)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Factuality Score: {self.factuality_score}",
                    f"Semantic Similarity Score: {self.semantic_similarity_score}",
                    f"Final Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        verdicts_dict = [
            {"statement": verdict.statement, "verdict": verdict.verdict, "reason": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = self.evaluation_template.generate_reason(
            question=input,
            verdicts=verdicts_dict,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=AnswerCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerCorrectnessScoreReason = (
                    await self.model.a_generate(
                        prompt, schema=AnswerCorrectnessScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        verdicts_dict = [
            {"statement": verdict.statement, "verdict": verdict.verdict, "reason": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = self.evaluation_template.generate_reason(
            question=input,
            verdicts=verdicts_dict,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=AnswerCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerCorrectnessScoreReason = self.model.generate(
                    prompt, schema=AnswerCorrectnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_statements(self, question: str, text: str) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            question=question, answer=text
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=StatementGenerationOutput)
            self.evaluation_cost += cost
            return res.statements
        else:
            try:
                res: StatementGenerationOutput = await self.model.a_generate(
                    prompt, schema=StatementGenerationOutput
                )
                return res.statements
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    def _generate_statements(self, question: str, text: str) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            question=question, answer=text
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=StatementGenerationOutput)
            self.evaluation_cost += cost
            return res.statements
        else:
            try:
                res: StatementGenerationOutput = self.model.generate(
                    prompt, schema=StatementGenerationOutput
                )
                return res.statements
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    async def _a_generate_verdicts(
        self, question: str, answer_statements: List[str], ground_truth_statements: List[str]
    ) -> List[StatementVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            question=question,
            answer_statements=answer_statements,
            ground_truth_statements=ground_truth_statements,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=AnswerCorrectnessVerdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: AnswerCorrectnessVerdicts = await self.model.a_generate(
                    prompt, schema=AnswerCorrectnessVerdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                print('verdicts: %s'%res)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    StatementVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, question: str, answer_statements: List[str], ground_truth_statements: List[str]
    ) -> List[StatementVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            question=question,
            answer_statements=answer_statements,
            ground_truth_statements=ground_truth_statements,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=AnswerCorrectnessVerdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: AnswerCorrectnessVerdicts = self.model.generate(
                    prompt, schema=AnswerCorrectnessVerdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    StatementVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _calculate_factuality_score(self):
        if not self.verdicts:
            return 0.0
            
        tp_count = sum(1 for v in self.verdicts if v.verdict == "TP")
        fp_count = sum(1 for v in self.verdicts if v.verdict == "FP")
        fn_count = sum(1 for v in self.verdicts if v.verdict == "FN")
        
        # Calculate F-beta score
        if tp_count == 0:
            return 0.0
            
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        beta_squared = self.beta ** 2
        f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        
        return f_beta

    def _calculate_semantic_similarity(self, actual_output: str, expected_output: str):
        # Simplified semantic similarity calculation
        # In a real implementation, you would use embeddings or other similarity measures
        # For now, return a basic similarity based on common words
        actual_words = set(actual_output.lower().split())
        expected_words = set(expected_output.lower().split())
        
        if not actual_words and not expected_words:
            return 1.0
        if not actual_words or not expected_words:
            return 0.0
            
        intersection = len(actual_words.intersection(expected_words))
        union = len(actual_words.union(expected_words))
        
        return intersection / union if union > 0 else 0.0

    def _calculate_final_score(self):
        return np.average(
            [self.factuality_score, self.semantic_similarity_score],
            weights=self.weights,
        )

    def is_successful(self) -> bool:
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
        return "Answer Correctness"