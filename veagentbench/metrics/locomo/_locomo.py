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

from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCaseParams
from veagentbench.test_case.agent_test_case import AgentTestCase
from veagentbench.evals.deepeval.metrics.utils import check_llm_test_case_params
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator

# Initialize stemmer
ps = PorterStemmer()

LENGTH_THRESHOLD = 5


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


class LocomoMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        metric_type: str = "f1",  # Options: "f1", "exact_match", "bert_score", "rouge"
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        model = None,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.metric_type = metric_type
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.tokenizer = SimpleTokenizer()

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
                # Calculate score based on metric type
                question_category = test_case.extra_fields.get('question_category', 2)
                self.score = self._calculate_score(
                    test_case.actual_output, 
                    test_case.expected_output,
                    question_category
                )
                self.reason = self._generate_reason(
                    test_case.actual_output, 
                    test_case.expected_output,
                    
                )
                self.success = self.score >= self.threshold

            return self.score

    async def a_measure(
        self,
        test_case: AgentTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        # check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            question_category = test_case.extra_fields.get('category', 2)
            # Calculate score based on metric type
            self.score = self._calculate_score(
                test_case.actual_output, 
                test_case.expected_output,
                question_category=question_category
            )
            self.reason = self._generate_reason(
                test_case.actual_output, 
                test_case.expected_output
            )
            self.success = self.score >= self.threshold

            return self.score

    def _calculate_score(self, prediction: str, ground_truth: str, question_category: str) -> float:
        """Calculate score based on the specified metric type and question category."""
        if self.metric_type == "f1":
            question_category = str(question_category)
            # For multi-hop questions, use special multi-answer F1 scoring
            if question_category == '1':
                return self._f1_multi_answer(prediction, ground_truth)
            elif question_category == '5':
                answer_key = eval(ground_truth)
                if len(prediction) == 1:
                    if 'a' in prediction:
                        prediction = answer_key['a']
                    else:
                        prediction = answer_key['b']
                elif len(prediction) == 3:
                    if '(a)' in prediction:
                        prediction = answer_key['a']
                    else:
                        prediction = answer_key['b']
                if 'no information available' in prediction.lower() or 'not mentioned' in prediction.lower():
                    return 1
                else:
                    return 0
            else:
                return self._f1_score(prediction, ground_truth)
        elif self.metric_type == "exact_match":
            return self._exact_match_score(prediction, ground_truth)
        elif self.metric_type == "bert_score":
            return self._bert_score(prediction, ground_truth)
        elif self.metric_type == "rouge":
            return self._rouge_score(prediction, ground_truth)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def _exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate exact match score."""
        prediction = self._normalize_answer(prediction)
        ground_truth = self._normalize_answer(ground_truth)
        return float(set(prediction.split()) == set(ground_truth.split()))

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score with stemming."""
        prediction_tokens = [ps.stem(w) for w in self._normalize_answer(prediction).split()]
        ground_truth_tokens = [ps.stem(w) for w in self._normalize_answer(ground_truth).split()]
        
        if not prediction_tokens and not ground_truth_tokens:
            return 1.0
        if not prediction_tokens or not ground_truth_tokens:
            return 0.0
            
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _f1_multi_answer(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score for multi-answer questions (multi-hop)."""
        # Split by comma to handle multiple answers
        predictions = [p.strip() for p in prediction.split(',')]
        ground_truths = [g.strip() for g in ground_truth.split(',')]
        
        # Calculate mean of max F1 scores for each ground truth
        return np.mean([max([self._f1_score(pred, gt) for pred in predictions]) for gt in ground_truths])

    def _bert_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate BERT score (simplified version using cosine similarity)."""
        try:
            from bert_score import score
            prediction = self._normalize_answer(prediction)
            ground_truth = self._normalize_answer(ground_truth)
            P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
            return max(0, F1[0].item())
        except ImportError:
            # Fallback to F1 score if bert_score is not available
            return self._f1_score(prediction, ground_truth)

    def _rouge_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate ROUGE-1 score."""
        try:
            from rouge import Rouge
            rouge = Rouge()
            prediction = ' '.join([ps.stem(w) for w in self._normalize_answer(prediction).split()])
            ground_truth = ' '.join([ps.stem(w) for w in self._normalize_answer(ground_truth).split()])
            
            try:
                scores = rouge.get_scores(prediction, ground_truth, avg=True)
                return scores["rouge-1"]["f"]
            except ValueError:  # "Hypothesis is empty."
                return 0.0
        except ImportError:
            # Fallback to F1 score if rouge is not available
            return self._f1_score(prediction, ground_truth)

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer text."""
        s = s.replace(',', "")
        
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _generate_reason(self, prediction: str, ground_truth: str) -> str:
        """Generate evaluation reason."""
        if not self.include_reason:
            return None
            
        score = self.score
        metric_name = self.metric_type.upper()
        
        if score >= self.threshold:
            return f"{metric_name} score of {score:.3f} meets the threshold of {self.threshold}."
        else:
            return f"{metric_name} score of {score:.3f} is below the threshold of {self.threshold}."

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
        return f"Locomo {self.metric_type.upper()}"


def get_or_create_event_loop():
    """Get or create event loop for async operations."""
    try:
        import asyncio
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
