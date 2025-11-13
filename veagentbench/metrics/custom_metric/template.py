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

from typing import Dict, Any, Optional, List
from string import Template


class CustomMetricTemplate:
    """Custom metric template that supports user-defined evaluation templates"""
    
    def __init__(self, 
                 evaluation_template: Optional[str] = None,
                 scoring_template: Optional[str] = None,
                 reason_template: Optional[str] = None):
        """
        Initialize custom metric template
        
        Args:
            evaluation_template: Template for evaluation prompt
            scoring_template: Template for scoring prompt  
            reason_template: Template for generating reason
        """
        self.evaluation_template = evaluation_template or self._default_evaluation_template()
        self.scoring_template = scoring_template or self._default_scoring_template()
        self.reason_template = reason_template or self._default_reason_template()
    
    def _default_evaluation_template(self) -> str:
        """Default evaluation template"""
        return """You are an expert evaluator. Please evaluate the following based on the given criteria:

**Evaluation Task:**
${evaluation_task}

**Input:**
${input}

**Expected Output:**
${expected_output}

**Actual Output:**
${actual_output}

**Evaluation Criteria:**
${evaluation_criteria}

**Additional Context:**
${additional_context}

Please provide a detailed evaluation following this format:
{
    "evaluation": "Your detailed evaluation here",
    "strengths": ["List of strengths"],
    "weaknesses": ["List of weaknesses"],
    "suggestions": ["List of improvement suggestions"]
}
"""
    
    def _default_scoring_template(self) -> str:
        """Default scoring template"""
        return """Based on the evaluation above, please provide a numerical score.

**Scoring Criteria:**
${scoring_criteria}

**Score Range:** ${score_range}
**Evaluation Details:**
${evaluation_details}

Please provide your score in this format:
{
    "score": ${score_value},
    "reason": "Brief explanation for the score"
}
"""
    
    def _default_reason_template(self) -> str:
        """Default reason template"""
        return """Given the evaluation and score, provide a concise summary explaining the score.

**Score:** ${score}
**Evaluation Summary:** ${evaluation_summary}

Provide a brief, clear reason for the score:
{
    "reason": "Your concise reason here"
}
"""
    
    def format_evaluation_prompt(self, 
                               input_text: str,
                               expected_output: str,
                               actual_output: str,
                               evaluation_task: str = "Evaluate the quality of the response",
                               evaluation_criteria: str = "Accuracy, relevance, and completeness",
                               additional_context: str = "",
                               **kwargs) -> str:
        """Format evaluation prompt with given parameters"""
        template = Template(self.evaluation_template)
        
        params = {
            'input': input_text,
            'expected_output': expected_output,
            'actual_output': actual_output,
            'evaluation_task': evaluation_task,
            'evaluation_criteria': evaluation_criteria,
            'additional_context': additional_context,
            **kwargs
        }
        
        return template.safe_substitute(params)
    
    def format_scoring_prompt(self,
                            evaluation_details: str,
                            scoring_criteria: str = "Higher scores indicate better quality",
                            score_range: str = "0.0 to 1.0",
                            score_value: str = "0.0",
                            **kwargs) -> str:
        """Format scoring prompt with given parameters"""
        template = Template(self.scoring_template)
        
        params = {
            'evaluation_details': evaluation_details,
            'scoring_criteria': scoring_criteria,
            'score_range': score_range,
            'score_value': score_value,
            **kwargs
        }
        
        return template.safe_substitute(params)
    
    def format_reason_prompt(self,
                           score: float,
                           evaluation_summary: str,
                           **kwargs) -> str:
        """Format reason prompt with given parameters"""
        template = Template(self.reason_template)
        
        params = {
            'score': str(score),
            'evaluation_summary': evaluation_summary,
            **kwargs
        }
        
        return template.safe_substitute(params)
    
    @staticmethod
    def create_template_from_string(template_string: str) -> 'CustomMetricTemplate':
        """Create a template from a string"""
        return CustomMetricTemplate(
            evaluation_template=template_string,
            scoring_template=None,
            reason_template=None
        )
    
    @staticmethod
    def create_qa_template() -> 'CustomMetricTemplate':
        """Create a question-answering evaluation template"""
        evaluation_template = """You are evaluating a question-answering system. Please assess the quality of the answer.

**Question:** ${input}

**Expected Answer:** ${expected_output}

**Actual Answer:** ${actual_output}

**Evaluation Focus:** ${evaluation_focus}

Please evaluate:
1. Accuracy - Does the answer contain correct information?
2. Completeness - Does the answer address all parts of the question?
3. Clarity - Is the answer clear and easy to understand?

Provide your evaluation in JSON format:
{
    "evaluation": "Detailed evaluation of the answer quality",
    "accuracy_score": "Score from 0-10 for accuracy",
    "completeness_score": "Score from 0-10 for completeness", 
    "clarity_score": "Score from 0-10 for clarity",
    "overall_assessment": "Overall quality assessment"
}
"""
        
        scoring_template = """Based on the evaluation scores:
Accuracy: ${accuracy_score}/10
Completeness: ${completeness_score}/10
Clarity: ${clarity_score}/10

Calculate an overall score (0.0-1.0) considering all aspects.
Provide the final score in JSON format:
{
    "score": ${final_score},
    "reason": "Brief explanation of the final score"
}
"""
        
        return CustomMetricTemplate(
            evaluation_template=evaluation_template,
            scoring_template=scoring_template
        )
    
    @staticmethod
    def create_summarization_template() -> 'CustomMetricTemplate':
        """Create a summarization evaluation template"""
        evaluation_template = """You are evaluating a text summarization system. Please assess the quality of the summary.

**Original Text:** ${input}

**Expected Summary:** ${expected_output}

**Generated Summary:** ${actual_output}

**Evaluation Criteria:**
1. Coverage - Does the summary capture the main points?
2. Conciseness - Is the summary brief yet informative?
3. Coherence - Is the summary well-structured and readable?
4. Accuracy - Does the summary preserve key information?

Provide your evaluation in JSON format:
{
    "evaluation": "Detailed evaluation of the summary quality",
    "coverage_score": "Score from 0-10 for coverage",
    "conciseness_score": "Score from 0-10 for conciseness",
    "coherence_score": "Score from 0-10 for coherence",
    "accuracy_score": "Score from 0-10 for accuracy",
    "key_points_captured": ["List of key points that were captured"],
    "key_points_missed": ["List of key points that were missed"]
}
"""
        
        scoring_template = """Based on the evaluation scores and analysis:
Coverage: ${coverage_score}/10
Conciseness: ${conciseness_score}/10  
Coherence: ${coherence_score}/10
Accuracy: ${accuracy_score}/10

Key Points Captured: ${key_points_captured}
Key Points Missed: ${key_points_missed}

Calculate an overall quality score (0.0-1.0).
Provide the final score in JSON format:
{
    "score": ${final_score},
    "reason": "Explanation considering all evaluation aspects"
}
"""
        
        return CustomMetricTemplate(
            evaluation_template=evaluation_template,
            scoring_template=scoring_template
        )
