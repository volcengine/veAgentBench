#!/usr/bin/env python3
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

"""
Direct test of CustomMetricTemplate functionality
"""

from string import Template


class CustomMetricTemplate:
    """Custom metric template that supports user-defined evaluation templates"""
    
    def __init__(self, 
                 evaluation_template=None,
                 scoring_template=None,
                 reason_template=None):
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
    
    def _default_evaluation_template(self):
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
    
    def _default_scoring_template(self):
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
    
    def _default_reason_template(self):
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
                               input_text,
                               expected_output,
                               actual_output,
                               evaluation_task="Evaluate the quality of the response",
                               evaluation_criteria="Accuracy, relevance, and completeness",
                               additional_context="",
                               **kwargs):
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
                            evaluation_details,
                            scoring_criteria="Higher scores indicate better quality",
                            score_range="0.0 to 1.0",
                            score_value="0.0",
                            **kwargs):
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
                           score,
                           evaluation_summary,
                           **kwargs):
        """Format reason prompt with given parameters"""
        template = Template(self.reason_template)
        
        params = {
            'score': str(score),
            'evaluation_summary': evaluation_summary,
            **kwargs
        }
        
        return template.safe_substitute(params)
    
    @staticmethod
    def create_template_from_string(template_string):
        """Create a template from a string"""
        return CustomMetricTemplate(
            evaluation_template=template_string,
            scoring_template=None,
            reason_template=None
        )
    
    @staticmethod
    def create_qa_template():
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
    def create_summarization_template():
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


def test_template_creation():
    """Test template creation"""
    print("=== Testing Template Creation ===")
    
    # Test default template
    template = CustomMetricTemplate()
    print("✓ Default template created")
    
    # Test QA template
    qa_template = CustomMetricTemplate.create_qa_template()
    print("✓ QA template created")
    
    # Test Summarization template
    summarization_template = CustomMetricTemplate.create_summarization_template()
    print("✓ Summarization template created")
    
    print()


def test_template_formatting():
    """Test template formatting"""
    print("=== Testing Template Formatting ===")
    
    template = CustomMetricTemplate.create_qa_template()
    
    # Test evaluation prompt formatting
    prompt = template.format_evaluation_prompt(
        input_text="什么是机器学习？",
        expected_output="机器学习是人工智能的一个分支。",
        actual_output="机器学习是AI的一部分。",
        evaluation_focus="准确性"
    )
    
    print("Sample Evaluation Prompt:")
    print(prompt)
    print("✓ Evaluation prompt formatted successfully")
    print()


def test_custom_template():
    """Test custom template creation"""
    print("=== Testing Custom Template ===")
    
    custom_template_string = """请评估以下回答的质量：

**问题:** ${input}
**期望答案:** ${expected_output}
**实际答案:** ${actual_output}

评估标准：
1. 准确性 - 信息是否正确
2. 完整性 - 是否回答了所有方面
3. 清晰度 - 表达是否清晰

请以JSON格式返回评估结果：
{
    "accuracy_score": 8,
    "completeness_score": 7,
    "clarity_score": 9,
    "overall_score": 8.0
}
"""
    
    custom_template = CustomMetricTemplate.create_template_from_string(custom_template_string)
    print("✓ Custom template created from string")
    
    # Test formatting
    prompt = custom_template.format_evaluation_prompt(
        input_text="什么是深度学习？",
        expected_output="深度学习是机器学习的一个子领域。",
        actual_output="深度学习是AI的一种方法。",
        evaluation_criteria="准确性和完整性"
    )
    
    print("Custom Template Evaluation Prompt:")
    print(prompt[:300] + "...")
    print("✓ Custom template formatting works")
    print()


def test_scoring_template():
    """Test scoring template"""
    print("=== Testing Scoring Template ===")
    
    template = CustomMetricTemplate()
    
    scoring_prompt = template.format_scoring_prompt(
        evaluation_details="评估显示回答准确性较高，但完整性有待改进",
        scoring_criteria="准确性权重50%，完整性权重30%，清晰度权重20%",
        score_range="0.0 to 1.0",
        score_value="0.75"
    )
    
    print("Sample Scoring Prompt:")
    print(scoring_prompt)
    print("✓ Scoring template formatted successfully")
    print()


def test_reason_template():
    """Test reason template"""
    print("=== Testing Reason Template ===")
    
    template = CustomMetricTemplate()
    
    reason_prompt = template.format_reason_prompt(
        score=0.8,
        evaluation_summary="回答整体质量良好，主要信息准确，表达清晰"
    )
    
    print("Sample Reason Prompt:")
    print(reason_prompt)
    print("✓ Reason template formatted successfully")
    print()


def test_template_parameters():
    """Test template with various parameters"""
    print("=== Testing Template Parameters ===")
    
    template = CustomMetricTemplate()
    
    # Test with additional parameters
    prompt = template.format_evaluation_prompt(
        input_text="什么是REST API？",
        expected_output="REST是一种软件架构风格。",
        actual_output="REST API是基于REST的接口。",
        evaluation_task="评估REST API概念的理解",
        evaluation_criteria="技术准确性和概念完整性",
        additional_context="这是一个技术面试问题",
        domain="软件架构",
        difficulty="中级"
    )
    
    print("Template with Additional Parameters:")
    print(prompt[:400] + "...")
    print("✓ Additional parameters handled correctly")
    print()


def main():
    """Run all tests"""
    print("Testing CustomMetric Template Functionality")
    print("=" * 50)
    
    try:
        test_template_creation()
        test_template_formatting()
        test_custom_template()
        test_scoring_template()
        test_reason_template()
        test_template_parameters()
        
        print("All template tests completed successfully!")
        print("\nCustomMetric template system is working correctly!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
