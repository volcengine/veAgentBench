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
Simple test script for CustomMetric template functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from veagentbench.metrics.custom_metric.template import CustomMetricTemplate


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
