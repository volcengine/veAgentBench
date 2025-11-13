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
Test script for CustomMetric
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from veagentbench.metrics.custom_metric import CustomMetric, CustomMetricTemplate
from veagentbench.evals.deepeval.test_case import LLMTestCase
from veagentbench.models.models import VolceOpenAI

llm_model = VolceOpenAI(
            # model="ep-20250618145117-ps5rq",
            model = os.environ.get("VOLCEMODEL"),
            temperature=0,
            base_url=os.environ.get("VOLCEBASEURL"),
            cost_per_input_token=0.000002,
            cost_per_output_token=0.000008,
            _openai_api_key=os.environ.get("ARK_API_KEY"),
            
            # api_key=os.environ.get("ARK_API_KEY"),
        )

def test_custom_template():
    """Test custom template"""
    print("=== Testing Custom Template ===")
    
    # Create custom template
    custom_template = """请评估以下回答的质量：

**问题:** ${input}
**期望答案:** ${expected_output}
**实际答案:** ${actual_output}

评估标准：
1. 准确性 - 信息是否正确

请以JSON格式返回评估结果：
{
    "score": 8,
    "resason": "回答总体准确，但缺少了一些细节信息。"
}
"""
    
    metric = CustomMetric(
        custom_template_string=custom_template,
        threshold=0.7,
        async_mode=False,
        model=llm_model
    )
    
    test_case = LLMTestCase(
        input="test",
        additional_metadata={
            "input_text": "解释深度学习",
            "actual_output": "深度学习是机器学习的一种方法。",
            "expected_output": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示。"
        }
    )
    
    score = metric.measure(test_case)
    
    print(f"Score: {score}")
    print(f"Success: {metric.is_successful()}")
    print(f"Evaluation Details: {metric.evaluation_details}")
    print()


def main():
    """Run all tests"""
    print("Testing CustomMetric Implementation")
    print("=" * 50)
    
    try:
        # test_basic_usage()
        test_custom_template()
        # test_summarization_mode()
        # test_template_params()
        # test_evaluation_details()
        # test_preset_templates()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
