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

import json
from typing import Dict, List, Any

class ToolCallingEvaluator:
    def __init__(self, test_set_path: str):
        self.test_set_path = test_set_path
        self.test_cases = self.load_test_cases()
        self.results = {
            'total_tests': 0,
            'pass_at_1': 0,
            'exact_match': 0,
            'details': []
        }
    
    def load_test_cases(self) -> List[Dict]:
        """Load tool calling test cases from the test set"""
        with open(self.test_set_path, 'r') as f:
            test_set = json.load(f)
        
        return [test for test in test_set['tests'] if test['type'] == 'tool_calling']
    
    def evaluate(self, agent):
        """Evaluate agent's tool calling capabilities"""
        self.results['total_tests'] = len(self.test_cases)
        
        for test_case in self.test_cases:
            # Simulate agent processing the input
            actual_calls = agent.process_input(test_case['input'])
            
            # Compare with expected tool calls
            pass_at_1 = self.check_pass_at_1(actual_calls, test_case['expected_tool_calls'])
            exact_match = self.check_exact_match(actual_calls, test_case['expected_tool_calls'])
            
            # Record results
            self.results['pass_at_1'] += 1 if pass_at_1 else 0
            self.results['exact_match'] += 1 if exact_match else 0
            
            self.results['details'].append({
                'test_id': test_case['id'],
                'input': test_case['input'],
                'expected': test_case['expected_tool_calls'],
                'actual': actual_calls,
                'pass_at_1': pass_at_1,
                'exact_match': exact_match
            })
        
        # Calculate final metrics
        self.results['pass_at_1_rate'] = self.results['pass_at_1'] / self.results['total_tests']
        self.results['exact_match_rate'] = self.results['exact_match'] / self.results['total_tests']
        
        return self.results
    
    def check_pass_at_1(self, actual_calls: List[Dict], expected_calls: List[Dict]) -> bool:
        """Check if the first tool call matches expectations"""
        if not actual_calls:
            return False
        
        first_call = actual_calls[0]
        return self.compare_tool_calls(first_call, expected_calls[0])
    
    def check_exact_match(self, actual_calls: List[Dict], expected_calls: List[Dict]) -> bool:
        """Check if all tool calls exactly match expectations"""
        if len(actual_calls) != len(expected_calls):
            return False
        
        for actual, expected in zip(actual_calls, expected_calls):
            if not self.compare_tool_calls(actual, expected):
                return False
        
        return True
    
    def compare_tool_calls(self, actual: Dict, expected: Dict) -> bool:
        """Compare individual tool calls"""
        return (
            actual.get('tool') == expected.get('tool') and
            actual.get('parameters') == expected.get('parameters')
        )


# Mock agent implementation for testing
class MockAgent:
    def process_input(self, input_text: str) -> List[Dict]:
        """Simulate agent tool calling behavior"""
        # In a real implementation, this would call the actual agent
        # For now, we'll return mock responses based on input
        if "weather" in input_text:
            return [{
                "tool": "weather",
                "parameters": {
                    "location": "San Francisco",
                    "unit": "celsius"
                }
            }]
        return []


if __name__ == "__main__":
    # Example usage
    evaluator = ToolCallingEvaluator("testsets/core.json")
    agent = MockAgent()
    results = evaluator.evaluate(agent)
    
    print("Tool Calling Evaluation Results:")
    print(f"Pass@1 Rate: {results['pass_at_1_rate']:.2%}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.2%}")
    print("\nDetailed Results:")
    for detail in results['details']:
        print(f"Test {detail['test_id']}: Pass@1={detail['pass_at_1']}, Exact Match={detail['exact_match']}")
