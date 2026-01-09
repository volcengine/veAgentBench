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

from typing import List, Dict, Any


class LiveMCPBenchTemplate:

    @staticmethod
    def evaluate_llm_judge_dimensions(
        task: str,
        key_points: str,
        response: str,
        tool_calls: str,
        tool_descriptions: str
    ):
        """基于mcp-bench LLMJudge的6维度评估模板"""
        

        return f"""You are an expert in evaluating the performance of a tool-use agent. The agent is designed to help a human user use multi-tools to complete a task. Given the user's task, the agent's final response, key points for task completion, and tool call history, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1. You must carefully check whether the information (e.g. the coordinates of the addresses) comes from the tool call, if the agent get it from the internal knowledge, it should be considered failed.
2: Some tasks require to create files to be considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
User Task: 
{task}

Key Points: 
{key_points}

Final Response: 
{response}

Tool Call History:
{tool_calls}

Tool Descriptions:
{tool_descriptions}
"""


        """分析工具调用的模板（保持向后兼容）"""
        return f"""Given an input query, the actual output, and a list of expected tools, analyze the tool calls made.

**Input Query**: {input_text}

**Actual Output**: {actual_output}

**Expected Tools**: {expected_tools}

Analyze and extract:
1. Which tools were actually called
2. The parameters passed to each tool
3. The results returned by each tool
4. Whether the expected tools were used

**
IMPORTANT: Please make sure to only return in JSON format.
**

JSON:
{{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{"param1": "value1"}},
            "result": "execution_result"
        }}
    ],
    "expected_tools": {expected_tools},
    "missing_tools": ["missing_tool1"],
    "unexpected_tools": ["unexpected_tool1"]
}}
"""