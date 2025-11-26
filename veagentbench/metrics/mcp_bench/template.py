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

from veagentbench.evals.deepeval.test_case import ToolCall

class MCPToolTemplate:
    @staticmethod
    def evaluate_mcp_benchmark_task(
        task_description: str,
        expected_behavior: str,
        actual_output: str,
        tool_calls: List[Dict[str, Any]]
    ):
        """基于mcp-bench的评估模板，评估MCP基准任务"""
        return f"""Evaluate an MCP (Model Context Protocol) benchmark task based on the task description, expected behavior, actual output, and tool calls made.

**TASK DESCRIPTION**: {task_description}

**EXPECTED BEHAVIOR**: {expected_behavior}

**ACTUAL OUTPUT**: {actual_output}

**TOOL CALLS MADE**: {tool_calls}

You are an expert evaluator judging the quality of an AI agent's multi-server tool-based task execution.

Evaluate the following aspects based on evidence from the task, solution, and tool usage:

### 1. Task Understanding (0.0-1.0)
How well did the agent understand what was being asked?
- Did it identify the key requirements correctly?
- Did it recognize the scope and constraints of the task?

### 2. Tool Selection (0.0-1.0) 
How appropriate were the tools selected for this task?
- Were the right tools chosen for each subtask?
- Were any important tools missed?
- Were any unnecessary tools used?

### 3. Tool Usage (0.0-1.0)
How correctly were the selected tools used?
- Were parameters accurate and complete?
- Were tools called in the right sequence?
- Were dependencies handled properly?

### 4. Output Quality (0.0-1.0)
How well does the final output meet the expected behavior?
- Is the information accurate and complete?
- Is it properly formatted and structured?
- Does it address all aspects of the task?

### 5. Efficiency (0.0-1.0)
How efficiently was the task completed?
- Were redundant tool calls avoided?
- Was parallelization used when possible?
- Was the execution path optimal?

**IMPORTANT**: 
- Score each dimension from 0.0 to 1.0 (not 1-10)
- Be objective and evidence-based
- Focus on actual performance, not potential
- Consider the complexity of the task when evaluating

**
IMPORTANT: Please make sure to only return in JSON format.
**

JSON:
{{
    "task_understanding_score": 0.0,
    "tool_selection_score": 0.0,
    "tool_usage_score": 0.0,
    "output_quality_score": 0.0,
    "efficiency_score": 0.0,
    "overall_score": 0.0,
    "reason": "Detailed evaluation explaining the scores for each aspect and overall performance"
}}
"""

    @staticmethod
    def evaluate_llm_judge_dimensions(
        task: str,
        final_solution: str,
        execution_summary: str,
        total_rounds: int,
        available_tools: Dict[str, Any],
        concrete_task_description: str = None,
        dependency_analysis: str = None,
        expected_tool_calls: List[Dict[str, Any]] = None
    ):
        """基于mcp-bench LLMJudge的6维度评估模板"""
        
        # 任务描述部分
        if concrete_task_description:
            task_section = f"""
**TASK PRESENTED TO AGENT**: "{task}"

**CONCRETE TASK REFERENCE (For evaluation context only)**: 
Note: The agent did NOT see this concrete version. It only saw the task above.
The task visible for the agent is the fuzzy version of the concrete task.
The agent's interpretation of the fuzzy task may differ but still be valid.
"{concrete_task_description}"
"""
        else:
            task_section = f'**ORIGINAL TASK**: "{task}"'
        
        # 依赖分析部分
        dependency_section = ""
        if dependency_analysis:
            dependency_section = f"""
**DEPENDENCY ANALYSIS (Reference Only)**:
Note: This analysis was generated during task creation to help understand tool dependencies.
The agent did NOT see this analysis. It is provided as reference for evaluation purposes.
{dependency_analysis}
"""
        
        # 预期工具调用部分
        expected_tools_section = ""
        if expected_tool_calls:
            formatted_expected_calls = MCPToolTemplate._format_expected_tool_calls(expected_tool_calls)
            expected_tools_section = f"""
**EXPECTED TOOL CALLS (Reference Only)**:
Note: These are the expected/ideal tool calls for this task. The agent did NOT see these expectations.
Use this as a reference to evaluate the appropriateness and accuracy of the agent's tool usage.

{formatted_expected_calls}
"""

        return f"""You are an expert AI task execution evaluator. Score each dimension objectively based on evidence.

{task_section}

**EXECUTION SUMMARY**:
{execution_summary}

**FINAL SOLUTION**: "{final_solution}"

**TOTAL ROUNDS**: {total_rounds}

**AVAILABLE TOOLS** ({len(available_tools) if available_tools else 0} tools):
{MCPToolTemplate._format_available_tools(available_tools)}

{dependency_section}

{expected_tools_section}
           
### Evaluation Guidance using EXPECTED TOOL CALLS
If the "EXPECTED TOOL CALLS" section is provided above, you MUST use it as the ideal plan to evaluate:
- Tool Appropriateness: Compare the agent's actual tools against the expected tools. Penalize missing expected tools, using wrong/irrelevant tools, or unnecessary extra tools.
- Parameter Accuracy: 
    - For each expected tool, verify the agent's parameters (keys and values) against the expected ones. Penalize missing required parameters, incorrect values, extra irrelevant parameters, and wrong types.
    - If an expected tool call has no expected parameters, verify the agent's parameters against to the task context and avalible tools to determine if they are appropriate. Penalize missing required parameters, incorrect values, extra irrelevant parameters, and wrong types.
- Dependency and Order: If the expected sequence implies dependencies, penalize incorrect ordering or ignored dependencies.
- Redundancy: Penalize repeated or redundant calls that deviate from the expected minimal plan.

You should explicitly reference mismatches with EXPECTED TOOL CALLS when explaining "tool_appropriateness_reasoning" and "parameter_accuracy_reasoning".

### Task Completion Rubric (1–10 per subdimension)

1. **Task Fulfillment**
- 1–3: Perfectly completes 10-30% of requirements.
- 4–6: Perfectly completes 40-60% of requirements.
- 7–8: Perfectly completes 70-80% of requirements.
- 9–10: Perfectly completes 90-100% of requirements.

2. **Grounding**
- 1–3: 10-30% of claims are perfectly grounded in tool outputs.
- 4–6: 40-60% of claims are perfectly grounded in tool outputs.
- 7–8: 70-80% of claims are perfectly grounded in tool outputs.
- 9–10: 90-100% of claims are perfectly grounded in tool outputs.

---

### Tool Usage Rubric (1–10 per subdimension)

1. **Tool Appropriateness**
- 1–3: 10-30% of tools were perfectly selected for their subtasks.
- 4–6: 40-60% of tools were perfectly selected for their subtasks.
- 7–8: 70-80% of tools were perfectly selected for their subtasks.
- 9–10: 90-100% of tools were perfectly selected for their subtasks.

2. **Parameter Accuracy**
- 1–3: 10-30% of tool calls have perfectly accurate and complete parameters.
- 4–6: 40-60% of tool calls have perfectly accurate and complete parameters.
- 7–8: 70-80% of tool calls have perfectly accurate and complete parameters.
- 9–10: 90-100% of tool calls have perfectly accurate and complete parameters.

---

### Planning Effectiveness and Efficiency (1–10 per subdimension)

1. **Dependency Awareness**
- 1–3: 10-30% of dependency chains are perfectly executed.
- 4–6: 40-60% of dependency chains are perfectly executed.
- 7–8: 70-80% of dependency chains are perfectly executed.
- 9–10: 90-100% of dependency chains are perfectly executed.

2. **Parallelism and Efficiency**
- 1–3: More than 70% redundant calls OR less than 30% of parallelizable tasks were executed in parallel.
- 4–6: 40-60% redundant calls OR 40-60% of parallelizable tasks were executed in parallel.
- 7–8: 20-30% redundant calls AND 70-80% of parallelizable tasks were executed in parallel.
- 9–10: Less than 10% redundant calls AND 90-100% of parallelizable tasks were executed in parallel.

---

### PERCENTAGE-BASED SCORING SYSTEM:

**How to Calculate Scores:**
For each dimension, calculate the DEFECT RATE:
- Defect Rate = (Number of Issues / Total Opportunities) × 100%

Then map defect rate to score:
- 0-10% defects → Score 9-10 (Excellent to Perfect)
- 10-30% defects → Score 7-9 (Good performance)
- 30-50% defects → Score 5-7 (Average performance)
- 50-70% defects → Score 3-5 (Poor performance)
- 70-100% defects → Score 0-3 (Failed)

**How to Score:**
1. When evaluating percentages, be EXTREMELY STRICT about what counts as "perfectly executed"
2. "Perfectly" means ALL of the following must be true:
    - Correct tool selection (not just "works" but OPTIMAL choice)
    - Complete and accurate parameters (not just valid, but IDEAL)
    - Zero redundancy (no repeated or unnecessary calls)
    - Proper error handling (graceful recovery from ANY failure)
    - Efficient execution (parallel when possible, minimal rounds)
    - Concise output (no verbose explanations unless requested)
3. If ANY of the above is missing, that portion is NOT perfectly executed (counts as 0%)
4. Example: Task completed correctly but with 1 redundant call = that portion is 0% perfect

**KEY PRINCIPLES:**
1. ALWAYS calculate as percentage, NOT absolute numbers
2. 10 errors in 100 calls (10%) = same score as 1 error in 10 calls (10%)
3. Consider the OPPORTUNITY COUNT for each dimension:
    - Tool calls: How many total calls were made?
    - Parallelization: How many tasks COULD have been parallel?
    - Parameters: How many total parameters across all calls?
    - Claims: How many factual statements were made?
    - Dependencies: How many dependency relationships exist?
---

CRITICAL: Apply the STRICTEST interpretation of "perfectly executed". If there's ANY doubt, score lower.

**CONCRETE SCORING EXAMPLES WITH PROPORTIONS:**

Task Fulfillment:
- Completed 19/20 requirements (5% defect rate) = Score 9
- Completed 16/20 requirements (20% defect rate) = Score 8
- Completed 12/20 requirements (40% defect rate) = Score 6
- Completed 8/20 requirements (60% defect rate) = Score 4

Tool Appropriateness:
- 19/20 tools optimal (5% defect rate) = Score 9
- 16/20 tools optimal (20% defect rate) = Score 8
- 12/20 tools optimal (40% defect rate) = Score 6
- 8/20 tools optimal (60% defect rate) = Score 4

Parallelism & Efficiency:
- 9/10 parallelizable tasks done in parallel (10% missed) = Score 9
- 8/10 parallelizable tasks done in parallel (20% missed) = Score 8
- 6/10 parallelizable tasks done in parallel (40% missed) = Score 6
- 4/10 parallelizable tasks done in parallel (60% missed) = Score 4

Grounding:
- 19/20 claims supported by evidence (5% unsupported) = Score 9
- 16/20 claims supported by evidence (20% unsupported) = Score 8
- 12/20 claims supported by evidence (40% unsupported) = Score 6
- 8/20 claims supported by evidence (60% unsupported) = Score 4

Parameter Accuracy:
- 95/100 parameters perfect (5% defect rate) = Score 9
- 80/100 parameters perfect (20% defect rate) = Score 8
- 60/100 parameters perfect (40% defect rate) = Score 6
- 40/100 parameters perfect (60% defect rate) = Score 4

FORMAT NOTE: Text output when JSON not required in the task present to the agent = NO PENALTY (0% defect)
FORMAT NOTE: Missing JSON when explicitly required in the task present to the agent = Count as failed requirement

Remember: Most real-world executions should score 4-6. Scores of 8+ should be EXCEPTIONAL.

FINAL REMINDER BEFORE SCORING:
- Default to 4-5 unless you have strong evidence for higher
- Count ONLY truly perfect executions toward the percentage
- Be your most critical self - find flaws first, then acknowledge successes
- If you're considering a score above 7, re-examine for ANY imperfection
- Server count is IRRELEVANT - using more servers is NOT better

Please score based on COMPLETION PERCENTAGES and PROPORTIONAL SUCCESS, not absolute numbers.
Return your evaluation scoring and reasoning in this exact JSON format:
{{

"task_fulfillment_reasoning": "Explain how well the agent fulfilled the detailed task objectives, referencing specific content from the CONCRETE TASK DESCRIPTION and what percentage was completed.",
"grounding_reasoning": "Explain how well the agent's outputs were grounded in actual tool results versus unsupported claims.",
"tool_appropriateness_reasoning": "Explain whether the tools selected were appropriate for each subtask requirement.",
"parameter_accuracy_reasoning": "Explain the accuracy and completeness of parameters used in tool calls, noting any missing required parameters or incorrect values.",
"dependency_awareness_reasoning": "Explain how well the agent understood and respected task dependencies (what percentage of dependencies were handled correctly), refer to the provided dependency analysis section.",
"parallelism_efficiency_reasoning": "Explain the efficiency of execution, including use of parallelism and avoiding redundancy, refer to the provided dependency analysis section." 

"task_fulfillment": X,
"grounding": X,

"tool_appropriateness": X,
"parameter_accuracy": X,

"dependency_awareness": X,
"parallelism_and_efficiency": X,

}}

Return **only** the JSON object.
"""

    @staticmethod
    def _format_available_tools(available_tools: Dict[str, Any]) -> str:
        """格式化可用工具列表"""
        if not available_tools:
            return "No tools available"
        
        # 按服务器分组工具
        servers = {}
        for tool_name, tool_info in available_tools.items():
            # 安全地获取server属性
            server = getattr(tool_info, 'server', None) or tool_info.get('server', 'Unknown') if isinstance(tool_info, dict) else 'Unknown'
            if server not in servers:
                servers[server] = []
            
            # 安全地获取工具描述，截断过长的描述
            if hasattr(tool_info, 'description'):
                description = tool_info.description
            elif hasattr(tool_info, 'discription'):  # 处理拼写错误的属性名
                description = tool_info.discription
            elif isinstance(tool_info, dict):
                description = tool_info.get('description', tool_info.get('discription', 'No description available'))
            else:
                description = 'No description available'
            
            if description is None:
                description = 'No description available'
            if len(description) > 500:
                description = description[:500] + "..."
            
            servers[server].append({
                'name': tool_name,
                'description': description
            })
        
        # 格式化输出
        lines = []
        for server, tools in sorted(servers.items()):
            lines.append(f"[{server}] ({len(tools)} tools)")
            
            # 显示所有工具及其描述
            for tool in tools:
                lines.append(f"  - {tool['name']}: {tool['description']}")
            
            lines.append("")  # 服务器之间的空行
        
        return '\n'.join(lines).strip() if lines else "No tools available"
    
    @staticmethod
    def _format_expected_tool_calls(expected_tool_calls: List[Dict[str, Any]]) -> str:
        """格式化预期工具调用列表"""
        if not expected_tool_calls:
            return "No expected tool calls provided"
        
        lines = []
        for i, tool_call in enumerate(expected_tool_calls, 1):
            tool_name = tool_call.get('tool_name', tool_call.get('name', 'Unknown'))
            server = tool_call.get('server', 'Unknown')
            parameters = tool_call.get('input_parameters', tool_call.get('arguments', {}))
            description = tool_call.get('description', '')
            
            lines.append(f"{i}. **{tool_name}** (Server: {server})")
            
            if description:
                lines.append(f"   Purpose: {description}")
            
            if parameters:
                lines.append("   Expected Parameters:")
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, str) and len(param_value) > 100:
                        param_value = param_value[:100] + "..."
                    lines.append(f"     - {param_name}: {param_value}")
            else:
                lines.append("   Expected Parameters: None")
            
            lines.append("")  # 空行分隔
        
        return '\n'.join(lines).strip()

    @staticmethod
    def analyze_tool_calls(input_text: str, actual_output: str, expected_tools: List[str]):
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