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

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from veagentbench.evals.deepeval.test_case import ToolCall

class ToolExecutionResult(ToolCall):
    """工具执行结果"""
    # tool: str
    # parameters: Dict[str, Any]
    # discription: str = ""
    # result: Optional[Any] = None
    success: bool = False
    server: str = "default"
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    


class MCPExecutionData(BaseModel):
    """MCP执行数据 - 专门存放工具调用相关信息"""
    tool_executions: List[ToolExecutionResult]  # 工具执行结果列表
    total_rounds: int = 1  # 总执行轮数
    accumulated_information: Optional[str] = None  # 累积信息
    # planning_json_compliance: float = 1.0  # 规划JSON合规性
    concrete_task_description: Optional[str] = None  # 具体任务描述
    dependency_analysis: Optional[str] = None  # 依赖分析
    expected_tool_calls: List[Dict[str, Any]] = []  # 期望的工具调用列表


class LLMJudgeScores(BaseModel):
    """LLM评判分数 - 基于mcp-bench的6个维度"""
    # Task Completion维度
    task_fulfillment: float  # 任务完成度 (1-10)
    grounding: float  # 基于工具输出的依据性 (1-10)
    
    # Tool Usage维度  
    tool_appropriateness: float  # 工具选择适当性 (1-10)
    parameter_accuracy: float  # 参数准确性 (1-10)
    
    # Planning Effectiveness维度
    dependency_awareness: float  # 依赖关系意识 (1-10)
    parallelism_and_efficiency: float  # 并行性和效率 (1-10)
    
    # 聚合分数
    task_completion_score: float  # 任务完成聚合分数
    tool_selection_score: float  # 工具选择聚合分数
    planning_effectiveness_score: float  # 规划效率聚合分数
    
    # 分析文本
    task_fulfillment_reasoning: Optional[str] = None
    grounding_reasoning: Optional[str] = None
    tool_appropriateness_reasoning: Optional[str] = None
    parameter_accuracy_reasoning: Optional[str] = None
    dependency_awareness_reasoning: Optional[str] = None
    parallelism_efficiency_reasoning: Optional[str] = None


class ToolAccuracyMetrics(BaseModel):
    """工具准确性指标 - 基于mcp-bench的工具匹配指标"""
    input_schema_compliance: Optional[float] = None  # 输入schema合规性 (0-1)
    valid_tool_name_rate: Optional[float] = None  # 有效工具名称率 (0-1)
    execution_success_rate: Optional[float] = None  # 执行成功率 (0-1)
    valid_call_failure_rate: Optional[float] = None  # 有效调用失败率 (0-1)
    # planning_json_compliance: float = 1.0  # 规划JSON合规性 (0-1)


class ServerUtilizationMetrics(BaseModel):
    """服务器利用率指标"""
    server_count: int  # 使用的服务器数量
    cross_server_coordination: bool  # 是否有跨服务器协调
    server_distribution: Dict[str, int]  # 服务器分布


class MCPBenchmarkEvaluation(BaseModel):
    """MCP基准测试评估结果"""
    task_understanding_score: float
    tool_selection_score: float
    tool_usage_score: float
    output_quality_score: float
    efficiency_score: float
    overall_score: float
    reason: str


class MCPToolEvaluationResult(BaseModel):
    """完整的MCP工具评估结果"""
    # LLM评估指标
    llm_scores: LLMJudgeScores
    
    # 工具匹配指标
    tool_accuracy_metrics: ToolAccuracyMetrics
    
    # 服务器利用率指标
    server_utilization_metrics: ServerUtilizationMetrics
    
    # 执行数据
    execution_data: MCPExecutionData
    
    # 综合评估
    final_score: float  # 最终综合分数 (0-1)
    success: bool  # 是否达到阈值
    reason: str  # 详细评估原因
    
    # 元数据
    evaluation_timestamp: Optional[float] = None
    threshold: float = 0.7


class MCPToolScoreReason(BaseModel):
    """MCP工具评分原因"""
    reason: str


# 兼容性别名

MCPToolVerdicts = List[ToolExecutionResult]
ToolCallAnalysis = List[ToolExecutionResult]