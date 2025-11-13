"""
MCPToolBenchPP评估指标模块

基于deepeval架构实现的MCPToolBenchPP评估指标，包括：
- PassAtKMetric: Pass@K指标，评估整体任务完成率
- ToolPassAtKMetric: Tool Pass@K指标，评估工具调用的正确性
- ParameterPassAtKMetric: Parameter Pass@K指标，评估工具参数的正确性

这些指标遵循MCPToolBenchPP的评估逻辑，用于评估AI代理在工具调用任务中的表现。
"""

from .metrics.pass_at_k import PassAtKMetric, estimate_pass_at_k
from .metrics.tool_pass_at_k import ToolPassAtKMetric, estimate_tool_pass_at_k
from .metrics.parameter_pass_at_k import ParameterPassAtKMetric, estimate_parameter_pass_at_k
from .test_case import MCPToolBenchTestCase
from .schema import (
    # 基础数据结构
    ToolCallResult,
    MCPToolBenchExecutionData,
    
    # Pass@K相关
    PassAtKVerdict,
    PassAtKVerdicts,
    
    # Tool Pass@K相关
    ToolPassAtKVerdict,
    ToolPassAtKVerdicts,
    
    # Parameter Pass@K相关
    ParameterPassAtKVerdict,
    ParameterPassAtKVerdicts,
)

__version__ = "1.0.0"

__all__ = [
    # 指标类
    "PassAtKMetric",
    "ToolPassAtKMetric", 
    "ParameterPassAtKMetric",
    
    # 测试用例
    "MCPToolBenchTestCase",
    
    # 数据结构
    "ToolCallResult",
    "MCPToolBenchExecutionData",
    "PassAtKVerdict",
    "PassAtKVerdicts",
    "ToolPassAtKVerdict",
    "ToolPassAtKVerdicts",
    "ParameterPassAtKVerdict",
    "ParameterPassAtKVerdicts",
    
    # 工具函数
    "estimate_pass_at_k",
    "estimate_tool_pass_at_k",
    "estimate_parameter_pass_at_k",
]