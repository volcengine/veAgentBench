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

from typing import Optional, List, Dict, Any, Union
from veagentbench.evals.deepeval.test_case import LLMTestCase, LLMTestCaseParams
from veagentbench.evals.deepeval.test_case import ToolCall
from pydantic import Field, AliasChoices, model_validator


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
    
class ToolCallExpected(ToolCall):
    """期望的工具调用"""
    # tool_name: str
    # parameters: Dict[str, Any]
    # description: Optional[str] = None
    server: str = "default"


class AgentTestCase(LLMTestCase):
    """Agent测试用例类
    
    继承自LLMTestCase，专门用于Agent和MCP工具的评估。
    将MCPExecutionData集成到测试用例中，提供统一的数据结构。
    
    主要特点：
    - 继承LLMTestCase的所有功能
    - 集成MCPExecutionData，包含工具执行信息
    - 提供统一的测试用例接口
    - 支持Agent特有的评估需求
    """
    
    available_tools: Optional[Dict[str, ToolCall]] = Field(
        default=None,
        serialization_alias="availableTools",
        validation_alias=AliasChoices("availableTools", "available_tools"),
    )
    
    tools_called: Optional[List[ToolExecutionResult]] = Field(
        default=None,
        serialization_alias="toolsCalled",
        validation_alias=AliasChoices("toolsCalled", "tools_called"),
    )
    
    expected_tools: Optional[List[ToolCallExpected]] = Field(
        default=None,
        serialization_alias="expectedTools",
        validation_alias=AliasChoices("expectedTools", "expected_tools"),
    )
    dependency_analysis: Optional[str] = Field(
        default='',
        serialization_alias="dependency_analysis",
        validation_alias=AliasChoices("dependencyAnalysis", "dependency_analysis"),
    )
    
    total_round: Optional[int] = Field(
        default=1,
        serialization_alias="total_round",
        validation_alias=AliasChoices("totalRound", "total_round"),
    )
    
    @model_validator(mode="before")
    def validate_input(cls, data):
        return data
