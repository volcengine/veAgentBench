from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ToolCallResult(BaseModel):
    """单个工具调用结果"""
    id: str = Field(description="工具调用ID")
    name: str = Field(description="工具名称")
    input: Dict[str, Any] = Field(description="工具输入参数")
    output: Dict[str, Any] = Field(description="工具输出结果")
    status_code: int = Field(description="状态码，200表示成功")
    
    @property
    def success(self) -> bool:
        """工具调用是否成功"""
        return self.status_code == 200


class ToolCallLabel(BaseModel):
    """工具调用标签（ground truth）"""
    tool_name: str = Field(description="期望的工具名称")
    tool_result: Optional[Union[str, Dict[str, Any]]] = Field(description="期望的工具结果")
    result: Optional[Union[str, Dict[str, Any]]] = Field(description="期望的结果（兼容字段）")


class MCPToolBenchExecutionData(BaseModel):
    """MCPToolBench执行数据"""
    query: str = Field(description="用户查询")
    tools: List[Dict[str, Any]] = Field(description="可用工具列表")
    function_call_result: List[ToolCallResult] = Field(description="实际工具调用结果")
    function_call_label: List[ToolCallLabel] = Field(description="期望工具调用标签")
    mcp_tools_dict: Optional[Dict[str, List[str]]] = Field(default=None, description="MCP服务器工具映射")
    
    # 试验相关
    evaluation_trial_per_task: int = Field(default=1, description="每个任务的评估试验次数")
    k_results: List[bool] = Field(default_factory=list, description="k次试验的整体结果")
    k_tool_correct_results: List[bool] = Field(default_factory=list, description="k次试验的工具正确性结果")
    k_parameter_correct_results: List[bool] = Field(default_factory=list, description="k次试验的参数正确性结果")


class PassAtKVerdict(BaseModel):
    """Pass@K单次判定结果"""
    trial_idx: int = Field(description="试验索引")
    if_pass: bool = Field(description="是否通过整体评估")
    tool_correctness: bool = Field(description="工具选择是否正确")
    parameter_correctness: bool = Field(description="参数是否正确")
    function_call_result: List[ToolCallResult] = Field(description="工具调用结果")


class PassAtKVerdicts(BaseModel):
    """Pass@K所有判定结果"""
    verdicts: List[PassAtKVerdict] = Field(description="所有试验的判定结果")
    k: int = Field(description="k值")
    num_trials: int = Field(description="试验总数")
    num_passed: int = Field(description="通过的试验数")
    pass_at_k: float = Field(description="pass@k分数")


class ToolPassAtKVerdict(BaseModel):
    """Tool Pass@K单次判定结果"""
    trial_idx: int = Field(description="试验索引")
    tool_correctness: bool = Field(description="工具选择是否正确")
    selected_tools: List[str] = Field(description="选择的工具列表")
    expected_tools: List[str] = Field(description="期望的工具列表")


class ToolPassAtKVerdicts(BaseModel):
    """Tool Pass@K所有判定结果"""
    verdicts: List[ToolPassAtKVerdict] = Field(description="所有试验的工具判定结果")
    k: int = Field(description="k值")
    num_trials: int = Field(description="试验总数")
    num_tool_correct: int = Field(description="工具正确的试验数")
    tool_pass_at_k: float = Field(description="tool_pass@k分数")


class ParameterPassAtKVerdict(BaseModel):
    """Parameter Pass@K单次判定结果"""
    trial_idx: int = Field(description="试验索引")
    parameter_correctness: bool = Field(description="参数是否正确")
    tool_name: str = Field(description="工具名称")
    actual_parameters: Dict[str, Any] = Field(description="实际参数")
    expected_parameters: Optional[Dict[str, Any]] = Field(description="期望参数")


class ParameterPassAtKVerdicts(BaseModel):
    """Parameter Pass@K所有判定结果"""
    verdicts: List[ParameterPassAtKVerdict] = Field(description="所有试验的参数判定结果")
    k: int = Field(description="k值")
    num_trials: int = Field(description="试验总数")
    num_parameter_correct: int = Field(description="参数正确的试验数")
    parameter_pass_at_k: float = Field(description="parameter_pass@k分数")


class MCPToolBenchPPResults(BaseModel):
    """MCPToolBenchPP综合评估结果"""
    pass_at_k_results: Dict[int, PassAtKVerdicts] = Field(description="pass@k结果")
    tool_pass_at_k_results: Dict[int, ToolPassAtKVerdicts] = Field(description="tool_pass@k结果")
    parameter_pass_at_k_results: Dict[int, ParameterPassAtKVerdicts] = Field(description="parameter_pass@k结果")
    
    # 汇总统计
    total_tasks: int = Field(description="总任务数")
    total_trials: int = Field(description="总试验数")
    overall_success_rate: float = Field(description="整体成功率")
    tool_accuracy: float = Field(description="工具选择准确率")
    parameter_accuracy: float = Field(description="参数准确率")