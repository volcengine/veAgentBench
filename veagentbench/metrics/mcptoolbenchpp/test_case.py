from typing import Optional, List, Dict, Any
from veagentbench.evals.deepeval.test_case import LLMTestCase
from .schema import MCPToolBenchExecutionData, ToolCallResult, ToolCallLabel


class MCPToolBenchTestCase(LLMTestCase):
    """
    MCPToolBench测试用例类
    
    继承自deepeval的LLMTestCase，专门用于MCP工具基准测试评估
    """
    
    def __init__(
        self,
        # 继承LLMTestCase的基础字段
        input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        comments: Optional[str] = None,
        
        # MCPToolBench特有字段
        execution_data: Optional[MCPToolBenchExecutionData] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        function_call_result: Optional[List[ToolCallResult]] = None,
        function_call_label: Optional[List[ToolCallLabel]] = None,
        mcp_tools_dict: Optional[Dict[str, List[str]]] = None,
        
        # 评估配置
        evaluation_trial_per_task: int = 1,
        k_values: Optional[List[int]] = None,
        
        # 元数据
        task_id: Optional[str] = None,
        category: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化MCPToolBench测试用例
        
        Args:
            input: 用户查询/输入
            actual_output: 实际输出结果
            expected_output: 期望输出结果
            context: 上下文信息
            retrieval_context: 检索上下文
            tools_called: 调用的工具列表
            additional_metadata: 额外元数据
            comments: 注释
            
            execution_data: 完整的执行数据
            tools: 可用工具列表
            function_call_result: 工具调用结果
            function_call_label: 工具调用标签
            mcp_tools_dict: MCP服务器工具映射
            
            evaluation_trial_per_task: 每个任务的评估试验次数
            k_values: k值列表，用于pass@k计算
            
            task_id: 任务ID
            category: 任务类别
            model: 使用的模型
        """
        
        # 如果没有提供execution_data，从其他参数构建
        if execution_data is None and (tools is not None or function_call_result is not None):
            execution_data = MCPToolBenchExecutionData(
                query=input,
                tools=tools or [],
                function_call_result=function_call_result or [],
                function_call_label=function_call_label or [],
                mcp_tools_dict=mcp_tools_dict,
                evaluation_trial_per_task=evaluation_trial_per_task
            )
        
        # 从execution_data提取tools_called
        if tools_called is None and execution_data and execution_data.function_call_result:
            tools_called = [result.name for result in execution_data.function_call_result]
        
        # 构建additional_metadata
        if additional_metadata is None:
            additional_metadata = {}
        
        additional_metadata.update({
            "task_id": task_id,
            "category": category,
            "model": model,
            "evaluation_trial_per_task": evaluation_trial_per_task,
            "k_values": k_values or [1],
            "has_execution_data": execution_data is not None
        })
        
        # 调用父类构造函数
        super().__init__(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            tools_called=tools_called,
            additional_metadata=additional_metadata,
            comments=comments
        )
        
        # 设置MCPToolBench特有属性
        self.execution_data = execution_data
        self.evaluation_trial_per_task = evaluation_trial_per_task
        self.k_values = k_values or [1]
        self.task_id = task_id
        self.category = category
        self.model = model
    
    @classmethod
    def from_mcptoolbench_data(
        cls,
        data: Dict[str, Any],
        actual_output: str = "",
        model: Optional[str] = None,
        task_id: Optional[str] = None,
        category: Optional[str] = None,
        evaluation_trial_per_task: int = 1,
        k_values: Optional[List[int]] = None
    ) -> "MCPToolBenchTestCase":
        """
        从MCPToolBench数据格式创建测试用例
        
        Args:
            data: MCPToolBench格式的数据
            actual_output: 实际输出
            model: 模型名称
            task_id: 任务ID
            category: 类别
            evaluation_trial_per_task: 评估试验次数
            k_values: k值列表
            
        Returns:
            MCPToolBenchTestCase实例
        """
        import json
        
        # 解析数据
        query = data.get("query", "")
        tools = data.get("tools", [])
        if isinstance(tools, str):
            tools = json.loads(tools)
        
        function_call_label = data.get("function_call_label", [])
        if isinstance(function_call_label, str):
            function_call_label = json.loads(function_call_label)
        
        mcp_tools_dict = data.get("mcp_tools_dict", {})
        if isinstance(mcp_tools_dict, str):
            mcp_tools_dict = json.loads(mcp_tools_dict)
        
        # 转换function_call_label为ToolCallLabel对象
        label_objects = []
        for label in function_call_label:
            label_objects.append(ToolCallLabel(
                tool_name=label.get("tool_name", ""),
                tool_result=label.get("tool_result"),
                result=label.get("result")
            ))
        
        # 创建执行数据
        execution_data = MCPToolBenchExecutionData(
            query=query,
            tools=tools,
            function_call_result=[],  # 将在评估时填充
            function_call_label=label_objects,
            mcp_tools_dict=mcp_tools_dict,
            evaluation_trial_per_task=evaluation_trial_per_task
        )
        
        return cls(
            input=query,
            actual_output=actual_output,
            execution_data=execution_data,
            task_id=task_id,
            category=category,
            model=model,
            evaluation_trial_per_task=evaluation_trial_per_task,
            k_values=k_values
        )
    
    def add_trial_result(
        self,
        trial_idx: int,
        function_call_result: List[ToolCallResult],
        if_pass: bool,
        tool_correctness: bool,
        parameter_correctness: bool
    ):
        """
        添加试验结果
        
        Args:
            trial_idx: 试验索引
            function_call_result: 工具调用结果
            if_pass: 是否通过
            tool_correctness: 工具正确性
            parameter_correctness: 参数正确性
        """
        if self.execution_data is None:
            return
        
        # 更新k_results
        while len(self.execution_data.k_results) <= trial_idx:
            self.execution_data.k_results.append(False)
        self.execution_data.k_results[trial_idx] = if_pass
        
        # 更新k_tool_correct_results
        while len(self.execution_data.k_tool_correct_results) <= trial_idx:
            self.execution_data.k_tool_correct_results.append(False)
        self.execution_data.k_tool_correct_results[trial_idx] = tool_correctness
        
        # 更新k_parameter_correct_results
        while len(self.execution_data.k_parameter_correct_results) <= trial_idx:
            self.execution_data.k_parameter_correct_results.append(False)
        self.execution_data.k_parameter_correct_results[trial_idx] = parameter_correctness
        
        # 如果是第一次试验，设置function_call_result
        if trial_idx == 0:
            self.execution_data.function_call_result = function_call_result
    
    def get_trial_statistics(self) -> Dict[str, Any]:
        """
        获取试验统计信息
        
        Returns:
            包含统计信息的字典
        """
        if self.execution_data is None:
            return {}
        
        return {
            "num_trials": len(self.execution_data.k_results),
            "num_passed": sum(self.execution_data.k_results),
            "num_tool_correct": sum(self.execution_data.k_tool_correct_results),
            "num_parameter_correct": sum(self.execution_data.k_parameter_correct_results),
            "pass_rate": sum(self.execution_data.k_results) / len(self.execution_data.k_results) if self.execution_data.k_results else 0,
            "tool_accuracy": sum(self.execution_data.k_tool_correct_results) / len(self.execution_data.k_tool_correct_results) if self.execution_data.k_tool_correct_results else 0,
            "parameter_accuracy": sum(self.execution_data.k_parameter_correct_results) / len(self.execution_data.k_parameter_correct_results) if self.execution_data.k_parameter_correct_results else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            字典表示
        """
        base_dict = super().to_dict()
        
        # 添加MCPToolBench特有字段
        base_dict.update({
            "execution_data": self.execution_data.dict() if self.execution_data else None,
            "evaluation_trial_per_task": self.evaluation_trial_per_task,
            "k_values": self.k_values,
            "task_id": self.task_id,
            "category": self.category,
            "model": self.model,
            "trial_statistics": self.get_trial_statistics()
        })
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPToolBenchTestCase":
        """
        从字典创建测试用例
        
        Args:
            data: 字典数据
            
        Returns:
            MCPToolBenchTestCase实例
        """
        # 提取execution_data
        execution_data = None
        if data.get("execution_data"):
            execution_data = MCPToolBenchExecutionData(**data["execution_data"])
        
        return cls(
            input=data.get("input", ""),
            actual_output=data.get("actual_output", ""),
            expected_output=data.get("expected_output"),
            context=data.get("context"),
            retrieval_context=data.get("retrieval_context"),
            tools_called=data.get("tools_called"),
            additional_metadata=data.get("additional_metadata"),
            comments=data.get("comments"),
            execution_data=execution_data,
            evaluation_trial_per_task=data.get("evaluation_trial_per_task", 1),
            k_values=data.get("k_values", [1]),
            task_id=data.get("task_id"),
            category=data.get("category"),
            model=data.get("model")
        )