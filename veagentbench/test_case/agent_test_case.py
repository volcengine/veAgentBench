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
from ..metrics.mcp_bench.schema import MCPExecutionData
from veagentbench.evals.deepeval.test_case import ToolCall

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
    id: str = ''
    execution_data: Optional[MCPExecutionData] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    available_tools: List[ToolCall] = None
    planning_strategy: Optional[str] = None
    execution_mode: Optional[str] = None
    trace_data: Optional[str] = None
    total_round: Optional[int] = 1
    dependency_analysis: Optional[str] = 'NA'
    def __init__(
        self,
        input: str,
        id: str,
        actual_output: Optional[str] = None,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        tools_called: Optional[List[str]] = None,
        expected_tools: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        comments: Optional[str] = None,
        # Agent特有的参数
        execution_data: Optional[MCPExecutionData] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        available_tools: Optional[Dict[str, Any]] = None,
        planning_strategy: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ):
        """初始化Agent测试用例
        
        Args:
            input: 用户输入/任务描述（对应mcp-bench中的task）
            actual_output: Agent基于工具执行结果生成的最终综合答案
                          （对应mcp-bench中的final_solution，是LLM基于accumulated_information生成的综合总结）
            expected_output: 期望的输出结果
            context: 上下文信息
            retrieval_context: 检索上下文
            tools_called: 实际调用的工具列表
            expected_tools: 期望调用的工具列表
            additional_metadata: 额外的元数据
            comments: 测试用例注释
            
            # Agent特有参数
            execution_data: 工具执行数据（包含所有工具调用信息）
            agent_id: Agent标识符
            session_id: 会话标识符
            conversation_history: 对话历史
            available_tools: 可用工具定义
            planning_strategy: 规划策略
            execution_mode: 执行模式
        """
        # 调用父类构造函数
        super().__init__(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            tools_called=tools_called,
            expected_tools=expected_tools,
            additional_metadata=additional_metadata,
            comments=comments
        )
        self.id: str = id
        # Agent特有属性
        self.execution_data = execution_data
        self.agent_id = agent_id
        self.session_id = session_id
        self.conversation_history = conversation_history or []
        self.available_tools = available_tools or {}
        self.planning_strategy = planning_strategy
        self.execution_mode = execution_mode
        
        # 自动从execution_data提取tools_called（如果没有显式提供）
        if not self.tools_called and self.execution_data:
            self.tools_called = [
                exec_result.tool 
                for exec_result in self.execution_data.tool_executions
            ]
    
    @classmethod
    def from_llm_test_case(
        cls, 
        llm_test_case: LLMTestCase, 
        execution_data: Optional[MCPExecutionData] = None,
        **agent_kwargs
    ) -> 'AgentTestCase':
        """从LLMTestCase创建AgentTestCase
        
        Args:
            llm_test_case: 原始的LLMTestCase
            execution_data: 工具执行数据
            **agent_kwargs: Agent特有的参数
            
        Returns:
            AgentTestCase实例
        """
        return cls(
            input=llm_test_case.input,
            actual_output=llm_test_case.actual_output,
            expected_output=llm_test_case.expected_output,
            context=llm_test_case.context,
            retrieval_context=llm_test_case.retrieval_context,
            tools_called=llm_test_case.tools_called,
            expected_tools=llm_test_case.expected_tools,
            additional_metadata=llm_test_case.additional_metadata,
            comments=llm_test_case.comments,
            execution_data=execution_data,
            **agent_kwargs
        )
    
    def to_llm_test_case(self) -> LLMTestCase:
        """转换为标准的LLMTestCase
        
        Returns:
            LLMTestCase实例
        """
        return LLMTestCase(
            input=self.input,
            actual_output=self.actual_output,
            expected_output=self.expected_output,
            context=self.context,
            retrieval_context=self.retrieval_context,
            tools_called=self.tools_called,
            expected_tools=self.expected_tools,
            additional_metadata=self.additional_metadata,
            comments=self.comments
        )
    
    def get_execution_summary(self) -> str:
        """获取执行摘要
        
        Returns:
            执行摘要字符串
        """
        if not self.execution_data or not self.execution_data.tool_executions:
            return "No tools were executed."
        
        execution_results = self.execution_data.tool_executions
        successful_tools = [r for r in execution_results if r.success]
        failed_tools = [r for r in execution_results if not r.success]
        
        summary_parts = [
            f"Total rounds: {self.execution_data.total_rounds}",
            f"Tools executed: {len(execution_results)}",
            f"Successful: {len(successful_tools)}",
            f"Failed: {len(failed_tools)}"
        ]
        
        if successful_tools:
            successful_tool_names = [r.tool for r in successful_tools]
            summary_parts.append(f"Successful tools: {', '.join(successful_tool_names)}")
        
        if failed_tools:
            failed_tool_names = [r.tool for r in failed_tools]
            summary_parts.append(f"Failed tools: {', '.join(failed_tool_names)}")
        
        return "; ".join(summary_parts)
    
    def get_tool_execution_stats(self) -> Dict[str, Any]:
        """获取工具执行统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        if not self.execution_data or not self.execution_data.tool_executions:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'unique_tools': 0,
                'servers_used': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0
            }
        
        executions = self.execution_data.tool_executions
        successful = [e for e in executions if e.success]
        failed = [e for e in executions if not e.success]
        unique_tools = set(e.tool for e in executions)
        servers = set(e.server for e in executions if e.server)
        
        execution_times = [e.execution_time for e in executions if e.execution_time is not None]
        total_time = sum(execution_times) if execution_times else 0.0
        avg_time = total_time / len(execution_times) if execution_times else 0.0
        
        return {
            'total_executions': len(executions),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'success_rate': len(successful) / len(executions) if executions else 0.0,
            'unique_tools': len(unique_tools),
            'servers_used': len(servers),
            'total_execution_time': total_time,
            'average_execution_time': avg_time
        }
    
    def add_conversation_turn(self, role: str, content: str):
        """添加对话轮次
        
        Args:
            role: 角色（user/assistant/system）
            content: 内容
        """
        self.conversation_history.append({
            'role': role,
            'content': content
        })
    
    def get_conversation_context(self) -> str:
        """获取对话上下文字符串
        
        Returns:
            格式化的对话历史
        """
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for turn in self.conversation_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            context_parts.append(f"{role.upper()}: {content}")
        
        return "\n".join(context_parts)
    
    def validate(self) -> List[str]:
        """验证测试用例的完整性
        
        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []
        
        # 基础验证
        if not self.input:
            errors.append("Input cannot be empty")
        
        if not self.actual_output:
            errors.append("Actual output cannot be empty")
        
        # Agent特有验证
        if self.execution_data:
            if not isinstance(self.execution_data.tool_executions, list):
                errors.append("execution_data.tool_executions must be a list")
            
            if self.execution_data.total_rounds < 1:
                errors.append("execution_data.total_rounds must be >= 1")
            
            # if not (0.0 <= self.execution_data.planning_json_compliance <= 1.0):
            #     errors.append("execution_data.planning_json_compliance must be between 0.0 and 1.0")
        
        # 工具相关验证
        if self.tools_called and self.available_tools:
            for tool in self.tools_called:
                if tool not in self.available_tools:
                    errors.append(f"Tool '{tool}' not found in available_tools")
        
        return errors
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"AgentTestCase("
            f"input='{self.input[:50]}...', "
            f"agent_id='{self.agent_id}', "
            f"tools_executed={len(self.execution_data.tool_executions) if self.execution_data else 0}, "
            f"success_rate={self.get_tool_execution_stats()['success_rate']:.2%}"
            f")"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            包含所有信息的字典
        """
        base_dict = {
            'input': self.input,
            'actual_output': self.actual_output,
            'expected_output': self.expected_output,
            'context': self.context,
            'retrieval_context': self.retrieval_context,
            'tools_called': self.tools_called,
            'expected_tools': self.expected_tools,
            'additional_metadata': self.additional_metadata,
            'comments': self.comments,
        }
        
        # Agent特有信息
        agent_dict = {
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            'conversation_history': self.conversation_history,
            'available_tools': self.available_tools,
            'planning_strategy': self.planning_strategy,
            'execution_mode': self.execution_mode,
            'execution_stats': self.get_tool_execution_stats(),
            'execution_summary': self.get_execution_summary()
        }
        
        # 合并字典
        result = {**base_dict, **agent_dict}
        
        # 添加execution_data（如果存在）
        if self.execution_data:
            result['execution_data'] = {
                'tool_executions': [
                    {
                        'tool': exec_result.tool,
                        'parameters': exec_result.parameters,
                        'result': exec_result.result,
                        'success': exec_result.success,
                        'server': exec_result.server,
                        'execution_time': exec_result.execution_time,
                        'error_message': exec_result.error_message
                    }
                    for exec_result in self.execution_data.tool_executions
                ],
                'total_rounds': self.execution_data.total_rounds,
                # 'planning_json_compliance': self.execution_data.planning_json_compliance,
                'concrete_task_description': self.execution_data.concrete_task_description,
                'dependency_analysis': self.execution_data.dependency_analysis,
                'accumulated_information': self.execution_data.accumulated_information
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTestCase':
        """从字典创建AgentTestCase
        
        Args:
            data: 包含测试用例信息的字典
            
        Returns:
            AgentTestCase实例
        """
        # 提取execution_data
        execution_data = None
        if 'execution_data' in data:
            from ..metrics.mcp_tool.schema import ToolExecutionResult
            
            tool_executions = []
            for exec_data in data['execution_data'].get('tool_executions', []):
                tool_executions.append(ToolExecutionResult(
                    tool=exec_data.get('tool'),
                    parameters=exec_data.get('parameters', {}),
                    result=exec_data.get('result'),
                    success=exec_data.get('success', False),
                    server=exec_data.get('server'),
                    execution_time=exec_data.get('execution_time'),
                    error_message=exec_data.get('error_message')
                ))
            
            execution_data = MCPExecutionData(
                tool_executions=tool_executions,
                total_rounds=data['execution_data'].get('total_rounds', 1),
                # planning_json_compliance=data['execution_data'].get('planning_json_compliance', 1.0),
                concrete_task_description=data['execution_data'].get('concrete_task_description'),
                dependency_analysis=data['execution_data'].get('dependency_analysis'),
                accumulated_information=data['execution_data'].get('accumulated_information')
            )
        
        return cls(
            input=data.get('input', ''),
            actual_output=data.get('actual_output', ''),
            expected_output=data.get('expected_output'),
            context=data.get('context'),
            retrieval_context=data.get('retrieval_context'),
            tools_called=data.get('tools_called'),
            expected_tools=data.get('expected_tools'),
            additional_metadata=data.get('additional_metadata'),
            comments=data.get('comments'),
            execution_data=execution_data,
            agent_id=data.get('agent_id'),
            session_id=data.get('session_id'),
            conversation_history=data.get('conversation_history', []),
            available_tools=data.get('available_tools', {}),
            planning_strategy=data.get('planning_strategy'),
            execution_mode=data.get('execution_mode')
        )