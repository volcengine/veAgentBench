from typing import Optional, List, Union, Dict, Any
import math
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCase
from veagentbench.evals.deepeval.utils import get_or_create_event_loop, prettify_list

from ..test_case import MCPToolBenchTestCase
from ..schema import (
    ToolPassAtKVerdict, 
    ToolPassAtKVerdicts,
    ToolCallResult,
    MCPToolBenchExecutionData
)


class ToolPassAtKMetric(BaseMetric):
    """
    Tool Pass@K指标 - 评估工具选择的正确性
    
    基于MCPToolBenchPP的tool_pass@k评估逻辑，计算在k次试验中至少有一次
    正确选择工具的概率。
    """
    
    def __init__(
        self,
        k: int = 1,
        threshold: float = 1.0,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        evaluation_trial_per_task: int = 1,
        tool_matching_strategy: str = "exact"  # "exact", "subset", "overlap"
    ):
        """
        初始化Tool Pass@K指标
        
        Args:
            k: k值，计算tool_pass@k时使用
            threshold: 阈值，必须为1.0（tool_pass@k是二元指标）
            include_reason: 是否包含详细原因
            async_mode: 是否使用异步模式
            strict_mode: 是否使用严格模式
            evaluation_trial_per_task: 每个任务的评估试验次数
            tool_matching_strategy: 工具匹配策略
                - "exact": 精确匹配，工具列表必须完全一致
                - "subset": 子集匹配，实际工具是期望工具的子集
                - "overlap": 重叠匹配，实际工具与期望工具有重叠
        """
        if threshold != 1.0:
            raise ValueError("Tool Pass@K指标的threshold必须为1.0，因为这是一个二元指标")
        
        self.k = k
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.evaluation_trial_per_task = evaluation_trial_per_task
        self.tool_matching_strategy = tool_matching_strategy
        
        # 评估结果
        self.verdicts: Optional[ToolPassAtKVerdicts] = None
        self.score: float = 0.0
        self.reason: Optional[str] = None
        self.success: bool = False
        
        # 统计信息
        self.num_trials: int = 0
        self.num_tool_correct: int = 0
        self.tool_accuracy: float = 0.0
    
    def measure(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估方法
        
        Args:
            test_case: 测试用例
            
        Returns:
            tool_pass@k分数
        """
        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(self.a_measure(test_case))
        else:
            return self._measure_sync(test_case)
    
    async def a_measure(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        异步评估方法
        
        Args:
            test_case: 测试用例
            
        Returns:
            tool_pass@k分数
        """
        return self._measure_sync(test_case)
    
    def _measure_sync(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估实现
        
        Args:
            test_case: 测试用例
            
        Returns:
            tool_pass@k分数
        """
        # 确保是MCPToolBenchTestCase
        if not isinstance(test_case, MCPToolBenchTestCase):
            raise ValueError("ToolPassAtKMetric只能用于MCPToolBenchTestCase")
        
        if test_case.execution_data is None:
            raise ValueError("测试用例必须包含execution_data")
        
        execution_data = test_case.execution_data
        
        # 获取期望的工具列表
        expected_tools = [label.tool_name for label in execution_data.function_call_label]
        
        # 获取试验结果
        k_tool_correct_results = execution_data.k_tool_correct_results
        if not k_tool_correct_results:
            # 如果没有试验结果，基于function_call_result创建
            if execution_data.function_call_result:
                selected_tools = [result.name for result in execution_data.function_call_result]
                tool_correctness = self._check_tool_correctness(selected_tools, expected_tools)
                k_tool_correct_results = [tool_correctness]
            else:
                k_tool_correct_results = [False] * self.evaluation_trial_per_task
        
        # 确保有足够的试验结果
        while len(k_tool_correct_results) < self.evaluation_trial_per_task:
            k_tool_correct_results.append(False)
        
        # 计算统计信息
        self.num_trials = len(k_tool_correct_results)
        self.num_tool_correct = sum(k_tool_correct_results)
        self.tool_accuracy = self.num_tool_correct / self.num_trials if self.num_trials > 0 else 0.0
        
        # 计算tool_pass@k
        tool_pass_at_k_score = self._estimate_pass_at_k(self.num_trials, self.num_tool_correct, self.k)
        
        # 创建判定结果
        verdicts = []
        for i, tool_correct in enumerate(k_tool_correct_results):
            # 获取实际选择的工具
            if i == 0 and execution_data.function_call_result:
                selected_tools = [result.name for result in execution_data.function_call_result]
            else:
                selected_tools = []  # 其他试验的工具信息可能不可用
            
            verdict = ToolPassAtKVerdict(
                trial_idx=i,
                tool_correctness=tool_correct,
                selected_tools=selected_tools,
                expected_tools=expected_tools
            )
            verdicts.append(verdict)
        
        self.verdicts = ToolPassAtKVerdicts(
            verdicts=verdicts,
            k=self.k,
            num_trials=self.num_trials,
            num_tool_correct=self.num_tool_correct,
            tool_pass_at_k=tool_pass_at_k_score
        )
        
        # 设置评估结果
        self.score = tool_pass_at_k_score
        self.success = tool_pass_at_k_score >= self.threshold
        
        if self.include_reason:
            self.reason = self._generate_reason()
        
        return self.score
    
    def _check_tool_correctness(self, selected_tools: List[str], expected_tools: List[str]) -> bool:
        """
        检查工具选择的正确性
        
        Args:
            selected_tools: 实际选择的工具列表
            expected_tools: 期望的工具列表
            
        Returns:
            工具选择是否正确
        """
        if self.tool_matching_strategy == "exact":
            # 精确匹配：工具列表必须完全一致（忽略顺序）
            return set(selected_tools) == set(expected_tools)
        
        elif self.tool_matching_strategy == "subset":
            # 子集匹配：实际工具是期望工具的子集
            return set(selected_tools).issubset(set(expected_tools))
        
        elif self.tool_matching_strategy == "overlap":
            # 重叠匹配：实际工具与期望工具有重叠
            return len(set(selected_tools) & set(expected_tools)) > 0
        
        else:
            raise ValueError(f"不支持的工具匹配策略: {self.tool_matching_strategy}")
    
    def _estimate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        估算tool_pass@k分数
        
        Args:
            n: 总试验次数
            c: 工具正确的试验次数
            k: k值
            
        Returns:
            tool_pass@k分数
        """
        if n - c < k:
            return 1.0
        
        try:
            # 计算 1 - C(n-c, k) / C(n, k)
            log_prob = 0.0
            for i in range(k):
                log_prob += math.log(n - c - i) - math.log(n - i)
            
            prob_all_fail = math.exp(log_prob)
            tool_pass_at_k = 1.0 - prob_all_fail
            
            return max(0.0, min(1.0, tool_pass_at_k))
        except (ValueError, OverflowError):
            # 如果计算出错，使用简单估算
            return min(1.0, c / n * k) if n > 0 else 0.0
    
    def _generate_reason(self) -> str:
        """
        生成评估原因
        
        Returns:
            详细的评估原因
        """
        if self.verdicts is None:
            return "评估未完成"
        
        reason_parts = [
            f"Tool Pass@{self.k}评估结果:",
            f"- 总试验次数: {self.num_trials}",
            f"- 工具正确次数: {self.num_tool_correct}",
            f"- 工具准确率: {self.tool_accuracy:.2%}",
            f"- Tool Pass@{self.k}分数: {self.score:.4f}",
            f"- 匹配策略: {self.tool_matching_strategy}",
            f"- 是否达到阈值: {'是' if self.success else '否'}"
        ]
        
        if self.num_trials > 0:
            reason_parts.append("\n试验详情:")
            for i, verdict in enumerate(self.verdicts.verdicts):
                status = "✓" if verdict.tool_correctness else "✗"
                selected_str = ", ".join(verdict.selected_tools) if verdict.selected_tools else "无"
                expected_str = ", ".join(verdict.expected_tools) if verdict.expected_tools else "无"
                reason_parts.append(
                    f"  试验{i+1}: {status} | 选择: [{selected_str}] | 期望: [{expected_str}]"
                )
        
        if self.strict_mode and not self.success:
            reason_parts.append(f"\n严格模式: Tool Pass@{self.k}分数未达到阈值{self.threshold}")
        
        return "\n".join(reason_parts)
    
    def is_successful(self) -> bool:
        """
        检查评估是否成功
        
        Returns:
            是否成功
        """
        return self.success
    
    @property
    def __name__(self):
        return f"Tool Pass@{self.k}"


def estimate_tool_pass_at_k(num_trials_array: List[int], num_tool_correct_array: List[int], k: int) -> List[float]:
    """
    批量估算tool_pass@k分数
    
    Args:
        num_trials_array: 每个任务的试验次数列表
        num_tool_correct_array: 每个任务的工具正确次数列表
        k: k值
        
    Returns:
        每个任务的tool_pass@k分数列表
    """
    tool_pass_at_k_scores = []
    
    for n, c in zip(num_trials_array, num_tool_correct_array):
        if n - c < k:
            tool_pass_at_k_scores.append(1.0)
        else:
            try:
                # 计算 1 - C(n-c, k) / C(n, k)
                log_prob = 0.0
                for i in range(k):
                    log_prob += math.log(n - c - i) - math.log(n - i)
                
                prob_all_fail = math.exp(log_prob)
                tool_pass_at_k = 1.0 - prob_all_fail
                tool_pass_at_k_scores.append(max(0.0, min(1.0, tool_pass_at_k)))
            except (ValueError, OverflowError):
                # 如果计算出错，使用简单估算
                tool_pass_at_k_scores.append(min(1.0, c / n * k) if n > 0 else 0.0)
    
    return tool_pass_at_k_scores