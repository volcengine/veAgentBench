from typing import Optional, List, Union, Dict, Any
import math
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCase
from veagentbench.evals.deepeval.utils import get_or_create_event_loop, prettify_list

from ..test_case import MCPToolBenchTestCase
from ..schema import (
    PassAtKVerdict, 
    PassAtKVerdicts,
    ToolCallResult,
    MCPToolBenchExecutionData
)


class PassAtKMetric(BaseMetric):
    """
    Pass@K指标 - 评估整体任务完成的正确性
    
    基于MCPToolBenchPP的pass@k评估逻辑，计算在k次试验中至少有一次
    完全正确完成任务的概率。
    """
    
    def __init__(
        self,
        k: int = 1,
        threshold: float = 1.0,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        evaluation_trial_per_task: int = 1
    ):
        """
        初始化Pass@K指标
        
        Args:
            k: k值，计算pass@k时使用
            threshold: 阈值，必须为1.0（pass@k是二元指标）
            include_reason: 是否包含详细原因
            async_mode: 是否使用异步模式
            strict_mode: 是否使用严格模式
            evaluation_trial_per_task: 每个任务的评估试验次数
        """
        if threshold != 1.0:
            raise ValueError("Pass@K指标的threshold必须为1.0，因为这是一个二元指标")
        
        self.k = k
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.evaluation_trial_per_task = evaluation_trial_per_task
        
        # 评估结果
        self.verdicts: Optional[PassAtKVerdicts] = None
        self.score: float = 0.0
        self.reason: Optional[str] = None
        self.success: bool = False
        
        # 统计信息
        self.num_trials: int = 0
        self.num_passed: int = 0
        self.pass_rate: float = 0.0
    
    def measure(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估方法
        
        Args:
            test_case: 测试用例
            
        Returns:
            pass@k分数
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
            pass@k分数
        """
        return self._measure_sync(test_case)
    
    def _measure_sync(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估实现
        
        Args:
            test_case: 测试用例
            
        Returns:
            pass@k分数
        """
        # 确保是MCPToolBenchTestCase
        if not isinstance(test_case, MCPToolBenchTestCase):
            raise ValueError("PassAtKMetric只能用于MCPToolBenchTestCase")
        
        if test_case.execution_data is None:
            raise ValueError("测试用例必须包含execution_data")
        
        execution_data = test_case.execution_data
        
        # 获取试验结果
        k_results = execution_data.k_results
        if not k_results:
            # 如果没有试验结果，创建默认结果
            k_results = [False] * self.evaluation_trial_per_task
        
        # 确保有足够的试验结果
        while len(k_results) < self.evaluation_trial_per_task:
            k_results.append(False)
        
        # 计算统计信息
        self.num_trials = len(k_results)
        self.num_passed = sum(k_results)
        self.pass_rate = self.num_passed / self.num_trials if self.num_trials > 0 else 0.0
        
        # 计算pass@k
        pass_at_k_score = self._estimate_pass_at_k(self.num_trials, self.num_passed, self.k)
        
        # 创建判定结果
        verdicts = []
        for i, passed in enumerate(k_results):
            verdict = PassAtKVerdict(
                trial_idx=i,
                if_pass=passed,
                tool_correctness=execution_data.k_tool_correct_results[i] if i < len(execution_data.k_tool_correct_results) else False,
                parameter_correctness=execution_data.k_parameter_correct_results[i] if i < len(execution_data.k_parameter_correct_results) else False,
                function_call_result=execution_data.function_call_result if i == 0 else []
            )
            verdicts.append(verdict)
        
        self.verdicts = PassAtKVerdicts(
            verdicts=verdicts,
            k=self.k,
            num_trials=self.num_trials,
            num_passed=self.num_passed,
            pass_at_k=pass_at_k_score
        )
        
        # 设置评估结果
        self.score = pass_at_k_score
        self.success = pass_at_k_score >= self.threshold
        
        if self.include_reason:
            self.reason = self._generate_reason()
        
        return self.score
    
    def _estimate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        估算pass@k分数
        
        基于MCPToolBenchPP的estimate_pass_at_k实现
        
        Args:
            n: 总试验次数
            c: 通过的试验次数
            k: k值
            
        Returns:
            pass@k分数
        """
        if n - c < k:
            return 1.0
        
        # 计算组合数 C(n-c, k) / C(n, k)
        # pass@k = 1 - C(n-c, k) / C(n, k)
        try:
            # 使用对数避免大数计算溢出
            log_prob = 0.0
            for i in range(k):
                log_prob += math.log(n - c - i) - math.log(n - i)
            
            prob_all_fail = math.exp(log_prob)
            pass_at_k = 1.0 - prob_all_fail
            
            return max(0.0, min(1.0, pass_at_k))
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
            f"Pass@{self.k}评估结果:",
            f"- 总试验次数: {self.num_trials}",
            f"- 通过试验次数: {self.num_passed}",
            f"- 通过率: {self.pass_rate:.2%}",
            f"- Pass@{self.k}分数: {self.score:.4f}",
            f"- 是否达到阈值: {'是' if self.success else '否'}"
        ]
        
        if self.num_trials > 0:
            reason_parts.append("\n试验详情:")
            for i, verdict in enumerate(self.verdicts.verdicts):
                status = "✓" if verdict.if_pass else "✗"
                tool_status = "✓" if verdict.tool_correctness else "✗"
                param_status = "✓" if verdict.parameter_correctness else "✗"
                reason_parts.append(
                    f"  试验{i+1}: {status} 整体 | {tool_status} 工具 | {param_status} 参数"
                )
        
        if self.strict_mode and not self.success:
            reason_parts.append(f"\n严格模式: Pass@{self.k}分数未达到阈值{self.threshold}")
        
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
        return f"Pass@{self.k}"


def estimate_pass_at_k(num_trials_array: List[int], num_pass_array: List[int], k: int) -> List[float]:
    """
    批量估算pass@k分数
    
    Args:
        num_trials_array: 每个任务的试验次数列表
        num_pass_array: 每个任务的通过次数列表
        k: k值
        
    Returns:
        每个任务的pass@k分数列表
    """
    pass_at_k_scores = []
    
    for n, c in zip(num_trials_array, num_pass_array):
        if n - c < k:
            pass_at_k_scores.append(1.0)
        else:
            try:
                # 计算 1 - C(n-c, k) / C(n, k)
                log_prob = 0.0
                for i in range(k):
                    log_prob += math.log(n - c - i) - math.log(n - i)
                
                prob_all_fail = math.exp(log_prob)
                pass_at_k = 1.0 - prob_all_fail
                pass_at_k_scores.append(max(0.0, min(1.0, pass_at_k)))
            except (ValueError, OverflowError):
                # 如果计算出错，使用简单估算
                pass_at_k_scores.append(min(1.0, c / n * k) if n > 0 else 0.0)
    
    return pass_at_k_scores