from typing import Optional, List, Union, Dict, Any
import math
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.test_case import LLMTestCase
from veagentbench.evals.deepeval.utils import get_or_create_event_loop, prettify_list

from ..test_case import MCPToolBenchTestCase
from ..schema import (
    ParameterPassAtKVerdict, 
    ParameterPassAtKVerdicts,
    ToolCallResult,
    MCPToolBenchExecutionData
)


class ParameterPassAtKMetric(BaseMetric):
    """
    Parameter Pass@K指标 - 评估工具参数的正确性
    
    基于MCPToolBenchPP的parameter_pass@k评估逻辑，计算在k次试验中至少有一次
    正确设置工具参数的概率。
    """
    
    def __init__(
        self,
        k: int = 1,
        threshold: float = 1.0,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        evaluation_trial_per_task: int = 1,
        parameter_matching_strategy: str = "semantic",  # "exact", "semantic", "key_match"
        required_parameters_only: bool = False
    ):
        """
        初始化Parameter Pass@K指标
        
        Args:
            k: k值，计算parameter_pass@k时使用
            threshold: 阈值，必须为1.0（parameter_pass@k是二元指标）
            include_reason: 是否包含详细原因
            async_mode: 是否使用异步模式
            strict_mode: 是否使用严格模式
            evaluation_trial_per_task: 每个任务的评估试验次数
            parameter_matching_strategy: 参数匹配策略
                - "exact": 精确匹配，参数必须完全一致
                - "semantic": 语义匹配，基于语义相似性判断
                - "key_match": 键匹配，只检查必需参数的键是否存在
            required_parameters_only: 是否只检查必需参数
        """
        if threshold != 1.0:
            raise ValueError("Parameter Pass@K指标的threshold必须为1.0，因为这是一个二元指标")
        
        self.k = k
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.evaluation_trial_per_task = evaluation_trial_per_task
        self.parameter_matching_strategy = parameter_matching_strategy
        self.required_parameters_only = required_parameters_only
        
        # 评估结果
        self.verdicts: Optional[ParameterPassAtKVerdicts] = None
        self.score: float = 0.0
        self.reason: Optional[str] = None
        self.success: bool = False
        
        # 统计信息
        self.num_trials: int = 0
        self.num_parameter_correct: int = 0
        self.parameter_accuracy: float = 0.0
    
    def measure(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估方法
        
        Args:
            test_case: 测试用例
            
        Returns:
            parameter_pass@k分数
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
            parameter_pass@k分数
        """
        return self._measure_sync(test_case)
    
    def _measure_sync(self, test_case: Union[LLMTestCase, MCPToolBenchTestCase]) -> float:
        """
        同步评估实现
        
        Args:
            test_case: 测试用例
            
        Returns:
            parameter_pass@k分数
        """
        # 确保是MCPToolBenchTestCase
        if not isinstance(test_case, MCPToolBenchTestCase):
            raise ValueError("ParameterPassAtKMetric只能用于MCPToolBenchTestCase")
        
        if test_case.execution_data is None:
            raise ValueError("测试用例必须包含execution_data")
        
        execution_data = test_case.execution_data
        
        # 获取试验结果
        k_parameter_correct_results = execution_data.k_parameter_correct_results
        if not k_parameter_correct_results:
            # 如果没有试验结果，基于function_call_result创建
            if execution_data.function_call_result and execution_data.function_call_label:
                parameter_correctness = self._check_parameter_correctness(
                    execution_data.function_call_result,
                    execution_data.function_call_label
                )
                k_parameter_correct_results = [parameter_correctness]
            else:
                k_parameter_correct_results = [False] * self.evaluation_trial_per_task
        
        # 确保有足够的试验结果
        while len(k_parameter_correct_results) < self.evaluation_trial_per_task:
            k_parameter_correct_results.append(False)
        
        # 计算统计信息
        self.num_trials = len(k_parameter_correct_results)
        self.num_parameter_correct = sum(k_parameter_correct_results)
        self.parameter_accuracy = self.num_parameter_correct / self.num_trials if self.num_trials > 0 else 0.0
        
        # 计算parameter_pass@k
        parameter_pass_at_k_score = self._estimate_pass_at_k(self.num_trials, self.num_parameter_correct, self.k)
        
        # 创建判定结果
        verdicts = []
        for i, param_correct in enumerate(k_parameter_correct_results):
            # 获取实际参数信息
            if i == 0 and execution_data.function_call_result:
                # 只显示第一个工具调用的参数信息
                first_result = execution_data.function_call_result[0]
                tool_name = first_result.name
                actual_parameters = first_result.input
                
                # 查找对应的期望参数
                expected_parameters = None
                for label in execution_data.function_call_label:
                    if label.tool_name == tool_name:
                        expected_parameters = getattr(label, 'parameters', None)
                        break
            else:
                tool_name = ""
                actual_parameters = {}
                expected_parameters = None
            
            verdict = ParameterPassAtKVerdict(
                trial_idx=i,
                parameter_correctness=param_correct,
                tool_name=tool_name,
                actual_parameters=actual_parameters,
                expected_parameters=expected_parameters
            )
            verdicts.append(verdict)
        
        self.verdicts = ParameterPassAtKVerdicts(
            verdicts=verdicts,
            k=self.k,
            num_trials=self.num_trials,
            num_parameter_correct=self.num_parameter_correct,
            parameter_pass_at_k=parameter_pass_at_k_score
        )
        
        # 设置评估结果
        self.score = parameter_pass_at_k_score
        self.success = parameter_pass_at_k_score >= self.threshold
        
        if self.include_reason:
            self.reason = self._generate_reason()
        
        return self.score
    
    def _check_parameter_correctness(
        self, 
        function_call_results: List[ToolCallResult], 
        function_call_labels: List[Any]
    ) -> bool:
        """
        检查参数的正确性
        
        Args:
            function_call_results: 实际工具调用结果
            function_call_labels: 期望工具调用标签
            
        Returns:
            参数是否正确
        """
        if not function_call_results or not function_call_labels:
            return False
        
        # 简化版本：检查第一个工具调用的参数
        # 在实际实现中，这里应该调用MCPToolBenchPP的check_ast函数
        # 或者实现更复杂的参数匹配逻辑
        
        result = function_call_results[0]
        actual_params = result.input
        
        # 查找对应的期望参数
        expected_params = None
        for label in function_call_labels:
            if hasattr(label, 'tool_name') and label.tool_name == result.name:
                expected_params = getattr(label, 'parameters', None)
                break
        
        if expected_params is None:
            # 如果没有期望参数，认为参数正确（可能是开放式任务）
            return True
        
        return self._match_parameters(actual_params, expected_params)
    
    def _match_parameters(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """
        匹配参数
        
        Args:
            actual: 实际参数
            expected: 期望参数
            
        Returns:
            参数是否匹配
        """
        if self.parameter_matching_strategy == "exact":
            # 精确匹配
            return actual == expected
        
        elif self.parameter_matching_strategy == "key_match":
            # 键匹配：检查必需参数的键是否存在
            if self.required_parameters_only:
                # 假设所有期望参数都是必需的
                required_keys = set(expected.keys())
                actual_keys = set(actual.keys())
                return required_keys.issubset(actual_keys)
            else:
                return set(actual.keys()) == set(expected.keys())
        
        elif self.parameter_matching_strategy == "semantic":
            # 语义匹配：基于值的语义相似性
            # 这里实现简化版本，实际应该使用更复杂的语义匹配
            if set(actual.keys()) != set(expected.keys()):
                return False
            
            for key in expected.keys():
                if key not in actual:
                    return False
                
                actual_val = actual[key]
                expected_val = expected[key]
                
                # 简单的语义匹配：类型一致且值相近
                if type(actual_val) != type(expected_val):
                    return False
                
                if isinstance(actual_val, str):
                    # 字符串：忽略大小写和空格
                    if actual_val.strip().lower() != expected_val.strip().lower():
                        return False
                elif isinstance(actual_val, (int, float)):
                    # 数值：允许小幅差异
                    if abs(actual_val - expected_val) > 0.01:
                        return False
                else:
                    # 其他类型：精确匹配
                    if actual_val != expected_val:
                        return False
            
            return True
        
        else:
            raise ValueError(f"不支持的参数匹配策略: {self.parameter_matching_strategy}")
    
    def _estimate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        估算parameter_pass@k分数
        
        Args:
            n: 总试验次数
            c: 参数正确的试验次数
            k: k值
            
        Returns:
            parameter_pass@k分数
        """
        if n - c < k:
            return 1.0
        
        try:
            # 计算 1 - C(n-c, k) / C(n, k)
            log_prob = 0.0
            for i in range(k):
                log_prob += math.log(n - c - i) - math.log(n - i)
            
            prob_all_fail = math.exp(log_prob)
            parameter_pass_at_k = 1.0 - prob_all_fail
            
            return max(0.0, min(1.0, parameter_pass_at_k))
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
            f"Parameter Pass@{self.k}评估结果:",
            f"- 总试验次数: {self.num_trials}",
            f"- 参数正确次数: {self.num_parameter_correct}",
            f"- 参数准确率: {self.parameter_accuracy:.2%}",
            f"- Parameter Pass@{self.k}分数: {self.score:.4f}",
            f"- 匹配策略: {self.parameter_matching_strategy}",
            f"- 只检查必需参数: {'是' if self.required_parameters_only else '否'}",
            f"- 是否达到阈值: {'是' if self.success else '否'}"
        ]
        
        if self.num_trials > 0:
            reason_parts.append("\n试验详情:")
            for i, verdict in enumerate(self.verdicts.verdicts):
                status = "✓" if verdict.parameter_correctness else "✗"
                tool_name = verdict.tool_name or "未知"
                param_count = len(verdict.actual_parameters)
                reason_parts.append(
                    f"  试验{i+1}: {status} | 工具: {tool_name} | 参数数量: {param_count}"
                )
        
        if self.strict_mode and not self.success:
            reason_parts.append(f"\n严格模式: Parameter Pass@{self.k}分数未达到阈值{self.threshold}")
        
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
        return f"Parameter Pass@{self.k}"


def estimate_parameter_pass_at_k(num_trials_array: List[int], num_parameter_correct_array: List[int], k: int) -> List[float]:
    """
    批量估算parameter_pass@k分数
    
    Args:
        num_trials_array: 每个任务的试验次数列表
        num_parameter_correct_array: 每个任务的参数正确次数列表
        k: k值
        
    Returns:
        每个任务的parameter_pass@k分数列表
    """
    parameter_pass_at_k_scores = []
    
    for n, c in zip(num_trials_array, num_parameter_correct_array):
        if n - c < k:
            parameter_pass_at_k_scores.append(1.0)
        else:
            try:
                # 计算 1 - C(n-c, k) / C(n, k)
                log_prob = 0.0
                for i in range(k):
                    log_prob += math.log(n - c - i) - math.log(n - i)
                
                prob_all_fail = math.exp(log_prob)
                parameter_pass_at_k = 1.0 - prob_all_fail
                parameter_pass_at_k_scores.append(max(0.0, min(1.0, parameter_pass_at_k)))
            except (ValueError, OverflowError):
                # 如果计算出错，使用简单估算
                parameter_pass_at_k_scores.append(min(1.0, c / n * k) if n > 0 else 0.0)
    
    return parameter_pass_at_k_scores