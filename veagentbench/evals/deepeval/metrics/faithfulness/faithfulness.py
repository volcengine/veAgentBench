from typing import List, Optional, Union, Type
import asyncio
import functools
import time
from typing import Callable, Any

from veagentbench.evals.deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from veagentbench.evals.deepeval.metrics import BaseMetric
from veagentbench.evals.deepeval.utils import get_or_create_event_loop, prettify_list
from veagentbench.evals.deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from veagentbench.evals.deepeval.models import DeepEvalBaseLLM
from veagentbench.evals.deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from veagentbench.evals.deepeval.metrics.indicator import metric_progress_indicator
from veagentbench.evals.deepeval.metrics.faithfulness.schema import (
    FaithfulnessVerdict,
    Verdicts,
    FaithfulnessScoreReason,
    Truths,
    Claims,
)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    throw_exceptions: bool = True,
    default_return: Optional[Any] = None
) -> Callable:
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的增长因子
        exceptions: 需要重试的异常类型元组
        throw_exceptions: 是否抛出异常，默认为True
        default_return: 默认返回值，当抛出异常且throw_exceptions为False时返回
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                        print(f"等待 {current_delay} 秒后重试...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                        print("已达到最大重试次数，不再重试")
            if throw_exceptions:
                raise last_exception
            else:
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                        print(f"等待 {current_delay} 秒后重试...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                        print("已达到最大重试次数，不再重试")
            
            raise last_exception
        
        # 如果是异步函数，返回异步包装器，否则返回同步包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class FaithfulnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        truths_extraction_limit: Optional[int] = None,
        penalize_ambiguous_claims: bool = False,
        evaluation_template: Type[FaithfulnessTemplate] = FaithfulnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.penalize_ambiguous_claims = penalize_ambiguous_claims

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                self.truths = self._generate_truths(test_case.retrieval_context)
                self.claims = self._generate_claims(test_case.actual_output)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                        f"Claims:\n{prettify_list(self.claims)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    @retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,))
    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.truths, self.claims = await asyncio.gather(
                self._a_generate_truths(test_case.retrieval_context),
                self._a_generate_claims(test_case.actual_output),
            )
            self.verdicts = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=FaithfulnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: FaithfulnessScoreReason = await self.model.a_generate(
                    prompt, schema=FaithfulnessScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=FaithfulnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: FaithfulnessScoreReason = self.model.generate(
                    prompt, schema=FaithfulnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            retrieval_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = await self.model.a_generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    def _generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            retrieval_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = self.model.generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            actual_output=actual_output
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            actual_output=actual_output
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

            if (
                self.penalize_ambiguous_claims
                and verdict.verdict.strip().lower() == "idk"
            ):
                faithfulness_count -= 1

        score = faithfulness_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Faithfulness"
