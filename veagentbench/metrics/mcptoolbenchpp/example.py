"""
MCPToolBenchPP指标使用示例

展示如何使用PassAtKMetric、ToolPassAtKMetric和ParameterPassAtKMetric
来评估AI代理在工具调用任务中的表现。
"""

import asyncio
from typing import List, Dict, Any

from mcptoolbenchpp import (
    PassAtKMetric,
    ToolPassAtKMetric,
    ParameterPassAtKMetric,
    MCPToolBenchTestCase,
    ToolCallResult,
    MCPToolBenchExecutionData
)


def create_sample_test_case() -> MCPToolBenchTestCase:
    """
    创建一个示例测试用例
    
    Returns:
        MCPToolBenchTestCase实例
    """
    # 模拟工具调用结果
    function_call_result = [
        ToolCallResult(
            name="get_weather",
            input={"location": "北京", "unit": "celsius"},
            output={"temperature": 25, "condition": "晴天"}
        ),
        ToolCallResult(
            name="send_email",
            input={"to": "user@example.com", "subject": "天气报告", "body": "今天北京天气晴朗，温度25度"},
            output={"status": "sent", "message_id": "12345"}
        )
    ]
    
    # 模拟期望的工具调用标签
    function_call_label = [
        type('ToolLabel', (), {
            'tool_name': 'get_weather',
            'parameters': {"location": "北京", "unit": "celsius"}
        })(),
        type('ToolLabel', (), {
            'tool_name': 'send_email', 
            'parameters': {"to": "user@example.com", "subject": "天气报告"}
        })()
    ]
    
    # 创建执行数据
    execution_data = MCPToolBenchExecutionData(
        function_call_result=function_call_result,
        function_call_label=function_call_label,
        k_correct_results=[True, False, True],  # 3次试验，2次成功
        k_tool_correct_results=[True, True, False],  # 工具调用正确性
        k_parameter_correct_results=[True, False, True],  # 参数正确性
        final_result="成功获取北京天气并发送邮件通知用户。",
        accumulated_information=["获取天气信息", "准备邮件内容", "发送邮件"]
    )
    
    # 创建测试用例
    test_case = MCPToolBenchTestCase(
        input="请获取北京的天气信息，并通过邮件发送给用户",
        actual_output="我已经获取了北京的天气信息（晴天，25度），并通过邮件发送给了用户。",
        expected_output="应该调用get_weather获取天气，然后调用send_email发送邮件",
        execution_data=execution_data
    )
    
    return test_case


def example_pass_at_k():
    """
    Pass@K指标使用示例
    """
    print("=== Pass@K指标示例 ===")
    
    # 创建测试用例
    test_case = create_sample_test_case()
    
    # 创建Pass@K指标（k=1）
    pass_at_1_metric = PassAtKMetric(
        k=1,
        threshold=0.5,
        include_reason=True
    )
    
    # 评估
    score = pass_at_1_metric.measure(test_case)
    
    print(f"Pass@1分数: {score:.4f}")
    print(f"是否成功: {pass_at_1_metric.is_successful()}")
    print(f"评估原因:\n{pass_at_1_metric.reason}")
    print()


def example_tool_pass_at_k():
    """
    Tool Pass@K指标使用示例
    """
    print("=== Tool Pass@K指标示例 ===")
    
    # 创建测试用例
    test_case = create_sample_test_case()
    
    # 创建Tool Pass@K指标（k=2）
    tool_pass_at_2_metric = ToolPassAtKMetric(
        k=2,
        threshold=0.7,
        include_reason=True,
        tool_matching_strategy="name_and_sequence"
    )
    
    # 评估
    score = tool_pass_at_2_metric.measure(test_case)
    
    print(f"Tool Pass@2分数: {score:.4f}")
    print(f"是否成功: {tool_pass_at_2_metric.is_successful()}")
    print(f"评估原因:\n{tool_pass_at_2_metric.reason}")
    print()


def example_parameter_pass_at_k():
    """
    Parameter Pass@K指标使用示例
    """
    print("=== Parameter Pass@K指标示例 ===")
    
    # 创建测试用例
    test_case = create_sample_test_case()
    
    # 创建Parameter Pass@K指标（k=1）
    param_pass_at_1_metric = ParameterPassAtKMetric(
        k=1,
        threshold=1.0,
        include_reason=True,
        parameter_matching_strategy="semantic"
    )
    
    # 评估
    score = param_pass_at_1_metric.measure(test_case)
    
    print(f"Parameter Pass@1分数: {score:.4f}")
    print(f"是否成功: {param_pass_at_1_metric.is_successful()}")
    print(f"评估原因:\n{param_pass_at_1_metric.reason}")
    print()


async def example_async_evaluation():
    """
    异步评估示例
    """
    print("=== 异步评估示例 ===")
    
    # 创建测试用例
    test_case = create_sample_test_case()
    
    # 创建多个指标
    metrics = [
        PassAtKMetric(k=1, async_mode=True),
        ToolPassAtKMetric(k=1, async_mode=True),
        ParameterPassAtKMetric(k=1, async_mode=True)
    ]
    
    # 并发评估
    tasks = [metric.a_measure(test_case) for metric in metrics]
    scores = await asyncio.gather(*tasks)
    
    metric_names = ["Pass@1", "Tool Pass@1", "Parameter Pass@1"]
    
    print("并发评估结果:")
    for name, score in zip(metric_names, scores):
        print(f"- {name}: {score:.4f}")
    print()


def example_batch_evaluation():
    """
    批量评估示例
    """
    print("=== 批量评估示例 ===")
    
    # 创建多个测试用例
    test_cases = []
    for i in range(3):
        test_case = create_sample_test_case()
        # 修改一些数据以模拟不同的结果
        if i == 1:
            # 第二个测试用例：工具调用失败
            test_case.execution_data.k_correct_results = [False, False, False]
            test_case.execution_data.k_tool_correct_results = [False, False, False]
        elif i == 2:
            # 第三个测试用例：参数错误
            test_case.execution_data.k_parameter_correct_results = [False, False, True]
        
        test_cases.append(test_case)
    
    # 创建指标
    pass_at_k_metric = PassAtKMetric(k=1, include_reason=False)
    
    # 批量评估
    results = []
    for i, test_case in enumerate(test_cases):
        score = pass_at_k_metric.measure(test_case)
        results.append((i, score, pass_at_k_metric.is_successful()))
    
    print("批量评估结果:")
    for case_id, score, success in results:
        status = "✓" if success else "✗"
        print(f"测试用例{case_id + 1}: {status} Pass@1={score:.4f}")
    
    # 计算平均分数
    avg_score = sum(score for _, score, _ in results) / len(results)
    success_rate = sum(1 for _, _, success in results if success) / len(results)
    
    print(f"\n总体统计:")
    print(f"- 平均Pass@1分数: {avg_score:.4f}")
    print(f"- 成功率: {success_rate:.2%}")
    print()


def example_different_k_values():
    """
    不同k值的比较示例
    """
    print("=== 不同k值比较示例 ===")
    
    # 创建测试用例
    test_case = create_sample_test_case()
    
    # 测试不同的k值
    k_values = [1, 2, 3]
    
    print("Pass@K指标在不同k值下的表现:")
    for k in k_values:
        metric = PassAtKMetric(k=k, include_reason=False)
        score = metric.measure(test_case)
        print(f"- Pass@{k}: {score:.4f}")
    
    print("\nTool Pass@K指标在不同k值下的表现:")
    for k in k_values:
        metric = ToolPassAtKMetric(k=k, include_reason=False)
        score = metric.measure(test_case)
        print(f"- Tool Pass@{k}: {score:.4f}")
    
    print("\nParameter Pass@K指标在不同k值下的表现:")
    for k in k_values:
        metric = ParameterPassAtKMetric(k=k, include_reason=False)
        score = metric.measure(test_case)
        print(f"- Parameter Pass@{k}: {score:.4f}")
    print()


def main():
    """
    主函数：运行所有示例
    """
    print("MCPToolBenchPP指标使用示例\n")
    
    # 基础示例
    example_pass_at_k()
    example_tool_pass_at_k()
    example_parameter_pass_at_k()
    
    # 异步示例
    asyncio.run(example_async_evaluation())
    
    # 批量评估示例
    example_batch_evaluation()
    
    # 不同k值比较
    example_different_k_values()
    
    print("所有示例运行完成！")


if __name__ == "__main__":
    main()