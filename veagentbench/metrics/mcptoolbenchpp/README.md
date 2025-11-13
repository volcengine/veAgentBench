# MCPToolBenchPP 评估指标

基于deepeval架构实现的MCPToolBenchPP评估指标模块，用于评估AI代理在工具调用任务中的表现。

## 概述

MCPToolBenchPP提供了三个核心评估指标：

1. **Pass@K**: 评估整体任务完成率
2. **Tool Pass@K**: 评估工具调用的正确性  
3. **Parameter Pass@K**: 评估工具参数的正确性

这些指标遵循MCPToolBenchPP的评估逻辑，采用pass@k的统计方法来衡量AI代理的性能。

## 安装

确保已安装deepeval：

```bash
pip install deepeval
```

## 快速开始

### 基本使用

```python
from mcptoolbenchpp import (
    PassAtKMetric,
    ToolPassAtKMetric, 
    ParameterPassAtKMetric,
    MCPToolBenchTestCase,
    ToolCallResult,
    MCPToolBenchExecutionData
)

# 创建工具调用结果
function_call_result = [
    ToolCallResult(
        name="get_weather",
        input={"location": "北京", "unit": "celsius"},
        output={"temperature": 25, "condition": "晴天"}
    )
]

# 创建执行数据
execution_data = MCPToolBenchExecutionData(
    function_call_result=function_call_result,
    k_correct_results=[True, False, True],  # 3次试验，2次成功
    final_result="成功获取北京天气信息"
)

# 创建测试用例
test_case = MCPToolBenchTestCase(
    input="请获取北京的天气信息",
    actual_output="北京今天晴天，温度25度",
    expected_output="应该调用get_weather获取天气信息",
    execution_data=execution_data
)

# 评估Pass@1
pass_at_1 = PassAtKMetric(k=1)
score = pass_at_1.measure(test_case)
print(f"Pass@1分数: {score:.4f}")
```

### 异步评估

```python
import asyncio

async def async_evaluation():
    # 创建异步指标
    metric = PassAtKMetric(k=1, async_mode=True)
    
    # 异步评估
    score = await metric.a_measure(test_case)
    print(f"异步评估结果: {score:.4f}")

# 运行异步评估
asyncio.run(async_evaluation())
```

## 指标详解

### 1. PassAtKMetric

评估整体任务完成率，计算在k次试验中至少有一次成功完成任务的概率。

**参数：**
- `k`: k值，默认为1
- `threshold`: 成功阈值，默认为0.5
- `include_reason`: 是否包含详细原因，默认为True
- `async_mode`: 是否使用异步模式，默认为True
- `strict_mode`: 是否使用严格模式，默认为False

**示例：**
```python
# 创建Pass@3指标
metric = PassAtKMetric(
    k=3,
    threshold=0.7,
    include_reason=True
)

score = metric.measure(test_case)
print(f"Pass@3: {score:.4f}")
print(f"评估原因: {metric.reason}")
```

### 2. ToolPassAtKMetric

评估工具调用的正确性，检查调用的工具名称、顺序和数量是否正确。

**参数：**
- `k`: k值，默认为1
- `threshold`: 成功阈值，默认为1.0
- `tool_matching_strategy`: 工具匹配策略
  - `"name_only"`: 只匹配工具名称
  - `"name_and_sequence"`: 匹配名称和调用顺序
  - `"exact_match"`: 精确匹配所有属性

**示例：**
```python
# 创建Tool Pass@2指标，使用名称和顺序匹配
metric = ToolPassAtKMetric(
    k=2,
    threshold=0.8,
    tool_matching_strategy="name_and_sequence"
)

score = metric.measure(test_case)
print(f"Tool Pass@2: {score:.4f}")
```

### 3. ParameterPassAtKMetric

评估工具参数的正确性，检查传递给工具的参数是否符合预期。

**参数：**
- `k`: k值，默认为1
- `threshold`: 成功阈值，必须为1.0（二元指标）
- `parameter_matching_strategy`: 参数匹配策略
  - `"exact"`: 精确匹配
  - `"semantic"`: 语义匹配
  - `"key_match"`: 键匹配
- `required_parameters_only`: 是否只检查必需参数

**示例：**
```python
# 创建Parameter Pass@1指标，使用语义匹配
metric = ParameterPassAtKMetric(
    k=1,
    parameter_matching_strategy="semantic",
    required_parameters_only=True
)

score = metric.measure(test_case)
print(f"Parameter Pass@1: {score:.4f}")
```

## 数据结构

### MCPToolBenchTestCase

继承自deepeval的LLMTestCase，专门用于工具调用任务的测试用例。

```python
test_case = MCPToolBenchTestCase(
    input="用户输入",
    actual_output="实际输出", 
    expected_output="期望输出",
    execution_data=execution_data  # MCPToolBenchExecutionData实例
)
```

### MCPToolBenchExecutionData

包含工具调用执行的详细信息：

```python
execution_data = MCPToolBenchExecutionData(
    function_call_result=[...],      # 实际工具调用结果
    function_call_label=[...],       # 期望工具调用标签
    k_correct_results=[...],         # k次试验的整体正确性
    k_tool_correct_results=[...],    # k次试验的工具正确性
    k_parameter_correct_results=[...], # k次试验的参数正确性
    final_result="最终结果",
    accumulated_information=[...]     # 累积信息
)
```

### ToolCallResult

表示单次工具调用的结果：

```python
tool_result = ToolCallResult(
    name="工具名称",
    input={"参数": "值"},
    output={"结果": "值"}
)
```

## 批量评估

```python
# 批量评估多个测试用例
test_cases = [test_case1, test_case2, test_case3]
metric = PassAtKMetric(k=1)

results = []
for test_case in test_cases:
    score = metric.measure(test_case)
    results.append(score)

# 计算平均分数
avg_score = sum(results) / len(results)
print(f"平均Pass@1分数: {avg_score:.4f}")
```

## 并发评估

```python
import asyncio

async def concurrent_evaluation(test_cases):
    # 创建异步指标
    metrics = [
        PassAtKMetric(k=1, async_mode=True),
        ToolPassAtKMetric(k=1, async_mode=True),
        ParameterPassAtKMetric(k=1, async_mode=True)
    ]
    
    # 并发评估所有指标
    tasks = []
    for metric in metrics:
        for test_case in test_cases:
            tasks.append(metric.a_measure(test_case))
    
    results = await asyncio.gather(*tasks)
    return results

# 运行并发评估
results = asyncio.run(concurrent_evaluation(test_cases))
```

## 配置选项

### 匹配策略

**工具匹配策略 (tool_matching_strategy):**
- `name_only`: 只检查工具名称是否匹配
- `name_and_sequence`: 检查工具名称和调用顺序
- `exact_match`: 精确匹配所有属性

**参数匹配策略 (parameter_matching_strategy):**
- `exact`: 参数必须完全一致
- `semantic`: 基于语义相似性判断
- `key_match`: 只检查必需参数的键是否存在

### 严格模式

启用严格模式后，只有达到阈值的评估才被认为是成功的：

```python
metric = PassAtKMetric(
    k=1,
    threshold=0.8,
    strict_mode=True  # 启用严格模式
)
```

## 最佳实践

1. **选择合适的k值**: 根据任务复杂度选择k值，简单任务使用k=1，复杂任务可以使用k=3或k=5

2. **设置合理的阈值**: Pass@K通常使用0.5-0.8的阈值，Tool Pass@K和Parameter Pass@K建议使用1.0

3. **使用异步模式**: 对于大批量评估，建议使用异步模式提高性能

4. **选择合适的匹配策略**: 根据任务要求选择匹配策略，严格任务使用exact匹配，灵活任务使用semantic匹配

5. **包含详细原因**: 在开发和调试阶段，建议设置`include_reason=True`获取详细的评估信息

## 示例代码

完整的使用示例请参考 `example.py` 文件，其中包含了：

- 基础指标使用
- 异步评估
- 批量评估  
- 不同k值比较
- 各种配置选项的使用

运行示例：

```bash
python mcptoolbenchpp/example.py
```

## 注意事项

1. **数据格式**: 确保`MCPToolBenchExecutionData`中的数据格式正确
2. **k值限制**: k值不能超过试验次数
3. **阈值设置**: Parameter Pass@K的阈值必须为1.0
4. **异步使用**: 在异步环境中使用时，确保正确处理事件循环

## 扩展

如需自定义评估逻辑，可以继承相应的指标类并重写评估方法：

```python
class CustomPassAtKMetric(PassAtKMetric):
    def _check_task_correctness(self, execution_data, expected_output):
        # 自定义任务正确性检查逻辑
        return custom_logic(execution_data, expected_output)