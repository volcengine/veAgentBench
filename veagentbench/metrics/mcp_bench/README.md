# MCP工具评估指标 (MCPToolMetric) - 统一的AgentTestCase结构

## 📋 概述

MCPToolMetric是基于mcp-bench的TaskEvaluator实现的评估指标，完全按照deepeval框架结构设计。该指标采用了统一的AgentTestCase结构，将MCPExecutionData集成到测试用例中，提供更加优雅和统一的数据管理方式。

## 🎯 核心设计理念

### 统一的AgentTestCase结构
- **继承LLMTestCase**: 完全兼容deepeval框架，保持所有原有功能
- **集成MCPExecutionData**: 将工具执行数据直接纳入测试用例中
- **Agent特有属性**: 支持agent_id、session_id、对话历史等Agent特有信息
- **向后兼容**: 支持从传统LLMTestCase转换和自动解析

### 两大类评估指标

#### 1. **LLM评估指标** (60%权重)
基于mcp-bench的LLMJudge实现，包含6个维度：

| 维度 | 描述 | 评分范围 |
|------|------|----------|
| **task_fulfillment** | 任务完成度 | 1-10分 |
| **grounding** | 信息准确性和基础性 | 1-10分 |
| **tool_appropriateness** | 工具选择的适当性 | 1-10分 |
| **parameter_accuracy** | 参数使用的准确性 | 1-10分 |
| **dependency_awareness** | 依赖关系感知能力 | 1-10分 |
| **parallelism_and_efficiency** | 并行处理和效率 | 1-10分 |

**聚合分数**:
- `task_completion_score` = (task_fulfillment + grounding) / 2
- `tool_selection_score` = (tool_appropriateness + parameter_accuracy) / 2  
- `planning_effectiveness_score` = (dependency_awareness + parallelism_and_efficiency) / 2

#### 2. **工具匹配指标** (40%权重)
基于mcp-bench的_calculate_tool_accuracy_metrics实现：

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| **input_schema_compliance** | 参数Schema合规性 | 合规调用数 / 有效工具调用数 |
| **valid_tool_name_rate** | 有效工具名称使用率 | 有效工具调用数 / 总调用数 |
| **execution_success_rate** | 执行成功率 | 成功执行数 / 总调用数 |
| **valid_call_failure_rate** | 有效调用失败率 | 有效工具失败数 / 有效工具调用数 |
| **planning_json_compliance** | 规划JSON合规性 | 规划格式正确性评分 |

## 🏗️ AgentTestCase数据结构

### 核心类定义
```python
class AgentTestCase(LLMTestCase):
    """Agent测试用例类
    
    继承自LLMTestCase，专门用于Agent和MCP工具的评估。
    将MCPExecutionData集成到测试用例中，提供统一的数据结构。
    """
    
    def __init__(
        self,
        # 继承自LLMTestCase的字段
        input: str,
        actual_output: str,
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
    )
```

### MCPExecutionData结构
```python
@dataclass
class MCPExecutionData:
    tool_executions: List[ToolExecutionResult]  # 工具执行结果列表
    total_rounds: int = 1                       # 总执行轮数
    planning_json_compliance: float = 1.0       # 规划JSON合规性
    concrete_task_description: Optional[str] = None      # 具体任务描述
    dependency_analysis: Optional[str] = None            # 依赖分析
    accumulated_information: Optional[str] = None        # 累积信息
```

### ToolExecutionResult结构
```python
@dataclass  
class ToolExecutionResult:
    tool: str                           # 工具名称
    parameters: Dict[str, Any]          # 调用参数
    result: Optional[Any] = None        # 执行结果
    success: bool = False               # 是否成功
    server: Optional[str] = None        # 服务器名称
    execution_time: Optional[float] = None      # 执行时间
    error_message: Optional[str] = None         # 错误信息
```

## 🚀 使用方法

### 推荐用法：AgentTestCase统一结构

```python
from evaluators.test_case import AgentTestCase
from evaluators.metrics.mcp_tool import MCPToolMetric
from evaluators.metrics.mcp_tool.schema import MCPExecutionData, ToolExecutionResult

# 1. 创建工具执行数据
execution_data = MCPExecutionData(
    tool_executions=[
        ToolExecutionResult(
            tool="weather_query",
            parameters={"location": "北京", "unit": "celsius"},
            result={"weather": "晴天", "temperature": 22},
            success=True,
            server="weather_server",
            execution_time=1.2
        ),
        ToolExecutionResult(
            tool="clothing_recommendation",
            parameters={"weather": "晴天", "temperature": 22},
            result={"recommendation": "轻薄长袖衬衫"},
            success=True,
            server="lifestyle_server",
            execution_time=0.8
        )
    ],
    total_rounds=1,
    planning_json_compliance=1.0,
    concrete_task_description="查询天气并推荐穿衣"
)

# 2. 创建AgentTestCase - 统一的数据结构
agent_test_case = AgentTestCase(
    # 基础字段
    input="请查询北京的天气并推荐合适的穿衣搭配",
    actual_output="根据查询，北京今天晴天22°C，建议穿轻薄长袖衬衫。",
    expected_output="应该提供天气信息和穿衣建议",
    
    # Agent特有字段
    execution_data=execution_data,  # 工具执行数据直接集成
    agent_id="weather_clothing_agent",
    session_id="session_20240928_001",
    available_tools=your_tools_definition,
    planning_strategy="sequential_execution",
    
    # 对话历史
    conversation_history=[
        {"role": "user", "content": "请查询北京的天气并推荐合适的穿衣搭配"},
        {"role": "assistant", "content": "我来帮您查询天气并推荐穿衣。"}
    ]
)

# 3. 创建指标并评估 - 结构统一，使用简单
metric = MCPToolMetric(threshold=0.7, include_reason=True)
score = metric.measure(agent_test_case)  # 只需要传入AgentTestCase
```

### 从LLMTestCase转换

```python
from veagentbench.evals.deepeval.test_case import LLMTestCase

# 原始的LLMTestCase
llm_test_case = LLMTestCase(
    input="查询天气",
    actual_output="今天晴天22°C"
)

# 转换为AgentTestCase
agent_test_case = AgentTestCase.from_llm_test_case(
    llm_test_case,
    execution_data=execution_data,
    agent_id="weather_agent",
    available_tools=tools_definition
)

# 评估
score = metric.measure(agent_test_case)
```

### 向后兼容用法

```python
# 传统方式仍然支持
llm_test_case = LLMTestCase(
    input="查询天气",
    actual_output="[调用工具: weather_query] ..."  # 包含工具调用信息
)

# 方式1: 传统LLMTestCase + execution_data
score = metric.measure(llm_test_case, execution_data)

# 方式2: 自动解析（向后兼容）
score = metric.measure(llm_test_case)  # 自动从actual_output解析
```

### 异步评估

```python
import asyncio

async def async_evaluation():
    score = await metric.a_measure(agent_test_case)
    return score

score = asyncio.run(async_evaluation())
```

## 📊 AgentTestCase的丰富功能

### 统计信息获取

```python
# 获取工具执行统计
stats = agent_test_case.get_tool_execution_stats()
print(f"总执行次数: {stats['total_executions']}")
print(f"成功率: {stats['success_rate']:.2%}")
print(f"平均执行时间: {stats['average_execution_time']:.2f}秒")

# 获取执行摘要
summary = agent_test_case.get_execution_summary()
print(f"执行摘要: {summary}")

# 获取对话上下文
context = agent_test_case.get_conversation_context()
print(f"对话历史:\n{context}")
```

### 对话管理

```python
# 添加对话轮次
agent_test_case.add_conversation_turn("user", "请继续")
agent_test_case.add_conversation_turn("assistant", "好的，我继续执行")

# 获取格式化的对话历史
conversation = agent_test_case.get_conversation_context()
```

### 序列化和持久化

```python
# 序列化为字典
case_dict = agent_test_case.to_dict()

# 保存到文件
import json
with open('test_case.json', 'w') as f:
    json.dump(case_dict, f, indent=2)

# 从字典恢复
restored_case = AgentTestCase.from_dict(case_dict)

# 数据验证
errors = agent_test_case.validate()
if errors:
    print(f"验证错误: {errors}")
```

## 📈 评估结果

### 完整的评估输出

```python
# 综合分数
print(f"最终分数: {metric.score:.3f}")
print(f"评估成功: {metric.success}")

# LLM评估指标 (1-10分制)
print(f"任务完成度: {metric.task_fulfillment}/10")
print(f"信息准确性: {metric.grounding}/10") 
print(f"工具适当性: {metric.tool_appropriateness}/10")
print(f"参数准确性: {metric.parameter_accuracy}/10")
print(f"依赖感知: {metric.dependency_awareness}/10")
print(f"并行效率: {metric.parallelism_and_efficiency}/10")

# 聚合分数
print(f"任务完成分数: {metric.task_completion_score}/10")
print(f"工具选择分数: {metric.tool_selection_score}/10")
print(f"规划效率分数: {metric.planning_effectiveness_score}/10")

# 工具匹配指标 (0-1范围)
print(f"Schema合规性: {metric.input_schema_compliance:.2%}")
print(f"工具名称有效率: {metric.valid_tool_name_rate:.2%}")
print(f"执行成功率: {metric.execution_success_rate:.2%}")
print(f"有效调用失败率: {metric.valid_call_failure_rate:.2%}")
print(f"规划JSON合规性: {metric.planning_json_compliance:.2%}")

# 服务器指标
print(f"使用服务器数量: {metric.server_count}")
print(f"跨服务器协调: {metric.cross_server_coordination}")
print(f"服务器分布: {metric.server_distribution}")

# 详细评估原因
if metric.include_reason:
    print(f"评估原因: {metric.reason}")
```

## 🔧 配置选项

### MCPToolMetric参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `threshold` | float | 0.7 | 成功阈值 |
| `model` | str/DeepEvalBaseLLM | None | 评估模型 |
| `include_reason` | bool | True | 是否包含详细原因 |
| `async_mode` | bool | True | 是否使用异步模式 |
| `strict_mode` | bool | False | 严格模式（阈值=1.0） |
| `verbose_mode` | bool | False | 详细日志模式 |
| `available_tools` | Dict | None | 可用工具定义（可从AgentTestCase自动获取） |

### AgentTestCase特有参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `execution_data` | MCPExecutionData | 工具执行数据 |
| `agent_id` | str | Agent标识符 |
| `session_id` | str | 会话标识符 |
| `conversation_history` | List[Dict] | 对话历史 |
| `available_tools` | Dict | 可用工具定义 |
| `planning_strategy` | str | 规划策略 |
| `execution_mode` | str | 执行模式 |

## 🎯 最佳实践

### 1. **使用AgentTestCase统一结构**
```python
# ✅ 推荐：使用AgentTestCase统一管理所有数据
agent_test_case = AgentTestCase(
    input="用户请求",
    actual_output="最终回答",
    execution_data=execution_data,  # 工具执行数据集成
    agent_id="my_agent",
    available_tools=tools_def
)

score = metric.measure(agent_test_case)  # 简洁的调用方式
```

### 2. **完整的工具执行信息**
```python
# ✅ 提供完整的执行信息
ToolExecutionResult(
    tool="tool_name",
    parameters={"param": "value"},
    result={"output": "result"},
    success=True,
    server="server_name",
    execution_time=1.2,
    error_message=None
)
```

### 3. **丰富的Agent上下文**
```python
# ✅ 提供丰富的Agent信息
AgentTestCase(
    # ... 基础字段
    agent_id="unique_agent_id",
    session_id="session_identifier", 
    planning_strategy="multi_step_reasoning",
    execution_mode="parallel_execution",
    conversation_history=[...]  # 完整对话历史
)
```

### 4. **数据验证和错误处理**
```python
# ✅ 验证数据完整性
errors = agent_test_case.validate()
if errors:
    print(f"数据验证失败: {errors}")
    return

# ✅ 异常处理
try:
    score = metric.measure(agent_test_case)
except Exception as e:
    print(f"评估失败: {e}")
```

### 5. **批量评估和序列化**
```python
# ✅ 批量评估
async def batch_evaluation(agent_test_cases):
    tasks = [metric.a_measure(case) for case in agent_test_cases]
    scores = await asyncio.gather(*tasks)
    return scores

# ✅ 持久化测试用例
test_cases_data = [case.to_dict() for case in agent_test_cases]
with open('test_cases.json', 'w') as f:
    json.dump(test_cases_data, f, indent=2)
```

## 🔍 故障排除

### 常见问题

1. **AgentTestCase导入失败**
   - 确保正确安装了evaluators模块
   - 检查Python路径配置

2. **execution_data为空**
   - 确保提供了完整的ToolExecutionResult列表
   - 检查工具执行数据的格式

3. **available_tools未自动获取**
   - 确保在AgentTestCase中设置了available_tools
   - 或者在MCPToolMetric初始化时显式提供

4. **序列化失败**
   - 检查execution_data中是否包含不可序列化的对象
   - 确保所有字段都是JSON兼容的类型

## 🆚 结构对比

### 传统方式 vs AgentTestCase

| 方面 | 传统方式 | AgentTestCase |
|------|----------|---------------|
| **数据结构** | 分离的LLMTestCase + execution_data | 统一的AgentTestCase |
| **调用方式** | `metric.measure(test_case, execution_data)` | `metric.measure(agent_test_case)` |
| **工具信息** | 需要单独管理available_tools | 集成在测试用例中 |
| **Agent信息** | 无法存储Agent特有信息 | 完整的Agent上下文 |
| **序列化** | 需要分别处理多个对象 | 一个对象包含所有信息 |
| **验证** | 手动验证各个组件 | 内置完整验证逻辑 |
| **统计** | 需要手动计算 | 内置丰富的统计方法 |

## 📈 性能优化

### 1. **批量评估优化**
```python
# 使用异步批量评估
async def optimized_batch_evaluation(agent_cases):
    # 预处理所有测试用例
    processed_cases = [case for case in agent_cases if case.validate() == []]
    
    # 并发评估
    semaphore = asyncio.Semaphore(10)  # 限制并发数
    
    async def evaluate_with_semaphore(case):
        async with semaphore:
            return await metric.a_measure(case)
    
    tasks = [evaluate_with_semaphore(case) for case in processed_cases]
    scores = await asyncio.gather(*tasks, return_exceptions=True)
    
    return scores
```

### 2. **内存管理**
```python
# 大批量处理时的内存优化
def process_large_dataset(test_cases_file):
    with open(test_cases_file, 'r') as f:
        for line in f:
            case_data = json.loads(line)
            agent_case = AgentTestCase.from_dict(case_data)
            
            # 评估
            score = metric.measure(agent_case)
            
            # 立即处理结果，释放内存
            yield score
            del agent_case
```

## 🔄 版本兼容性

- **v1.0**: 基础实现，使用LLMTestCase + execution_data
- **v1.1**: 引入MCPExecutionData优化输入结构
- **v1.2**: 创建AgentTestCase统一结构（当前版本）
- **向后兼容**: 完全支持所有历史版本的使用方式

## 📝 更新日志

### v1.2.0 (2024-09-28)
- ✅ 创建AgentTestCase统一结构
- ✅ 继承LLMTestCase，完全兼容deepeval框架
- ✅ 集成MCPExecutionData到测试用例中
- ✅ 添加Agent特有属性和方法
- ✅ 提供完整的序列化和验证功能
- ✅ 保持完全向后兼容

### v1.1.0 (2024-09-28)
- ✅ 优化输入结构设计
- ✅ 引入MCPExecutionData专门存放工具调用信息
- ✅ 保持向后兼容性

### v1.0.0 (2024-09-24)
- ✅ 基于mcp-bench实现两大类指标
- ✅ 完整的deepeval框架集成
- ✅ 支持同步/异步评估

---

**更新时间**: 2024-09-28  
**版本**: v1.2.0  
**兼容性**: deepeval >= 0.21.0, mcp-bench >= 1.0.0

## 🎉 总结

AgentTestCase提供了一个统一、优雅、功能完整的测试用例结构，将MCPExecutionData完美集成到deepeval框架中。这种设计不仅简化了使用方式，还提供了丰富的Agent特有功能，是Agent和MCP工具评估的理想解决方案。