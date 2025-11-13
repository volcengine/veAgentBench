# Metrics 提取和分析工具

## 📋 概述

本目录包含用于提取和分析测试案例指标数据的工具集，能够正确提取每个test case对应的metricsData里面的score、reason等字段。

## 🛠️ 工具列表

### 1. `extract_metrics_corrected.py` - 完整数据提取器
**功能**: 从测试运行文件中提取完整的metrics数据，包括每个指标的score、reason、success等所有字段。

**使用方法**:
```bash
# 基础使用
python extract_metrics_corrected.py

# 指定输入文件和输出前缀
python extract_metrics_corrected.py --input .deepeval/.latest_test_run.json --output my_analysis

# 显示详细信息
python extract_metrics_corrected.py --show-details
```

**输出文件**:
- `*_detailed.csv`: 包含所有字段的详细数据
- `*_metrics_only.csv`: 仅包含指标分数和成功状态的简化表格
- `*_summary.json`: 统计汇总信息

### 2. `view_case_metrics.py` - 交互式查看器
**功能**: 提供多种方式查看和分析测试案例的指标数据。

**使用方法**:
```bash
# 查看特定案例的详细信息
python view_case_metrics.py --case 0 --reasons

# 显示所有案例的对比表格
python view_case_metrics.py --table --output comparison.csv

# 显示统计汇总
python view_case_metrics.py --stats

# 查看前几个案例（不显示原因）
python view_case_metrics.py
```

## 📊 数据结构说明

### MetricsData 结构
每个测试案例的 `metricsData` 是一个列表，包含多个指标对象：

```json
{
  "metricsData": [
    {
      "name": "Answer Correctness",
      "score": 0.285,
      "reason": "详细的评估原因...",
      "success": false,
      "threshold": 0.5,
      "strictMode": false,
      "evaluationModel": "Custom Volce OpenAI Model",
      "verboseLogs": "详细日志..."
    }
  ]
}
```

### 提取的字段
对于每个指标，工具会提取以下字段：
- `{metric_name}_score`: 指标分数
- `{metric_name}_reason`: 评估原因
- `{metric_name}_success`: 是否通过
- `{metric_name}_threshold`: 阈值
- `{metric_name}_strict_mode`: 严格模式
- `{metric_name}_evaluation_model`: 评估模型

## 📈 当前指标类型

根据测试数据，系统包含以下4种指标：

1. **Argument Correctness** (参数正确性)
   - 平均分数: 0.992
   - 通过率: 98.8%
   - 评估工具调用参数的正确性

2. **Contextual Recall** (上下文召回)
   - 平均分数: 0.711
   - 通过率: 71.1%
   - 评估检索上下文的完整性

3. **Tool Correctness** (工具正确性)
   - 平均分数: 0.627
   - 通过率: 62.7%
   - 评估工具调用的正确性

4. **Answer Correctness** (答案正确性)
   - 平均分数: 0.285
   - 通过率: 18.1%
   - 评估最终答案的正确性

## 🔧 使用示例

### 快速查看整体情况
```bash
python view_case_metrics.py --stats
```

### 分析特定失败案例
```bash
# 查看案例0的详细信息
python view_case_metrics.py --case 0 --reasons
```

### 生成完整分析报告
```bash
# 提取所有数据并生成多种格式的报告
python extract_metrics_corrected.py --show-details --output full_analysis

# 生成对比表格
python view_case_metrics.py --table --output metrics_comparison.csv
```

### 批量分析
```bash
# 提取详细数据
python extract_metrics_corrected.py --input .deepeval/.latest_test_run.json --output batch_analysis

# 查看统计信息
python view_case_metrics.py --stats --input .deepeval/.latest_test_run.json
```

## 📋 输出格式说明

### CSV格式 (详细数据)
包含每个测试案例的完整信息：
- 基础信息: case_id, case_name, input, actual_output, expected_output
- 执行信息: success, run_duration
- 指标数据: 每个指标的score, reason, success, threshold等

### CSV格式 (仅指标)
简化的指标对比表格：
- 案例基础信息
- 每个指标的分数和通过状态
- 便于Excel等工具进一步分析

### JSON格式 (统计汇总)
包含整体统计信息：
- 总案例数和成功率
- 每个指标的平均分数、分数范围、通过率等

## ⚠️ 注意事项

1. **数据结构**: metricsData是列表格式，不是字典
2. **指标名称**: 会自动转换为合法的列名（小写+下划线）
3. **文件路径**: 默认读取 `.deepeval/.latest_test_run.json`
4. **编码格式**: 所有输出文件使用UTF-8编码

## 🔄 版本历史

- **v1.1** (2025-09-24): 修正了metrics数据提取逻辑，正确处理metricsData列表结构
- **v1.0** (2025-09-24): 初始版本，存在数据结构理解错误

## 🚀 未来计划

- [ ] 支持多个测试运行文件的对比分析
- [ ] 添加可视化图表生成功能
- [ ] 实现指标趋势分析
- [ ] 支持自定义指标阈值设置