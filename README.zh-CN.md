# veAgentBench

<a href="https://huggingface.co/datasets/bytedance-research/veAgentBench" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg?style=for-the-badge"
         alt="Hugging Face Dataset" />
</a>

![veAgentBench Logo](assets/veagentbench_logo.jpeg)

**veAgentBench** 是面向AI Agent生态的专业评估框架，内置评估工具和数据集，提供LLM裁判评分、多维度指标分析、工具调用匹配等核心能力，配套完整的端到端分析报表系统，助力构建可信的智能体评测体系。

## 🚀 最新发布

[2025/11/12] 🔥 **veAgentBench 正式开源工具+评测集** - 企业级AI Agent评估解决方案

## 项目介绍

### 🎯 核心优势

- **多维度评估体系**：集成LLM裁判评分、工具匹配度、响应质量等全方位指标
- **深度指标分析**：提供细粒度的性能分解和中间指标透出
- **可视化报表**：自动生成专业的分析报告，支持多格式输出
- **高性能架构**：支持并发评测，优化评估效率
- **灵活扩展**：模块化设计，支持自定义评估指标和维度
- **支持多种评测对象接入**：本地开发对象、http+sse、a2a

### 内置评测集

配套法律、教育、金融分析、个人助理评测数据，支持一键引用评测。数据集详细介绍见:[veAgentBench-data](https://huggingface.co/datasets/bytedance-research/veAgentBench)

## 快速开始

### 环境要求

- **Python**: 3.10+
- **环境管理**: 推荐使用虚拟环境
- **依赖管理**: 支持uv/pip等主流工具

### 安装

```bash
pip install git+https://github.com/volcengine/veAgentBench.git
```

### 命令行使用指南

#### 查看帮助信息

```bash
veagentbench --help
```

#### 查看基本信息

```bash
# 查看可用指标
veagentbench info --metrics

# 查看可用代理
veagentbench info --agents

# 查看配置模板类型
veagentbench info --templates
```

#### 生成基础配置

```bash
veagentbench config generate --task-name my_test --output my_config.yaml
```

#### 并行执行（默认）

```bash
veagentbench run --config my_config.yaml --parallel
```

#### 顺序执行

```bash
veagentbench run --config my_config.yaml --sequential
```

### 配置文件说明

```yaml
tasks:
  - name: legal_assistant    # 评测任务名称
    datasets:
      - name: bytedance-research/veAgentBench   # 测试集名称
        description: 法律援助助手                 # 测试集描述
        property:                               # 测试集相关属性
          type: huggingface                     # 测试集类型，支持csv、huggingface
          config_name: legal_aid   
          split: "test[:1]"        
          input_column: "input"
          expected_output_column: "expect_output"
    metrics:                                    # 评测指标
      - AnswerCorrectnessMetric
    judge_model:                   # 裁判模型配置
      model_name: "gpt-4"          # 模型名称
      base_url: "https://api.openai.com/v1"  # OpenAPI的base_url
      api_key: "your_api_key"      # API密钥（需要替换）
    agent:                         # 被测Agent配置
      type: AdkAgent              # 被测Agent类型：AdkAgent/LocalAdkAgent/A2AAgent
      property:
        agent_name: "financial_analysis_agent"  # Agent名称
        end_point: "http://127.0.0.1:8000/invoke"  # 调用端点
        api_key: "your_api_key"     # Agent API密钥（需要替换）
    max_concurrent: 5              # 调用被测agent并发数
    measure_concurrent: 100        # 评测并发数：100个样本
    cache_dir: "./cache"           # 缓存目录路径
```

#### 测试集配置说明

##### HuggingFace测试集配置

```yaml
    datasets:
      - name: bytedance-research/veAgentBench   # HuggingFace测试集名称
        description: 金融分析测试集
        property:
          type: huggingface                    # 测试集类型
          config_name: financial_analysis      # subset名称
          split: "test[:1]"                    # split，可以不用填，如果要跑少量case，可以指定
          input_column: "input"                 # 输入列名
          expected_output_column: "expect_output"   # 预期响应列名
          expected_tool_call_column: "expected_tool_calls"  # 预期工具调用列名
```

##### 本地CSV文件测试集配置

```yaml
    datasets:
      - name: legal                     # 测试集名称
        description: 法律咨询客服评测集    # 测试集描述
        property:
          type: csv                     # 测试集类型
          csv_file_path: "dataset/test1.csv"       # 测试集本地文件
          input_column: "input"                    # 输入列名
          expected_output_column: "expect_output"   # 预期响应列名
          expected_tool_call_column: "expected_tool_calls"    # 预期工具调用列名
```

#### 被测对象agent配置说明

##### agentkit platform agent接入

```yaml
    agent:                         # 被测Agent配置
      type: AdkAgent              # 被测Agent类型：AdkAgent/LocalAdkAgent/A2AAgent
      property:
        agent_name: "financial_analysis_agent"  # Agent名称
        end_point: "http://127.0.0.1:8000/invoke"  # 调用端点
        api_key: "your_api_key"     # Agent API密钥（需要替换）
```

##### 本地通过agentkit开发的agent对象

```yaml
  agent:
    type: LocalAdkAgent   
    property:
      agent_name: local_finantial_agent  
      agent_dir_path: "agents/legal"        # 本地agent对象目录
```

### 离线评测

离线评测适用于已有评测数据的场景，适合上线前的效果准出评测。

#### 内置Benchmark评测集评测

veAgentBench 提供了内置评测数据集，覆盖多个专业领域：

**1. 准备评测配置**

准备评测配置test_config.yaml，示例参考如下：

**财务分析评测配置：**

```yaml
tasks:
  - name: financial_analysis_test
    datasets:
      - name: bytedance-research/veAgentBench   # HuggingFace测试集名称
        description: 金融分析测试集
        property:
          type: huggingface
          config_name: financial_analysis      # subset名称
          split: "test[:1]"                    # split，可以不用填，如果要跑少量case，可以指定
          input_column: "input"
          expected_output_column: "expect_output"
          expected_tool_call_column: "expected_tool_calls"
    metrics: ["MCPToolMetric"]
    judge_model:                   # 裁判模型配置
      model_name: "gpt-4"          # 模型名称
      base_url: "https://api.openai.com/v1"  # OpenAPI的base_url
      api_key: "your_api_key"      # API密钥（需要替换）
    agent:                         # 被测Agent配置
      type: AdkAgent              # 被测Agent类型：AdkAgent/LocalAdkAgent/A2AAgent
      property:
        agent_name: "financial_analysis_agent"  # Agent名称
        end_point: "http://127.0.0.1:8000/invoke"  # 调用端点
        api_key: "your_api_key"     # Agent API密钥（需要替换）
    max_concurrent: 5              # 调用被测agent并发数
    measure_concurrent: 100        # 评测并发数：100个样本
    cache_dir: "./cache"           # 缓存目录路径
```

**法律援助评测配置：**

```yaml
tasks:
  - name: legal_assistant
    datasets:
      - name: bytedance-research/veAgentBench   # HuggingFace测试集名称
        description: 法律援助助手
        property:
          type: huggingface
          config_name: legal_aid       # subset名称
          split: "test[:1]"                    # split，可以不用填，如果要跑少量case，可以指定
          input_column: "input"
          expected_output_column: "expect_output"
    metrics:
      - AnswerCorrectnessMetric
      - AnswerRelevancyMetric
      - ContextualPrecisionMetric
      - ContextualRecallMetric
      - FaithfulnessMetric
      - ContextualRelevancyMetric
    judge_model:                   # 裁判模型配置
      model_name: "gpt-4"          # 模型名称
      base_url: "https://api.openai.com/v1"  # OpenAPI的base_url
      api_key: "your_api_key"      # API密钥（需要替换）
    agent:                         # 被测Agent配置
      type: AdkAgent              # 被测Agent类型：AdkAgent/LocalAdkAgent/A2AAgent
      property:
        agent_name: "financial_analysis_agent"  # Agent名称
        end_point: "http://127.0.0.1:8000/invoke"  # 调用端点
        api_key: "your_api_key"     # Agent API密钥（需要替换）
    max_concurrent: 5              # 调用被测agent并发数
    measure_concurrent: 100        # 评测并发数：100个样本
    cache_dir: "./cache"           # 缓存目录路径
```

**2. 准备被测对象**

参照[veAgentBench-agent](https://huggingface.co/datasets/bytedance-research/veAgentBench/tree/main/agents) 对应的agents文件，在本地开发，或部署到火山agentkit platform进行评测。

**3. 执行测试命令**

```bash
veagentbench run --config test_config.yaml  --parallel
```

#### 自定义数据集评测

支持用户使用自己的数据集进行评测，灵活适应各种业务场景：

**1. 数据格式要求**

- **CSV格式**：支持本地CSV文件，包含输入、期望输出、期望工具调用等列
- **HuggingFace格式**：支持从HuggingFace Hub加载数据集

**2. 配置自定义数据集**

```yaml
# CSV数据集配置示例，一般要求必须有input_column、expected_output_column，
datasets:
  - name: custom_testset
    property:
      type: csv  # 数据集类型：csv/huggingface/trace
      csv_file_path: "path/to/your/dataset.csv"  # 数据文件路径
      input_column: "question"  # 输入列名
      expected_output_column: "expected_answer"  # 期望输出列名
      expected_tool_call_column: "expected_tools"  # 期望工具调用列名
```

**3. 执行测试命令**

```bash
veagentbench run --config test_config.yaml  --parallel
```

### 在线评测（预留）

在线评测功能正在开发中，将支持实时调用Agent进行动态评估，适合在线agent性能监控场景。

**即将支持的功能：**

- 🔌 实时Agent调用和评测
- 📊 动态性能监控
- ⚡ 开发调试支持
- 🔄 持续集成集成
- 📈 实时指标展示

## 🗺️ 产品路线图

### 近期规划

- [ ] 扩展Agent框架支持（LangChain、AutoGPT等）
- [ ] 增加领域专用评估指标（金融、医疗、法律等）
- [ ] 优化评测性能和并发处理能力
- [ ] 完善可视化分析功能

### 长期愿景

- [ ] 支持分布式评测架构
- [ ] 建立行业标准评估体系

## 🤝 参与贡献

我们欢迎社区开发者参与veAgentBench的建设：

- 📋 提交Issue反馈问题和建议
- 🔧 贡献代码和文档改进
- 📊 分享使用案例和最佳实践
- 💡 提出新功能需求

## 📄 开源许可

基于 **Apache 2.0** 许可证开源 - 详见 [LICENSE](LICENSE)

---

**veAgentBench** - 专业、可信、高效的AI Agent评估框架
