# veAgentBench

![veAgentBench Logo](assets/veagentbench_logo.jpeg)

**veAgentBench** 是面向AI Agent生态的专业评估框架，提供LLM裁判评分、多维度指标分析、工具调用匹配等核心能力，配套完整的端到端分析报表系统，助力构建可信的智能体评测体系。

## 🚀 最新发布

[2025/11/12] 🔥 **veAgentBench 正式开源** - 企业级AI Agent评估解决方案

## 🎯 核心优势

- **📊 多维度评估体系**：集成LLM裁判评分、工具匹配度、响应质量等全方位指标
- **🔍 深度指标分析**：提供细粒度的性能分解和中间指标透出
- **📈 可视化报表**：自动生成专业的分析报告，支持多格式输出
- **⚡ 高性能架构**：支持并发评测，优化评估效率
- **🔧 灵活扩展**：模块化设计，支持自定义评估指标和维度

## 🛠️ 环境要求

- **Python**: 3.10+
- **环境管理**: 推荐使用虚拟环境
- **依赖管理**: 支持uv/pip等主流工具

## 📦 快速安装

```bash
pip install git+https://code.byted.org/iaas/veAgentBench.git
```

## 🚀 快速开始

### 1. 环境配置

#### 安装uv（可选，推荐）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 配置环境变量

```bash
export VOLCEMODEL=          # 大模型推理接入点
export VOLCEBASEURL=        # 大模型推理baseurl
export ARK_API_KEY=         # 大模型推理API_KEY
export MAX_CONCURRENCY="5"  # 评测并发数
export VEAB_CACHE_DIR=".cache/veagentbench"  # 缓存目录
```

#### 开通火山方舟服务

默认使用 [Doubao-1.5-pro-32k](https://cloud.bytedance.net/ark/region:ark+cn-beijing/model/detail?Id=doubao-1-5-pro-32k) 模型
参考文档：[部署在线推理点](https://www.volcengine.com/docs/82379/1182403)

### 2. 安装依赖

```bash
uv sync  # 或 pip install -r requirements.txt
```

## 📊 评测方法

### 评测数据

1. **法律援助子数据集**（250个问题）：聚焦考察Agent的知识检索分层能力——优先从RAG知识库精准提取法律信息，当RAG覆盖不足时触发联网检索，确保回答准确性与时效性；
2. **财务分析子数据集**（57个问题）：重点验证Agent的多工具协同深度研究能力，考察其根据财务场景需求选择工具、组合调用流程及输出分析结论的能力；
3. **教育辅导子数据集**（74个问题）：增加RAG的数量级，进一步验证RAG提取信息的精确度，同时考察Agent通过memory获取关键信息的能力。

### 🎯 基于trace的评测

基于已有trace数据进行评估分析，适合结果复盘和性能分析场景。

#### 数据准备

准备Agent执行后的trace文件和eval_set文件，参考 `example_dataset/`

#### 执行评测

```bash
# 运行评测
uv run examples/mcptest.py

# 提取指标
uv run veagentbench/report/extract_metrics_corrected.py -o metrics_out -d
```

#### 输出结果

- `metrics_out_detailed.csv` - 详细评测数据
- `metrics_out_metrics_only.csv` - 核心指标
- `metrics_out_summary.json` - 统计摘要
- `metrics_out_report.html` - 可视化报告

### 🌐 动态评测

实时调用Agent进行动态评估，适合开发调试和性能监控场景。

#### Agent开发

```python
from veadk import Agent, Runner

# 定义工具函数
def stock_sse_summary():
    """上交所股票数据总貌查询接口"""
    # 工具实现逻辑
    pass

# 创建评测Agent
agent = Agent(
    name="financial_deep_research",
    instruction='财务分析专家，擅长股票数据统计分析',
    tools=[vesearch, stock_sse_summary],
)
```

#### 执行评测

```python
from veagentbench.task.mcp_task.mcptask import MCPTaskOnline

task = MCPTaskOnline(
    agents=[agent],
    task_name='金融分析场景',
    testset_file='example_dataset/mcptask/testcase.csv',
    enable_cache=True
)
task.evaluate()
```

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
