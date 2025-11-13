# veAgentBench

![veAgentBench Logo](assets/veagentbench_logo.jpeg)

**veAgentBench** is a professional evaluation framework for AI Agent ecosystems, providing core capabilities such as LLM judge scoring, multi-dimensional metric analysis, and tool call matching. It comes with a complete end-to-end analysis and reporting system, helping to build a trustworthy intelligent agent evaluation system.

> **📖 中文版本** | [查看中文文档](./README.zh-CN.md)

## 🚀 Latest Release

[2025/11/12] 🔥 **veAgentBench Officially Open Source** - Enterprise-grade AI Agent Evaluation Solution

## 🎯 Core Advantages

- **📊 Multi-dimensional Evaluation System**: Integrates comprehensive metrics including LLM judge scoring, tool matching accuracy, and response quality
- **🔍 Deep Metric Analysis**: Provides fine-grained performance breakdown and intermediate metric visibility
- **📈 Visualized Reporting**: Automatically generates professional analysis reports with multi-format output support
- **⚡ High-performance Architecture**: Supports concurrent evaluation with optimized assessment efficiency
- **🔧 Flexible Extension**: Modular design supporting custom evaluation metrics and dimensions

## 🛠️ Environment Requirements

- **Python**: 3.10+
- **Environment Management**: Virtual environment recommended
- **Dependency Management**: Supports mainstream tools like uv/pip

## 📦 Quick Installation

```bash
pip install git+https://code.byted.org/iaas/veAgentBench.git
```

## 🚀 Quick Start

### 1. Environment Configuration

#### Install uv (Optional, Recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Configure Environment Variables
```bash
export VOLCEMODEL=          # Large model inference endpoint
export VOLCEBASEURL=        # Large model inference base URL
export ARK_API_KEY=         # Large model inference API key
export MAX_CONCURRENCY="5"  # Evaluation concurrency
export VEAB_CACHE_DIR=".cache/veagentbench"  # Cache directory
```

#### Enable Volcano Engine Service
Uses [Doubao-1.5-pro-32k](https://cloud.bytedance.net/ark/region:ark+cn-beijing/model/detail?Id=doubao-1-5-pro-32k) model by default  
Reference: [Deploy Online Inference Endpoint](https://www.volcengine.com/docs/82379/1182403)

### 2. Install Dependencies
```bash
uv sync  # or pip install -r requirements.txt
```

## 📊 Evaluation Methods

### Evaluation Datasets
1. **Legal Aid Sub-dataset** (250 questions): Focuses on evaluating Agent's knowledge retrieval hierarchical capabilities - prioritizing accurate legal information extraction from RAG knowledge base, triggering web search when RAG coverage is insufficient to ensure answer accuracy and timeliness;
2. **Financial Analysis Sub-dataset** (57 questions): Emphasizes validating Agent's multi-tool collaborative deep research capabilities, examining its ability to select tools based on financial scenario requirements, combine call workflows, and output analytical conclusions;
3. **Educational Tutoring Sub-dataset** (74 questions): Increases RAG scale to further validate RAG information extraction accuracy, while examining Agent's ability to obtain key information through memory.

### 🎯 Offline Evaluation

Based on existing trace data for assessment and analysis, suitable for result review and performance analysis scenarios.

#### Data Preparation
Prepare Agent execution trace files and eval_set files, refer to `example_dataset/`

#### Execute Evaluation
```bash
# Run evaluation
uv run examples/mcptest.py

# Extract metrics
uv run veagentbench/report/extract_metrics_corrected.py -o metrics_out -d
```

#### Output Results
- `metrics_out_detailed.csv` - Detailed evaluation data
- `metrics_out_metrics_only.csv` - Core metrics
- `metrics_out_summary.json` - Statistical summary
- `metrics_out_report.html` - Visualized report

### 🌐 Online Evaluation

Real-time Agent invocation for dynamic assessment, suitable for development debugging and performance monitoring scenarios.

#### Agent Development
```python
from veadk import Agent, Runner

# Define tool functions
def stock_sse_summary():
    """Shanghai Stock Exchange stock data overview query interface"""
    # Tool implementation logic
    pass

# Create evaluation Agent
agent = Agent(
    name="financial_deep_research",
    instruction='Financial analysis expert, proficient in stock data statistical analysis',
    tools=[vesearch, stock_sse_summary],
)
```

#### Execute Evaluation
```python
from veagentbench.task.mcp_task.mcptask import MCPTaskOnline

task = MCPTaskOnline(
    agents=[agent],
    task_name='Financial Analysis Scenario',
    testset_file='example_dataset/mcptask/testcase.csv',
    enable_cache=True
)
task.evaluate()
```

## 🗺️ Product Roadmap

### Near-term Plans
- [ ] Expand Agent framework support (LangChain, AutoGPT, etc.)
- [ ] Add domain-specific evaluation metrics (finance, healthcare, legal, etc.)
- [ ] Optimize evaluation performance and concurrent processing capabilities
- [ ] Enhance visualized analysis features

### Long-term Vision
- [ ] Support distributed evaluation architecture
- [ ] Establish industry-standard evaluation system

## 🤝 Contributing

We welcome community developers to participate in veAgentBench development:

- 📋 Submit Issues for feedback and suggestions
- 🔧 Contribute code and documentation improvements
- 📊 Share use cases and best practices
- 💡 Propose new feature requirements

## 📄 Open Source License

Open source under **Apache 2.0** license - see [LICENSE](LICENSE) for details

---

**veAgentBench** - Professional, Trustworthy, and Efficient AI Agent Evaluation Framework
