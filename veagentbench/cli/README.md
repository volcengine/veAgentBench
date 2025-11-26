# VeAgentBench CLI 使用文档

VeAgentBench CLI 是一个统一的命令行工具，集成了配置生成和任务执行功能。

## 安装

```bash
# 安装包
pip install -e .

# 或者使用开发模式
pip install -e . --no-deps
```

## 基本用法

### 查看帮助信息

```bash
veagentbench --help
```

### 查看系统信息

```bash
# 查看可用指标
veagentbench info --metrics

# 查看可用代理
veagentbench info --agents

# 查看配置模板类型
veagentbench info --templates

# 查看所有信息
veagentbench info --metrics --agents --templates
```

## 配置生成

### 生成基础配置

```bash
veagentbench config generate --type basic --task-name my_test --output my_config.yaml
```

### 生成MCP测试配置

```bash
veagentbench config generate --type mcp --task-name mcp_tool_test --output mcp_config.yaml
```

### 生成RAG测试配置

```bash
veagentbench config generate --type rag --task-name rag_qa_test --output rag_config.yaml
```

### 自定义配置参数

```bash
veagentbench config generate \
  --type basic \
  --task-name custom_test \
  --output custom_config.yaml \
  --dataset-path example_dataset/mcptask/testcase.csv \
  --agent-class AdkAgent \
  --judge-model gpt-4 \
  --max-concurrent 10 \
  --cache-dir ./my_cache
```

### 多任务配置

```bash
# 首先创建多任务配置JSON文件
cat > multi_tasks.json << EOF
[
  {
    "name": "task1",
    "type": "mcp",
    "dataset_path": "example_dataset/mcptask/testcase.csv"
  },
  {
    "name": "task2", 
    "type": "rag",
    "dataset_path": "example_dataset/ragtask/testcase.csv"
  }
]
EOF

# 生成多任务配置
veagentbench config generate \
  --type multi \
  --output multi_config.yaml \
  --tasks-config multi_tasks.json
```

## 配置验证

```bash
veagentbench config validate --config my_config.yaml
```

## 任务执行

### 并行执行（默认）

```bash
veagentbench run --config my_config.yaml --parallel
```

### 顺序执行

```bash
veagentbench run --config my_config.yaml --sequential
```

### 自定义输出路径

```bash
veagentbench run --config my_config.yaml --output results/my_results.json
```

### 干运行模式（只验证配置）

```bash
veagentbench run --config my_config.yaml --dry-run
```

### 指定工作进程数

```bash
veagentbench run --config my_config.yaml --workers 4
```

## 完整示例

### 1. 生成MCP测试配置并执行

```bash
# 生成配置
veagentbench config generate \
  --type mcp \
  --task-name financial_research_test \
  --output financial_research_config.yaml \
  --mcp-dataset-path example_dataset/mcptask/testcase.csv

# 验证配置
veagentbench config validate --config financial_research_config.yaml

# 执行任务
veagentbench run --config financial_research_config.yaml --parallel
```

### 2. 生成RAG测试配置并执行

```bash
# 生成配置
veagentbench config generate \
  --type rag \
  --task-name knowledge_qa_test \
  --output knowledge_qa_config.yaml \
  --rag-dataset-path example_dataset/ragtask/testcase.csv

# 执行任务
veagentbench run --config knowledge_qa_config.yaml --sequential
```

### 3. 批量处理多个任务

```bash
# 生成多任务配置
veagentbench config generate \
  --type multi \
  --output batch_config.yaml \
  --tasks-config batch_tasks.json

# 执行所有任务
veagentbench run --config batch_config.yaml --workers 8
```

## 命令行参数详解

### config generate 参数

- `--type, -t`: 配置模板类型 (basic|mcp|rag|multi)
- `--output, -o`: 输出文件路径
- `--task-name`: 任务名称
- `--dataset-path`: 数据集文件路径
- `--judge-model`: 评判模型名称 (默认: gpt-4)
- `--judge-base-url`: 评判模型API基础URL
- `--judge-api-key`: 评判模型API密钥
- `--agent-class`: 代理类名 (AdkAgent|SimpleAgent)
- `--agent-endpoint`: 代理API端点
- `--api-key`: 代理API密钥
- `--max-concurrent`: 最大并发数 (默认: 5)
- `--cache-dir`: 缓存目录 (默认: ./cache)
- `--mcp-dataset-path`: MCP数据集路径
- `--rag-dataset-path`: RAG数据集路径
- `--tasks-config`: 多任务配置文件路径 (JSON格式)

### run 参数

- `--config, -c`: 配置文件路径 (必需)
- `--parallel, -p`: 并行执行 (默认)
- `--sequential, -s`: 顺序执行
- `--output, -o`: 结果输出路径
- `--workers, -w`: 工作进程数
- `--dry-run`: 干运行模式，只验证配置

### info 参数

- `--metrics`: 显示可用指标
- `--agents`: 显示可用代理
- `--templates`: 显示配置模板类型

## 错误处理

CLI工具提供了详细的错误信息和日志记录：

- 配置文件格式错误
- 数据集文件不存在
- 代理连接失败
- 指标创建失败
- 任务执行异常

所有错误都会显示详细的错误信息和堆栈跟踪，便于调试。

## 日志级别

默认日志级别为INFO，可以通过环境变量调整：

```bash
export LOG_LEVEL=DEBUG
veagentbench run --config my_config.yaml
```

支持的日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL

## 最佳实践

1. **先生成配置，再验证，最后执行**
   ```bash
   veagentbench config generate --type mcp --output config.yaml
   veagentbench config validate --config config.yaml
   veagentbench run --config config.yaml
   ```

2. **使用干运行模式测试配置**
   ```bash
   veagentbench run --config config.yaml --dry-run
   ```

3. **合理设置并发数**
   - 根据系统资源设置 `--max-concurrent`
   - 对于网络密集型任务，可以适当提高并发数
   - 对于CPU密集型任务，建议设置为CPU核心数

4. **使用缓存目录**
   - 指定专门的缓存目录避免重复计算
   - 定期清理缓存目录释放空间

5. **批量任务处理**
   - 使用多任务配置处理复杂的评测场景
   - 合理分配工作进程数避免资源竞争
