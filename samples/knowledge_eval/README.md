## 前置准备

### 环境准备

```bash
uv pip install veadk-python
```

### 环境变量

```bash
export DATABASE_TOS_BUCKET=""  # 数据集上传TOS实例名称
export VOLCENGINE_ACCESS_KEY=
export VOLCENGINE_SECRET_KEY=
export DATABASE_TYPE=viking  #  knowledge 类型
export DATABASE_COLLECTION=  # 知识库库名称

```

## 记忆库导入

```bash
python import.py
```

## 执行评测

### 修改任务配置文件
bench_config.yaml，修改judge_model的配置即可。

### 任务执行
```
veagentbench run --config samples/knowledge_eval/bench_config.yaml 
```
