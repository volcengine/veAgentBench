## 前置准备

### 环境准备

```bash
uv pip install git+https://code.byted.org/pris/mem0.git
uv pip install vikingdb-python-sdk
uv pip install git+https://github.com/brianxb001/veadk-python.git@memtest   #安装制定版本veadk，主要是适配了viking memory关于记忆使用方式
```

### 数据集准备

```bash
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

### 环境变量

```bash
export PATH_TO_LONGMEM_S_DATA=""    #longmemeval_s_cleaned.json 数据集路径
export VIKING_MEMORY_API_KEY=""      #viking记忆库的api_key，注意需要给key添加特定的记忆库权限
export VOLCENGINE_ACCESS_KEY=""
export VOLCENGINE_SECRET_KEY=""


#mem0记忆库环境变量配置，从agentkit 记忆库中的集成代码中拷贝
export DATABASE_MEM0_BASE_URL=https://mem0-cnlfjzigaku8gczkzo.mem0.volces.com:8000
export DATABASE_MEM0_API_KEY=53f9ff1d-a5d0-5024-a80f-f3a6e9ccc3ce
export MEM0_BATCHSIZE    #mem0记忆库导入时的batchsize配置，默认为4

#viking记忆库环境变量配置，从agentkit 记忆库中的集成代码中拷贝
export DATABASE_VIKINGMEM_COLLECTION=longmem_s_default
export DATABASE_VIKINGMEM_MEMORY_TYPE=memory_summary_as6ox5,memory_semantic_hrei4c,memory_userpreference_5u7np1

```

## 记忆库导入

```bash
python  add_mem0_longmem.py   #导入mem0记忆。导入开始到解析完成预计要等一晚上
python add_viking_longmem.py   #导入viking记忆，预计要2小时
```

## 执行评测

```
veagentbench run --config samples/longmem_eval/test_mem0.yaml     #测试mem0
veagentbench run --config samples/longmem_eval/test_viking.yaml    #测试viking
```
