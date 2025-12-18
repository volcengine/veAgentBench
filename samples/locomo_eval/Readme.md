## 前置准备

### 环境准备

```bash
uv pip install git+https://code.byted.org/pris/mem0.git
uv pip install vikingdb-python-sdk
uv pip install git+https://github.com/brianxb001/veadk-python.git@memtest   #安装制定版本veadk，主要是适配了viking memory关于记忆使用方式
```


### 环境变量

```bash
export PATH_TO_LOCOMO_DATA="samples/locomo_eval/locomo10.json"    #longmemeval_s_cleaned.json 数据集路径
export VIKING_MEMORY_API_KEY=""      #viking记忆库的api_key，注意需要给key添加特定的记忆库权限
export VOLCENGINE_ACCESS_KEY=""
export VOLCENGINE_SECRET_KEY=""


#mem0记忆库环境变量配置，从agentkit 记忆库中的集成代码中拷贝
export DATABASE_MEM0_BASE_URL=
export DATABASE_MEM0_API_KEY=
export MEM0_BATCHSIZE=    #mem0记忆库导入时的batchsize配置，默认为4

#viking记忆库环境变量配置，从agentkit 记忆库中的集成代码中拷贝
export DATABASE_VIKINGMEM_COLLECTION=
export DATABASE_VIKINGMEM_MEMORY_TYPE=
export MEMORY_API_KEY=

```

## 记忆库导入

```bash
python  add_locomo_mem0.py   #导入mem0记忆
python add_locomo_viking.py   #导入viking记忆，

## 执行评测

### 修改任务配置文件

test_mem0.yaml和test_viking.yaml中，修改judge_model的配置即可。

### 任务执行

```
veagentbench run --config samples/locomo_eval/test_mem0.yaml     #测试mem0
veagentbench run --config samples/locomo_eval/test_viking.yaml    #测试viking
```
