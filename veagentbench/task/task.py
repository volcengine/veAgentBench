
from veagentbench.task.base import BaseTask
from veagentbench.test_case import AgentTestCase
import asyncio
import tqdm
import json
import os
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from veagentbench.agents.adk_agents import AgentOutPut
from veagentbench.test_case.agent_test_case import ToolExecutionResult, ToolCallExpected
from veadk.utils.logger import get_logger
from veagentbench.evals.deepeval.evaluate import AsyncConfig, ErrorConfig, CacheConfig
from veagentbench.evals.deepeval import evaluate
from veagentbench.dataset.dataset import Dataset
from veagentbench.models.models import VolceOpenAI
from veagentbench.evals.deepeval.test_case import ToolCall
from veagentbench.agents.base_agent import BaseAgent
import traceback
logger = get_logger(__name__)
class AgentTask(BaseTask):
    enable_cache: bool = True
    user_id: str='VeAgentBench'
    max_concurrent: int = 10  # 默认并发数
    cache_dir: str = './cache'  # 缓存目录
    
    def __init__(
        self, 
        task_name: str, 
        metrics: list, 
        datasets: List[Dataset], 
        agent: BaseAgent=None, 
        max_concurrent: int = 1, 
        measure_concurrent: int=1, 
        cache_dir: str = None, 
        enable_cache: bool=True,
        enable_score_cache: bool=False
    ):
        """初始化AgentTask，支持设置最大并发数和缓存目录"""
        super().__init__(task_name=task_name, metrics=metrics, datasets=datasets, agent=agent)
        self.enable_cache = enable_cache
        if max_concurrent is not None:
            self.max_concurrent = max_concurrent
        if measure_concurrent is not None:
            self.measure_concurrent = measure_concurrent
        if cache_dir is not None:
            self.cache_dir = "%s/%s/%s"%(cache_dir, task_name, agent.agent_name)
        self.enable_score_cache = enable_score_cache
        logger.info(f"measure_concurrent: {self.measure_concurrent}")
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def run(self):
        measure_result = self.generate_agent_response()
        # logger.info(measure_result)
        return measure_result
    
    async def generate_agent_response_async(self, max_concurrent: int = None):
        """异步生成agent响应，支持并发执行和进度条显示"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
            
        measure_result = []
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        for dataset in self.datasets:
            # 获取所有testcases
            cache_dir = "%s/%s"%(self.cache_dir, dataset.name)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            testcases = list(dataset.get_testcase())
            total_testcases = len(testcases)
            
            logger.info(f"开始处理数据集 {dataset.name}，共 {total_testcases} 个测试用例，最大并发数: {max_concurrent}")
            
            # 创建所有agent测试用例
            agent_testcases = []
            for testcase in testcases:
                available_tools=[
                    ToolCall(
                        name=x.get('name', ''),
                        input_parameters=x.get('input_parameters', {}),
                        description=x.get('description', '')
                    ) for x in testcase.get("available_tools", [])
                ]
                available_tools_dict = {}
                for tool in available_tools:
                    available_tools_dict[tool.name] = tool
                agent_testcase = AgentTestCase(
                    input=testcase.get("input", ''),
                    expected_output=testcase.get("expected_output", ''),
                    expected_tools=[ToolCallExpected(
                        name=x.get('name', ''),
                        input_parameters=x.get('input_parameters', {}),
                        output=x.get('output', ''),
                        server=x.get('server', 'default'),
                    ) for x in testcase.get("expected_tools", [])],
                    available_tools=available_tools_dict
                )
                agent_testcases.append(agent_testcase)
            
            # 并发执行所有测试用例，带进度条
            with tqdm.tqdm(total=total_testcases, desc=f"处理 {dataset.name}") as pbar:
                async def run_with_semaphore(agent_testcase):
                    async with semaphore:
                        result = await self._run_one(agent_testcase, cache_dir)
                        pbar.update(1)
                        return result
                
                # 并发执行所有测试用例
                await asyncio.gather(*[run_with_semaphore(tc) for tc in agent_testcases])
            
            # 评估结果
            if agent_testcases:
                evaluate_result = evaluate(agent_testcases, self.metrics, 
                                             async_config=AsyncConfig(max_concurrent=self.measure_concurrent), 
                                             error_config=ErrorConfig(ignore_errors=False), 
                                             cache_config=CacheConfig(use_cache=self.enable_score_cache, write_cache=True)
                                            )
                _m_result = {
                    'dataset_name': dataset.name,
                    'measure_result': evaluate_result.model_dump_json(by_alias=True)
                }
                measure_result.append(_m_result)
                logger.info(f"数据集 {dataset.name} 处理完成，成功执行 {len(agent_testcases)} 个测试用例")
        
        return measure_result
    
    def _get_cache_key(self, testcase: AgentTestCase) -> str:
        """生成测试用例的缓存键"""
        # 基于输入和预期输出生成哈希键
        content = f"{testcase.input}_{testcase.expected_output}_{self.task_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, cache_dir) -> Path:
        """获取缓存文件路径"""
        return Path(cache_dir) / f"{cache_key}.json"
    
    def _read_cache(self, testcase: AgentTestCase, cache_dir) -> Optional[Dict[str, Any]]:
        """读取缓存"""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(testcase)
        cache_path = self._get_cache_path(cache_key, cache_dir)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"缓存命中: {cache_key}")
                    return cache_data
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _write_cache(self, testcase: AgentTestCase, result: AgentOutPut, cache_dir: str):
        """写入缓存"""
        if result is None:
            return
        
        try:
            cache_key = self._get_cache_key(testcase)
            cache_path = self._get_cache_path(cache_key, cache_dir)
            
            # 准备缓存数据
            cache_data = {
                'input': testcase.input,
                'expected_output': testcase.expected_output,
                'actual_output': result.final_response,
                'tool_called': result.tool_called,
                'success': result.success,
                'task_name': self.task_name,
                'user_id': self.user_id,
                'timestamp': time.time()
            }
            print(cache_data)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"缓存写入成功: {cache_key}")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"写入缓存失败: {e}")
    
    def _apply_cache_to_testcase(self, testcase: AgentTestCase, cache_data: Dict[str, Any]):
        """将缓存数据应用到测试用例"""
        try:
            testcase.actual_output = cache_data.get('actual_output', '')
            tool_called = cache_data.get('tool_called', {})
            retrieved_context = []
            for _, tool_call in tool_called.items():
                if tool_call['name'] == 'load_knowledgebase':
                    output = json.loads(tool_call['output']['result'])
                    context_pieces = output['knowledges'] if 'knowledges' in output else []
                    retrieved_context = [c['content'] for c in context_pieces if 'content' in c]
            testcase.tools_called = [
                ToolExecutionResult(
                    name=x.get('name', ''),
                    input_parameters=x.get('input_parameters', {}),
                    output=x.get('output', ''),
                    success=x.get('success', True),
                    server=x.get('server', 'default'),
                ) for _, x in tool_called.items()
            ]
            testcase.retrieval_context = retrieved_context[:5]

            logger.info(f"应用缓存数据到测试用例")
        except Exception as e:
            logger.warning(f"应用缓存数据失败: {e}")
    
    def generate_agent_response(self, max_concurrent: int = None):
        """同步接口，调用异步版本"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        return asyncio.run(self.generate_agent_response_async(max_concurrent=max_concurrent))
    
    
    async def _run_one(self, testcase: AgentTestCase, cache_dir: str):
        """执行单个测试用例，支持缓存机制"""
        # 首先尝试读取缓存
        if self.enable_cache:
            try:
                cache_data = self._read_cache(testcase, cache_dir)
                if cache_data:
                    # 缓存命中，直接应用缓存数据
                    self._apply_cache_to_testcase(testcase, cache_data)
                    logger.info(f"测试用例缓存命中，跳过执行: {testcase.input[:50]}...")
                    if cache_data['success']:
                        return testcase
                    else:
                        logger.warning(f"缓存结果中，agent执行状态失败，重新调用agent: {testcase.input[:50]}...")
            except Exception as e:
                logger.warning(f"缓存读取失败，继续执行: {e}")
        
        try:
            # 找到可用的执行方法，按优先级选择
            fn = getattr(self.agent, 'generate_output')

            # 根据方法签名调用
            result: AgentOutPut = None
            result = await fn(prompt=testcase.input, user_id = self.user_id)

            # 将结果写回用例，方便后续评估
            if result is not None and (result.success):
                try:
                    testcase.actual_output = result.final_response
                    tool_called = result.tool_called
                    retrieved_context = []
                    # 提取load_knowledge知识的结果作为检索上下文
                    for name, tool_call in tool_called.items():
                        if name == 'load_knowledgebase':
                            output = json.loads(tool_call['output'])
                            context_pieces = tool_call['output']['response']['result']['knowledges'] if isinstance(output, dict) and 'response' in output and 'result' in output['response'] and 'knowledges' in output['response']['result'] else []
                            retrieved_context = [c['content'] for c in context_pieces if 'content' in c]
                    testcase.tools_called = [
                        ToolExecutionResult(
                            name = x.get('name', ''),
                            input_parameters = x.get('input_parameters', {}),
                            output = x.get('output', ''),
                            success = x.get('success', True),
                            server = x.get('server', 'default')
                        ) for _, x in tool_called.items()
                    ]
                    testcase.retrieval_context = retrieved_context
                    # 执行成功后写入缓存
                    # if self.enable_cache:
                    try:
                        await asyncio.to_thread(self._write_cache, testcase, result, cache_dir)
                    except Exception as e:
                        logger.warning(f"缓存写入失败: {e}")
                    
                    return testcase
                except Exception as e:
                    logger.exception(f"处理执行结果失败: {e}")
                    raise
            else:
                raise Exception("testcase fail: %s"%result.final_response)
        except Exception as e:
            logger.exception(f"Execution failed for testcase with agent: {e}")
            return testcase
    
from veagentbench.metrics.mcp_bench._mcp_tool import MCPToolMetric
import os
from veagentbench.agents.adk_agents import AdkAgent
if __name__ == "__main__":
    dataset = dataset = Dataset(name='test', description='test')
    dataset.load(
        load_type='csv', 
        csv_file='example_dataset/mcptask/testcase2.csv', 
        input_column='input', 
        expected_column='expect_output', 
        expected_tool_call_column='expected_tool_calls'
    )
    llm_model = VolceOpenAI(
        model = os.environ.get("VOLCEMODEL"),
        temperature=0,
        base_url=os.environ.get("VOLCEBASEURL"),
        cost_per_input_token=0.000002,
        cost_per_output_token=0.000008,
        _openai_api_key=os.environ.get("ARK_API_KEY"),
            
    )
    metrics = [MCPToolMetric(model=llm_model, enable_judge_stability=False)]

    task = AgentTask(
        task_name='test',
        metrics=metrics,
        datasets=[dataset],
        agent=AdkAgent(
            end_point="http://127.0.0.1:8000/invoke",
            api_key= "IRa2yiORMmZqoCBmn_bn5hZJ_LVtcLSYdvBT"
        ),
        
    )
    
    task.run()
