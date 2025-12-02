## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http:##www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


from veagentbench.task.task import AgentTask
from veagentbench.dataset.dataset import Dataset
from veagentbench.models.models import VolceOpenAI
from veagentbench.agents.adk_agents import AdkAgent
from veagentbench import metrics as metrics_module
from veagentbench import agents as agents_module
from typing import List, Dict, Any, Optional
import yaml
import os
import asyncio
import multiprocessing
from veadk.utils.logger import get_logger
from pathlib import Path
from datetime import datetime
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import traceback
# 设置日志
logger = get_logger(__name__)

class AgentTestRunner:
    """Agent测试运行器，支持从配置文件加载任务并执行多进程评测"""
    
    def __init__(self, config_path: str = None, max_workers: int = None):
        """
        初始化运行器
        
        Args:
            config_path: 配置文件路径
            max_workers: 最大工作进程数，默认为CPU核心数
        """
        self.config_path = config_path
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.tasks: List[AgentTask] = []
        self.config_data: Dict[str, Any] = None
        
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        加载任务配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径
            
        Returns:
            配置数据字典
        """
        if config_path:
            self.config_path = config_path
            
        if not self.config_path:
            raise ValueError("配置文件路径不能为空")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
                
            logger.info(f"成功加载配置文件: {self.config_path}")
            return self.config_data
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _create_dataset(self, dataset_config: Dict[str, Any]) -> Dataset:
        """根据配置创建数据集"""
        dataset = Dataset(
            name=dataset_config.get('name', 'unknown'),
            description=dataset_config.get('description', '')
        )
        
        property_config = dataset_config.get('property', {})
        load_type = property_config.get('type', 'csv')
        
        if load_type == 'csv':
            dataset.load(
                load_type='csv',
                csv_file=property_config.get('csv_file_path', ''),
                input_column=property_config.get('input_column', 'input'),
                expected_column=property_config.get('expected_output_column', 'expected_output'),
                expected_tool_call_column=property_config.get('expected_tool_call_column', 'expected_tool_calls')
            )
        elif load_type == 'huggingface':
            dataset.load(
                load_type='huggingface',
                config_name=property_config.get('config_name', ''),
                split=property_config.get('split', 'test'),
                input_key=property_config.get('input_key', 'input'),
                expected_key=property_config.get('expected_key', 'expected_output'),
                expected_tool_call_key=property_config.get('expected_tool_call_key', 'expected_tool_calls')
            )
        else:
            raise ValueError(f"不支持的数据集类型: {load_type}")
            
        return dataset
    
    def _create_metrics(self, metric_names: List[str], judge_model_config: Dict[str, Any]) -> List[Any]:
        """根据配置创建评测指标"""
        metrics = []
        
        # 创建评判模型
        judge_model = VolceOpenAI(
            model=judge_model_config.get('model_name', 'gpt-4'),
            base_url=judge_model_config.get('base_url', ''),
            _openai_api_key=judge_model_config.get('api_key', ''),
            temperature=0,
            cost_per_input_token=0.000002,
            cost_per_output_token=0.000008
        )
        
        for metric_name in metric_names:
            try:
                # 动态获取指标类
                if hasattr(metrics_module, metric_name):
                    metric_class = getattr(metrics_module, metric_name)
                    
                    # 根据指标类型创建实例

                    metrics.append(metric_class(model=judge_model))
                    
                    logger.info(f"成功创建指标: {metric_name}")
                else:
                    logger.warning(f"未知的指标类型: {metric_name}，可用的指标: {', '.join([attr for attr in dir(metrics_module) if not attr.startswith('_')])}")
                    
            except Exception as e:
                logger.error(f"创建指标 {metric_name} 失败: {e}")
                continue
                
        return metrics
    
    def _create_agent(self, agent_config: Dict[str, Any]) -> Any:
        """根据配置创建代理"""
        agent_class = agent_config.get('type', 'AdkAgent')
        property_config = agent_config.get('property', {})
        
        try:
            # 通过 agents_module 动态获取代理类
            if hasattr(agents_module, agent_class):
                agent_cls = getattr(agents_module, agent_class)
                
                # 根据代理类型创建实例
                return agent_cls(**property_config)
            else:
                raise ValueError(f"不支持的代理类型: {agent_class}，可用的代理类型: {[attr for attr in dir(agents_module) if not attr.startswith('_') and attr.endswith('Agent')]}")
                
        except Exception as e:
            logger.error(f"创建代理 {agent_class} 失败: {e}")
            raise ValueError(f"创建代理 {agent_class} 失败: {str(e)}")
    
    def create_tasks_from_config(self) -> List[AgentTask]:
        """
        从配置数据创建AgentTask列表
        
        Returns:
            AgentTask列表
        """
        if not self.config_data:
            raise ValueError("配置数据未加载，请先调用load_config()")
            
        tasks = []
        
        for task_config in self.config_data.get('tasks', []):
            try:
                # 创建数据集
                datasets = []
                for dataset_config in task_config.get('datasets', []):
                    datasets.append(self._create_dataset(dataset_config))
                
                # 创建指标
                judge_model_config = task_config.get('judge_model', {})
                metric_names = task_config.get('metrics', [])
                metrics = self._create_metrics(metric_names, judge_model_config)
                
                # 创建代理
                agent_config = task_config.get('agent', {})
                agent = self._create_agent(agent_config)
                
                # 创建任务
                task = AgentTask(
                    enable_cache=task_config.get('enable_cache', False),
                    task_name=task_config.get('name', 'unknown_task'),
                    metrics=metrics,
                    datasets=datasets,
                    agent=agent,
                    max_concurrent=task_config.get('max_concurrent', 10),
                    cache_dir=task_config.get('cache_dir', './cache'),
                    measure_concurrent=task_config.get('measure_concurrent', 10),
                    enable_score_cache = task_config.get('enable_score_cache', False)
                )
                
                tasks.append(task)
                logger.info(f"成功创建任务: {task.task_name}")
                
            except Exception as e:
                logger.error(f"创建任务失败: {e}")
                logger.error(traceback.format_exc())
                continue
                
        self.tasks = tasks
        return tasks
    
    def _run_single_task(self, task: AgentTask, task_id: int) -> Dict[str, Any]:
        """
        运行单个任务（用于多进程）
        
        Args:
            task_config: 任务配置字典
            task_id: 任务ID
            
        Returns:
            任务执行结果
        """
        try:
            task_name = task.task_name
            logger.info(f"开始执行任务 {task_id}: {task_name}")
            
            # 重新创建任务对象（避免序列化问题）
            # task = self._create_task_from_config(task_config)
            
            # 运行任务
            result = task.run()
            
            # 准备结果（只保存可序列化的数据）
            task_result = {
                'task_id': task_id,
                'task_name': task_name,
                'status': 'success',
                'result': self._serialize_result(result),
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'error': None
            }
            
            logger.info(f"任务 {task_name} 执行成功")
            return task_result
            
        except Exception as e:
            logger.error(f"任务 {task_name} 执行失败: {e}")
            traceback.print_exc()
            error_result = {
                'task_id': task_id,
                'task_name': task_name,
                'status': 'failed',
                'result': None,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'error': str(e)
            }
            return error_result
    
    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """序列化任务结果，确保所有数据都是可pickle的"""
        if isinstance(result, dict):
            # 处理字典类型的结果
            serialized = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serialized[key] = value
                else:
                    # 将复杂对象转换为字符串
                    serialized[key] = str(value)
            return serialized
        elif isinstance(result, (str, int, float, bool)):
            return {'result': result}
        else:
            # 其他类型转换为字符串
            return {'result': str(result)}
    
    def _create_task_from_config(self, task_config: Dict[str, Any]) -> AgentTask:
        """从配置创建单个任务对象"""
        try:
            # 创建数据集
            datasets = []
            for dataset_config in task_config.get('datasets', []):
                datasets.append(self._create_dataset(dataset_config))
            
            # 创建指标
            judge_model_config = task_config.get('judge_model', {})
            metric_names = task_config.get('metrics', [])
            metrics = self._create_metrics(metric_names, judge_model_config)
            
            # 创建代理
            agent_config = task_config.get('agent', {})
            agent = self._create_agent(agent_config)
            
            # 创建任务
            task = AgentTask(
                task_name=task_config.get('name', 'unknown_task'),
                metrics=metrics,
                datasets=datasets,
                agent=agent,
                max_concurrent=task_config.get('max_concurrent', 10),
                cache_dir=task_config.get('cache_dir', './cache'),
                measure_concurrent=task_config.get('measure_concurrent', 10),
            )
            
            return task
            
        except Exception as e:
            logger.error(f"从配置创建任务失败: {e}")
            raise
    
    def run_parallel(self, tasks: List[AgentTask] = None) -> List[Dict[str, Any]]:
        """
        并行运行多个任务
        
        Args:
            tasks: 要运行的任务列表，如果为None则使用self.tasks
            
        Returns:
            任务执行结果列表
        """
        if tasks is None:
            tasks = self.tasks
            
        if not tasks:
            logger.warning("没有任务需要执行")
            return []
        
        logger.info(f"开始并行执行 {len(tasks)} 个任务，使用 {self.max_workers} 个工作进程")
        
        results = []
        
        # 使用原始配置数据
        task_configs = self.config_data.get('tasks', [])
        
        # 使用线程池执行器
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._run_single_task, task, i): (task, i) 
                for i, task in enumerate(tasks)
            }
            
            # 收集结果
            with tqdm.tqdm(total=len(tasks), desc="执行任务") as pbar:
                for future in as_completed(future_to_task):
                    task_config, task_id = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"任务 {task_config['name']} 执行异常: {e}")
                        error_result = {
                            'task_id': task_id,
                            'task_name': task_config['name'],
                            'status': 'error',
                            'result': None,
                            'start_time': datetime.now().isoformat(),
                            'end_time': datetime.now().isoformat(),
                            'error': str(e)
                        }
                        results.append(error_result)
                        pbar.update(1)
        
        logger.info(f"所有任务执行完成，成功: {sum(1 for r in results if r['status'] == 'success')}, 失败: {sum(1 for r in results if r['status'] != 'success')}")
        return results
    
    def run_sequential(self, tasks: List[AgentTask] = None) -> List[Dict[str, Any]]:
        """
        顺序运行多个任务
        
        Args:
            tasks: 要运行的任务列表，如果为None则使用self.tasks
            
        Returns:
            任务执行结果列表
        """
        if tasks is None:
            tasks = self.tasks
            
        if not tasks:
            logger.warning("没有任务需要执行")
            return []
        
        logger.info(f"开始顺序执行 {len(tasks)} 个任务")
        
        results = []
                
        with tqdm.tqdm(total=len(tasks), desc="执行任务") as pbar:
            for i, task in enumerate(tasks):
                try:
                    result = self._run_single_task(task, i)
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"任务 {task.task_name} 执行异常: {e}")
                    error_result = {
                        'task_id': i,
                        'task_name': task.task_name,
                        'status': 'error',
                        'result': None,
                        'start_time': datetime.now().isoformat(),
                        'end_time': datetime.now().isoformat(),
                        'error': str(e)
                    }
                    results.append(error_result)
                    pbar.update(1)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        保存执行结果到文件
        
        Args:
            results: 执行结果列表
            output_path: 输出文件路径，如果为None则使用默认路径
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def generate_report(self, results: List[Dict[str, Any]], output_dir: str = None):
        '''
        根据评测结果生成总结评测报告
        '''
        from veagentbench.report.extract_metrics_corrected import _extract_metrics_data, process_single_task

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./agent_test_reports_{timestamp}"
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            task_data_dict = _extract_metrics_data(results)
            
            for task_name, task_data in task_data_dict.items(): 
                process_single_task(task_name, task_data, output_dir, False)
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
    
    def run(self, parallel: bool = True, save_results: bool = True, output_path: str = None):
        """
        运行所有任务的主方法
        
        Args:
            parallel: 是否并行执行
            save_results: 是否保存结果
            output_path: 结果输出路径
            
        Returns:
            任务执行结果列表
        """
        try:
            # 加载配置
            if not self.config_data:
                self.load_config()
            
            # 创建任务
            if not self.tasks:
                self.create_tasks_from_config()
            
            # 执行任务
            if parallel:
                results = self.run_parallel()
            else:
                results = self.run_sequential()
            
            # 保存结果
            if save_results:
                self.save_results(results, output_path)
            
            self.generate_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"运行失败: {e}")
            logger.error(traceback.format_exc())
            raise




# 命令行接口
def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent测试运行器')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    parser.add_argument('--parallel', '-p', action='store_true', default=True, help='并行执行')
    parser.add_argument('--sequential', '-s', action='store_true', help='顺序执行')
    parser.add_argument('--output', '-o', help='结果输出路径')
    parser.add_argument('--workers', '-w', type=int, help='工作进程数')
    
    args = parser.parse_args()
    print(args)
    try:
        # 创建运行器
        runner = AgentTestRunner(
            config_path=args.config,
            max_workers=args.workers
        )
        
        # 运行
        results = runner.run(
            parallel=not args.sequential,
            save_results=True,
            output_path=args.output
        )
        
        # 打印统计信息
        success_count = sum(1 for r in results if r['status'] == 'success')
        total_count = len(results)
        
        print(f"\n执行完成!")
        print(f"总任务数: {total_count}")
        print(f"成功: {success_count}")
        print(f"失败: {total_count - success_count}")
        
    except Exception as e:
        logger.error(f"执行失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
