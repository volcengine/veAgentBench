#!/usr/bin/env python3
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
"""
VeAgentBench CLI - 统一的命令行工具

集成了配置生成和任务执行功能
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from veagentbench.runner.run import AgentTestRunner
from veagentbench.utils.config_generator import ConfigGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VeAgentBenchCLI:
    """VeAgentBench统一CLI工具"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.config_generator = ConfigGenerator()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建主参数解析器"""
        parser = argparse.ArgumentParser(
            description='VeAgentBench - Agent评测工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  # 生成配置模板
  veagentbench config generate --task-name my_test
  
  # 运行评测任务
  veagentbench run --config my_config.yaml --parallel
  
  # 顺序执行任务
  veagentbench run --config my_config.yaml --sequential
            """
        )
        
        # 创建子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # config子命令
        config_parser = subparsers.add_parser('config', help='配置相关操作')
        config_subparsers = config_parser.add_subparsers(dest='config_command')
        
        # config generate子命令
        generate_parser = config_subparsers.add_parser('generate', help='生成配置模板')
        self._add_generate_args(generate_parser)
        
        # config validate子命令
        validate_parser = config_subparsers.add_parser('validate', help='验证配置文件')
        validate_parser.add_argument('--config', '-c', required=True, help='配置文件路径')
        
        # run子命令
        run_parser = subparsers.add_parser('run', help='运行评测任务')
        self._add_run_args(run_parser)
        
        # info子命令
        info_parser = subparsers.add_parser('info', help='显示系统信息')
        info_parser.add_argument('--metrics', action='store_true', help='显示可用指标')
        info_parser.add_argument('--agents', action='store_true', help='显示可用代理')
        info_parser.add_argument('--templates', action='store_true', help='显示配置模板类型')
        
        return parser
    
    def _add_generate_args(self, parser: argparse.ArgumentParser):
        """添加配置生成参数"""
        parser.add_argument('--output', '-o', 
                           default='config_template.yaml', 
                           help='输出文件路径')
        parser.add_argument('--task-name', 
                           default='test_task', 
                           help='任务名称')
        parser.add_argument('--dataset-path', 
                           help='数据集文件路径')
        
        # 模型相关参数
        parser.add_argument('--judge-model', 
                           default='gpt-4', 
                           help='评判模型名称')
        parser.add_argument('--judge-base-url', 
                           help='评判模型API基础URL')
        parser.add_argument('--judge-api-key', 
                           help='评判模型API密钥')
        
        # 代理相关参数
        parser.add_argument('--agent-class', 
                           choices=['AdkAgent', 'SimpleAgent'], 
                           default='AdkAgent', 
                           help='代理类名')
        parser.add_argument('--agent-endpoint', 
                           help='代理API端点 (仅AdkAgent)')
        parser.add_argument('--api-key', 
                           help='代理API密钥')
        
        # 执行参数
        parser.add_argument('--max-concurrent', 
                           type=int, 
                           default=5, 
                           help='最大并发数')
        parser.add_argument('--cache-dir', 
                           default='./cache', 
                           help='缓存目录')
        
        # 多任务配置
        parser.add_argument('--tasks-config', 
                           help='多任务配置文件路径 (JSON格式)')
    
    def _add_run_args(self, parser: argparse.ArgumentParser):
        """添加运行参数"""
        parser.add_argument('--config', '-c', 
                           required=True, 
                           help='配置文件路径')
        parser.add_argument('--parallel', '-p', 
                           action='store_true', 
                           default=True, 
                           help='并行执行')
        parser.add_argument('--sequential', '-s', 
                           action='store_true', 
                           help='顺序执行')
        parser.add_argument('--output', '-o', 
                           help='结果输出路径')
        parser.add_argument('--workers', '-w', 
                           type=int, 
                           help='工作进程数')
        parser.add_argument('--dry-run', 
                           action='store_true', 
                           help='干运行模式，只验证配置不执行任务')
    
    def handle_config_generate(self, args) -> int:
        """处理配置生成命令"""
        try:
            # 构建参数
            kwargs = {
                "task_name": args.task_name,
                "max_concurrent": args.max_concurrent,
                "cache_dir": args.cache_dir,
                "agent_class": args.agent_class,
                "judge_model": args.judge_model
            }
            
            # 添加可选参数
            if args.dataset_path:
                kwargs["dataset_path"] = args.dataset_path
            if args.judge_base_url:
                kwargs["judge_base_url"] = args.judge_base_url
            if args.judge_api_key:
                kwargs["judge_api_key"] = args.judge_api_key
            if args.api_key:
                kwargs["api_key"] = args.api_key
            if args.agent_endpoint:
                kwargs["agent_endpoint"] = args.agent_endpoint
                
            # 处理多任务配置
            if args.tasks_config:
                with open(args.tasks_config, 'r', encoding='utf-8') as f:
                    tasks_config = json.load(f)
                kwargs["tasks"] = tasks_config
            
            # 生成并保存配置（使用默认的basic模板）
            config = self.config_generator.generate_and_save("basic", args.output, **kwargs)
            
            # 打印生成的配置摘要
            print(f"\n配置生成完成!")
            print(f"输出文件: {args.output}")
            print(f"任务数量: {len(config.get('tasks', []))}")
            
            for i, task in enumerate(config.get('tasks', [])):
                print(f"  任务 {i+1}: {task['name']}")
                print(f"    数据集: {len(task['datasets'])} 个")
                print(f"    指标: {', '.join(task['metrics'])}")
                print(f"    代理: {task['agent']['type']}")
                
            print(f"\n配置文件已生成，请根据需要进行修改。")
            return 0
            
        except Exception as e:
            logger.error(f"生成配置文件失败: {e}")
            return 1
    
    def handle_config_validate(self, args) -> int:
        """处理配置验证命令"""
        try:
            logger.info(f"正在验证配置文件: {args.config}")
            
            # 尝试加载配置
            runner = AgentTestRunner(config_path=args.config)
            config = runner.load_config()
            
            # 验证配置结构
            if 'tasks' not in config:
                logger.error("配置文件中缺少 'tasks' 字段")
                return 1
            
            tasks = config.get('tasks', [])
            if not tasks:
                logger.error("配置文件中没有任务配置")
                return 1
            
            # 验证每个任务
            for i, task in enumerate(tasks):
                task_name = task.get('name', f'task_{i}')
                logger.info(f"验证任务 {i+1}: {task_name}")
                
                # 检查必需字段
                required_fields = ['datasets', 'metrics', 'agent']
                for field in required_fields:
                    if field not in task:
                        logger.error(f"任务 {task_name} 缺少必需字段: {field}")
                        return 1
                
                # 检查数据集
                datasets = task.get('datasets', [])
                if not datasets:
                    logger.error(f"任务 {task_name} 没有配置数据集")
                    return 1
                
                # 检查代理配置
                agent_config = task.get('agent', {})
                if 'type' not in agent_config:
                    logger.error(f"任务 {task_name} 的代理配置缺少 'class' 字段")
                    return 1
            
            logger.info("配置文件验证通过!")
            return 0
            
        except Exception as e:
            logger.error(f"配置文件验证失败: {e}")
            return 1
    
    def handle_run(self, args) -> int:
        """处理运行命令"""
        try:
            logger.info(f"正在运行配置文件: {args.config}")
            
            # 创建运行器
            runner = AgentTestRunner(
                config_path=args.config,
                max_workers=args.workers
            )
            
            # 干运行模式
            if args.dry_run:
                logger.info("干运行模式 - 只验证配置")
                try:
                    runner.load_config()
                    runner.create_tasks_from_config()
                    logger.info("配置验证通过!")
                    return 0
                except Exception as e:
                    logger.error(f"配置验证失败: {e}")
                    return 1
            
            # 运行任务
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
            
            return 0 if success_count == total_count else 1
            
        except Exception as e:
            logger.error(f"运行失败: {e}")
            return 1
    
    def handle_info(self, args) -> int:
        """处理信息查询命令"""
        try:
            if args.metrics:
                print("\n可用指标:")
                from veagentbench import metrics as metrics_module
                available_metrics = [attr for attr in dir(metrics_module) 
                                   if not attr.startswith('_') and attr[0].isupper()]
                for metric in available_metrics:
                    print(f"  - {metric}")
            
            if args.agents:
                print("\n可用代理:")
                from veagentbench import agents as agents_module
                available_agents = [attr for attr in dir(agents_module) 
                                   if not attr.startswith('_') and attr.endswith('Agent')]
                for agent in available_agents:
                    print(f"  - {agent}")
            
            if args.templates:
                print("\n配置模板类型:")
                templates = {
                    "basic": "基础测试配置",
                    "multi": "多任务测试配置"
                }
                for template_type, description in templates.items():
                    print(f"  - {template_type}: {description}")
            
            if not any([args.metrics, args.agents, args.templates]):
                # 显示基本信息
                print("\nVeAgentBench - Agent评测工具")
                print("版本: 1.0.0")
                print("\n使用 'veagentbench --help' 查看可用命令")
                print("使用 'veagentbench info --metrics' 查看可用指标")
                print("使用 'veagentbench info --agents' 查看可用代理")
                print("使用 'veagentbench info --templates' 查看配置模板")
            
            return 0
            
        except Exception as e:
            logger.error(f"获取信息失败: {e}")
            return 1
    
    def run(self, argv=None) -> int:
        """运行CLI"""
        args = self.parser.parse_args(argv)
        
        if not args.command:
            self.parser.print_help()
            return 0
        
        try:
            if args.command == 'config':
                if args.config_command == 'generate':
                    return self.handle_config_generate(args)
                elif args.config_command == 'validate':
                    return self.handle_config_validate(args)
                else:
                    print("错误: 请指定config子命令 (generate 或 validate)")
                    return 1
            
            elif args.command == 'run':
                return self.handle_run(args)
            
            elif args.command == 'info':
                return self.handle_info(args)
            
            else:
                print(f"错误: 未知命令 '{args.command}'")
                return 1
                
        except KeyboardInterrupt:
            logger.info("用户中断执行")
            return 130
        except Exception as e:
            logger.error(f"执行失败: {e}")
            return 1


def main():
    """CLI入口函数"""
    cli = VeAgentBenchCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
