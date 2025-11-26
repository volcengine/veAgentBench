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
配置文件生成器 - 自动生成AgentTestRunner的YAML配置文件模板
"""

import yaml
import argparse
import os
from typing import Dict, Any, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigGenerator:
    """配置文件生成器"""
    
    def __init__(self):
        self.default_metrics = [
            "AnswerCorrectnessMetric",
            "AnswerRelevancyMetric", 
            "ContextualPrecisionMetric",
            "ContextualRecallMetric",
            "FaithfulnessMetric",
            "ContextualRelevancyMetric"
        ]
        
        self.default_agents = {
            "AdkAgent": {
                "end_point": "http://127.0.0.1:8000/invoke",
                "api_key": "your_api_key_here"
            },
            "SimpleAgent": {
                "model_name": "gpt-4",
                "api_key": "your_openai_api_key_here"
            }
        }
        
        self.dataset_templates = {
            "csv": {
                "type": "csv",
                "csv_file_path": "path/to/your/dataset.csv",
                "input_column": "input",
                "expected_output_column": "expect_output",
                "expected_tool_call_column": "expected_tool_calls"
            },
            "json": {
                "type": "json",
                "json_file_path": "path/to/your/dataset.json",
                "input_key": "input",
                "expected_key": "expected_output",
                "expected_tool_call_key": "expected_tool_calls"
            }
        }
    
    def generate_basic_config(self, task_name: str = "basic_test", **kwargs) -> Dict[str, Any]:
        """生成基础配置"""
        config = {
            "tasks": [
                {
                    "name": task_name,
                    "datasets": [
                        {
                            "name": f"{task_name}_dataset",
                            "description": f"{task_name}测试数据集",
                            "property": self.dataset_templates["csv"].copy()
                        }
                    ],
                    "metrics": kwargs.get("metrics", self.default_metrics),
                    "judge_model": {
                        "model_name": kwargs.get("judge_model", "gpt-4"),
                        "base_url": kwargs.get("judge_base_url", "https://api.openai.com/v1"),
                        "api_key": kwargs.get("judge_api_key", "your_judge_api_key_here")
                    },
                    "agent": {
                        "type": kwargs.get("agent_class", "AdkAgent"),
                        "property": self.default_agents[kwargs.get("agent_class", "AdkAgent")].copy()
                    },
                    "max_concurrent": kwargs.get("max_concurrent", 5),
                    "cache_dir": kwargs.get("cache_dir", "./cache")
                }
            ]
        }
        
        # 更新数据集路径
        if "dataset_path" in kwargs:
            config["tasks"][0]["datasets"][0]["property"]["csv_file_path"] = kwargs["dataset_path"]
        
        # 更新API密钥
        if "api_key" in kwargs:
            agent_class = kwargs.get("agent_class", "AdkAgent")
            if agent_class == "AdkAgent":
                config["tasks"][0]["agent"]["property"]["api_key"] = kwargs["api_key"]
            elif agent_class == "SimpleAgent":
                config["tasks"][0]["agent"]["property"]["api_key"] = kwargs["api_key"]
        
        if "judge_api_key" in kwargs:
            config["tasks"][0]["judge_model"]["api_key"] = kwargs["judge_api_key"]
        
        return config
    
    def generate_mcp_config(self, task_name: str = "mcp_test", **kwargs) -> Dict[str, Any]:
        """生成MCP工具测试配置"""
        config = self.generate_basic_config(task_name, **kwargs)
        
        # 专门针对MCP测试的配置
        config["tasks"][0]["metrics"] = ["MCPToolMetric"]
        config["tasks"][0]["datasets"][0]["description"] = "MCP工具调用测试数据集"
        
        # 设置MCP相关的默认路径
        if "mcp_dataset_path" in kwargs:
            config["tasks"][0]["datasets"][0]["property"]["csv_file_path"] = kwargs["mcp_dataset_path"]
        else:
            config["tasks"][0]["datasets"][0]["property"]["csv_file_path"] = "example_dataset/mcptask/testcase.csv"
            
        return config
    
    def generate_rag_config(self, task_name: str = "rag_test", **kwargs) -> Dict[str, Any]:
        """生成RAG测试配置"""
        config = self.generate_basic_config(task_name, **kwargs)
        
        # RAG测试常用的指标
        config["tasks"][0]["metrics"] = [
            "AnswerCorrectness",
            "AnswerRelevancy",
            "ContextualPrecision",
            "ContextualRecall",
            "Faithfulness"
        ]
        config["tasks"][0]["datasets"][0]["description"] = "RAG问答测试数据集"
        
        if "rag_dataset_path" in kwargs:
            config["tasks"][0]["datasets"][0]["property"]["csv_file_path"] = kwargs["rag_dataset_path"]
            
        return config
    
    def generate_multi_task_config(self, task_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成多任务配置"""
        config = {"tasks": []}
        
        for i, task_config in enumerate(task_configs):
            task_name = task_config.get("name", f"task_{i+1}")
            base_config = self.generate_basic_config(task_name, **task_config)
            config["tasks"].append(base_config["tasks"][0])
            
        return config
    
    def generate_template(self, template_type: str = "basic", **kwargs) -> Dict[str, Any]:
        """根据模板类型生成配置 - 现在只支持基本配置"""
        if template_type == "basic":
            return self.generate_basic_config(**kwargs)
        elif template_type == "multi":
            return self.generate_multi_task_config(kwargs.get("tasks", []))
        else:
            # 其他类型都使用基本配置
            logger.info(f"模板类型 '{template_type}' 已简化，使用基本配置")
            return self.generate_basic_config(**kwargs)
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """保存配置到文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存配置文件
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"配置文件已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def generate_and_save(self, template_type: str = "basic", output_path: str = "config_template.yaml", **kwargs):
        """生成并保存配置模板"""
        config = self.generate_template(template_type, **kwargs)
        self.save_config(config, output_path)
        return config


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='AgentTestRunner配置文件生成器')
    
    # 基本参数
    parser.add_argument('--type', '-t', choices=['basic', 'mcp', 'rag', 'multi'], 
                       default='basic', help='配置模板类型')
    parser.add_argument('--output', '-o', default='config_template.yaml', 
                       help='输出文件路径')
    parser.add_argument('--task-name', default='test_task', 
                       help='任务名称')
    parser.add_argument('--dataset-path', 
                       help='数据集文件路径')
    
    # 模型相关参数
    parser.add_argument('--judge-model', default='gpt-4', 
                       help='评判模型名称')
    parser.add_argument('--judge-base-url', 
                       help='评判模型API基础URL')
    parser.add_argument('--judge-api-key', 
                       help='评判模型API密钥')
    
    # 代理相关参数
    parser.add_argument('--agent-class', choices=['AdkAgent', 'SimpleAgent'], 
                       default='AdkAgent', help='代理类名')
    parser.add_argument('--agent-endpoint', 
                       help='代理API端点 (仅AdkAgent)')
    parser.add_argument('--api-key', 
                       help='代理API密钥')
    
    # 执行参数
    parser.add_argument('--max-concurrent', type=int, default=5, 
                       help='最大并发数')
    parser.add_argument('--cache-dir', default='./cache', 
                       help='缓存目录')
    
    # 多任务配置
    parser.add_argument('--tasks-config', 
                       help='多任务配置文件路径 (JSON格式)')
    
    # 特殊模板参数
    parser.add_argument('--mcp-dataset-path', 
                       help='MCP数据集路径')
    parser.add_argument('--rag-dataset-path', 
                       help='RAG数据集路径')
    
    args = parser.parse_args()
    
    try:
        generator = ConfigGenerator()
        
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
        if args.mcp_dataset_path:
            kwargs["mcp_dataset_path"] = args.mcp_dataset_path
        if args.rag_dataset_path:
            kwargs["rag_dataset_path"] = args.rag_dataset_path
            
        # 处理多任务配置
        if args.type == "multi" and args.tasks_config:
            import json
            with open(args.tasks_config, 'r', encoding='utf-8') as f:
                tasks_config = json.load(f)
            kwargs["tasks"] = tasks_config
        
        # 生成并保存配置
        config = generator.generate_and_save(args.type, args.output, **kwargs)
        
        # 打印生成的配置摘要
        print(f"\n配置生成完成!")
        print(f"模板类型: {args.type}")
        print(f"输出文件: {args.output}")
        print(f"任务数量: {len(config.get('tasks', []))}")
        
        for i, task in enumerate(config.get('tasks', [])):
            print(f"  任务 {i+1}: {task['name']}")
            print(f"    数据集: {len(task['datasets'])} 个")
            print(f"    指标: {', '.join(task['metrics'])}")
            print(f"    代理: {task['agent']['class']}")
            
        print(f"\n配置文件已生成，请根据需要进行修改。")
        
    except Exception as e:
        logger.error(f"生成配置文件失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
