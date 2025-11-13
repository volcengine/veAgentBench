## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import json
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import numpy as np


class PerformanceDataExtractor:
    """从trace数据中提取性能数据的工具类"""
    
    def __init__(self):
        self.trace_data = None
        self.spans = []
        
    def load_trace_data(self, trace_data: Union[str, Dict, List]) -> None:
        """加载trace数据"""
        if isinstance(trace_data, str):
            try:
                self.trace_data = json.loads(trace_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        else:
            self.trace_data = trace_data
            
        # 提取所有spans
        self.spans = self._extract_spans(self.trace_data)
    
    def _extract_spans(self, trace_data: Union[Dict, List]) -> List[Dict[str, Any]]:
        """从trace数据中提取所有spans"""
        spans = []
        
        if isinstance(trace_data, dict):
            # OpenTelemetry格式
            if 'resourceSpans' in trace_data:
                for resource_span in trace_data['resourceSpans']:
                    if 'scopeSpans' in resource_span:
                        for scope_span in resource_span['scopeSpans']:
                            if 'spans' in scope_span:
                                spans.extend(scope_span['spans'])
            elif 'spans' in trace_data:
                spans = trace_data['spans']
            else:
                spans = [trace_data]
            # 支持自定义trace格式（包含trace_data字段）
            if 'trace_data' in trace_data and isinstance(trace_data['trace_data'], list):
                spans = trace_data['trace_data']
        elif isinstance(trace_data, list):
            spans = trace_data
            
        return spans
    
    def extract_end_to_end_duration(self) -> Optional[float]:
        """提取端到端耗时（秒）"""
        if not self.spans:
            return None
            
        # 找到根span（没有parent的span）
        root_spans = [span for span in self.spans if not span.get('parent_span_id')]
        
        if not root_spans:
            # 如果没有明确的根span，使用最早和最晚的时间戳
            start_times = []
            end_times = []
            
            for span in self.spans:
                start_time = span.get('start_time') or span.get('startTimeUnixNano')
                end_time = span.get('end_time') or span.get('endTimeUnixNano')
                
                if start_time and end_time:
                    start_times.append(int(start_time))
                    end_times.append(int(end_time))
            
            if start_times and end_times:
                min_start = min(start_times)
                max_end = max(end_times)
                return (max_end - min_start) / 1e9  # 转换为秒
                
            return None
        
        # 使用根span的持续时间
        root_span = root_spans[0]
        start_time = root_span.get('start_time') or root_span.get('startTimeUnixNano')
        end_time = root_span.get('end_time') or root_span.get('endTimeUnixNano')
        
        if start_time and end_time:
            return (int(end_time) - int(start_time)) / 1e9
            
        return None
    
    def extract_tool_call_durations(self) -> Dict[str, List[float]]:
        """提取工具调用耗时"""
        tool_durations = {}
        
        for span in self.spans:
            span_name = span.get('name', '').lower()
            
            # 识别工具调用相关的span
            if self._is_tool_call_span(span):
                tool_name = self._extract_tool_name(span)
                start_time = span.get('start_time') or span.get('startTimeUnixNano')
                end_time = span.get('end_time') or span.get('endTimeUnixNano')
                
                if start_time and end_time and tool_name:
                    duration = (int(end_time) - int(start_time)) / 1e9  # 转换为秒
                    
                    if tool_name not in tool_durations:
                        tool_durations[tool_name] = []
                    tool_durations[tool_name].append(duration)
        
        return tool_durations
    
    def extract_llm_call_durations(self) -> List[float]:
        """提取LLM调用耗时"""
        llm_durations = []
        
        for span in self.spans:
            span_name = span.get('name', '').lower()
            
            # 识别LLM调用相关的span
            if self._is_llm_call_span(span):
                start_time = span.get('start_time') or span.get('startTimeUnixNano')
                end_time = span.get('end_time') or span.get('endTimeUnixNano')
                
                if start_time and end_time:
                    duration = (int(end_time) - int(start_time)) / 1e9  # 转换为秒
                    llm_durations.append(duration)
        
        return llm_durations
    
    def extract_agent_run_durations(self) -> Dict[str, List[float]]:
        """提取agent执行耗时"""
        agent_durations = {}
        
        for span in self.spans:
            span_name = span.get('name', '')
            
            # 识别agent执行相关的span
            if self._is_agent_run_span(span):
                agent_name = self._extract_agent_name(span)
                start_time = span.get('start_time') or span.get('startTimeUnixNano')
                end_time = span.get('end_time') or span.get('endTimeUnixNano')
                
                if start_time and end_time and agent_name:
                    duration = (int(end_time) - int(start_time)) / 1e9  # 转换为秒
                    
                    if agent_name not in agent_durations:
                        agent_durations[agent_name] = []
                    agent_durations[agent_name].append(duration)
        
        return agent_durations
    
    def extract_performance_summary(self) -> Dict[str, Any]:
        """提取完整的性能数据汇总"""
        summary = {
            'end_to_end_duration': self.extract_end_to_end_duration(),
            'tool_call_stats': {},
            'llm_call_stats': {},
            'agent_run_stats': {},
            'total_tool_calls': 0,
            'total_llm_calls': 0,
            'total_agent_runs': 0,
            'trace_summary': self._get_trace_summary()
        }
        
        # 工具调用统计
        tool_durations = self.extract_tool_call_durations()
        summary['total_tool_calls'] = sum(len(durations) for durations in tool_durations.values())
        
        for tool_name, durations in tool_durations.items():
            if durations:
                summary['tool_call_stats'][tool_name] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': np.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'std_duration': np.std(durations) if len(durations) > 1 else 0
                }
        
        # LLM调用统计（优化：计算真实的LLM耗时，排除工具调用子span）
        llm_durations = self.extract_llm_call_durations()
        summary['total_llm_calls'] = len(llm_durations)
        
        if llm_durations:
            # 计算真实的LLM耗时（排除工具调用子span）
            pure_llm_durations = self._extract_pure_llm_durations()
            
            summary['llm_call_stats'] = {
                'count': len(llm_durations),
                'total_duration': sum(llm_durations),
                'avg_duration': np.mean(llm_durations),
                'min_duration': min(llm_durations),
                'max_duration': max(llm_durations),
                'std_duration': np.std(llm_durations) if len(llm_durations) > 1 else 0,
                # 新增：真实的LLM耗时（排除工具调用）
                'pure_llm_duration': sum(pure_llm_durations) if pure_llm_durations else sum(llm_durations),
                'pure_llm_avg_duration': np.mean(pure_llm_durations) if pure_llm_durations else np.mean(llm_durations)
            }
        
        # Agent执行统计
        agent_durations = self.extract_agent_run_durations()
        summary['total_agent_runs'] = sum(len(durations) for durations in agent_durations.values())
        
        for agent_name, durations in agent_durations.items():
            if durations:
                summary['agent_run_stats'][agent_name] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': np.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'std_duration': np.std(durations) if len(durations) > 1 else 0
                }
        
        return summary
    
    def _is_tool_call_span(self, span: Dict[str, Any]) -> bool:
        """判断span是否为工具调用"""
        span_name = span.get('name', '').lower()
        attributes = span.get('attributes', {})
        
        # 首先检查是否为LLM调用，如果是，则返回False
        if self._is_llm_call_span(span):
            return False
        
        # 忽略合并的工具调用统计节点
        if '(merged tools)' in span_name:
            return False
        
        # 工具调用模式（排除了llm_call等LLM相关模式）
        tool_patterns = [
            'execute_tool', 'function_call', 'execute_command',
            'read_file', 'write_to_file', 'replace_in_file', 'list_files',
            'search_files', 'use_mcp_tool', 'puppeteer_', 'vesearch'
        ]
        
        # 检查span名称
        if any(pattern in span_name for pattern in tool_patterns):
            return True
            
        # 检查attributes中的工具调用标识
        for key, value in attributes.items():
            key_lower = key.lower()
            if 'tool.name' in key_lower or 'gen_ai.tool.name' in key_lower:
                return True
            if 'tool' in key_lower and 'call' in key_lower and 'llm' not in key_lower:
                return True
                
        return False
    
    def _is_llm_call_span(self, span: Dict[str, Any]) -> bool:
        """判断span是否为LLM调用"""
        span_name = span.get('name', '').lower()
        attributes = span.get('attributes', {})
        
        # 首先检查span名称中的精确LLM调用标识
        if 'call_llm' in span_name or 'llm_call' in span_name:
            return True
            
        # 检查attributes中的LLM调用标识（更严格的条件）
        for key, value in attributes.items():
            key_lower = key.lower()
            # 必须有gen_ai.request.model或gen_ai.system且是chat操作
            if 'gen_ai.request.model' in key_lower and 'gen_ai.operation.name' in str(attributes).lower():
                op_name = attributes.get('gen_ai.operation.name', '').lower()
                if op_name in ['chat', 'completion']:
                    return True
            if 'gen_ai.system' in key_lower:
                # 同时需要有请求模型信息才认为是LLM调用
                if any('gen_ai.request.model' in k.lower() for k in attributes.keys()):
                    return True
                
        return False
    
    def _is_agent_run_span(self, span: Dict[str, Any]) -> bool:
        """判断span是否为agent执行"""
        span_name = span.get('name', '').lower()
        
        # 真正的agent执行span应该有特定的名称模式
        # 优先识别方括号格式的agent_run
        if 'agent_run [' in span_name:
            return True
        # 然后是普通agent_run（但不是call_llm等）
        if span_name == 'agent_run' or span_name == 'agent.run':
            return True
            
        # 其他span即使有agent.name属性，也不是agent执行span
        # 它们只是属于某个agent的调用，但不是agent本身的执行
        
        return False
    
    def _extract_agent_name(self, span: Dict[str, Any]) -> Optional[str]:
        """从span中提取agent名称"""
        span_name = span.get('name', '')
        attributes = span.get('attributes', {})
        
        # 优先从标准字段提取
        for key, value in attributes.items():
            key_lower = key.lower()
            if 'agent.name' in key_lower:
                return str(value)
            if 'gen_ai.agent.name' in key_lower:
                return str(value)
        
        # 从span名称推断
        if 'agent_run [' in span_name:
            # 提取方括号中的agent名称
            import re
            match = re.search(r'agent_run \[(.*?)\]', span_name, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 默认使用span名称
        return span_name
    
    def _extract_tool_name(self, span: Dict[str, Any]) -> Optional[str]:
        """从span中提取工具名称"""
        span_name = span.get('name', '')
        
        # 忽略合并的工具调用统计节点
        if '(merged tools)' in span_name.lower() or '(merged)' in span_name.lower():
            return None
            
        attributes = span.get('attributes', {})
        
        # 优先从标准字段提取
        for key, value in attributes.items():
            key_lower = key.lower()
            if 'tool.name' in key_lower:
                return str(value)
            if 'gen_ai.tool.name' in key_lower:
                return str(value)
        
        # 从span名称推断
        if 'execute_tool' in span_name.lower():
            # 尝试从attributes中提取具体的工具名
            for key, value in attributes.items():
                if 'input' in key.lower() or 'params' in key.lower():
                    try:
                        params = json.loads(value) if isinstance(value, str) else value
                        if isinstance(params, dict) and 'name' in params:
                            return params['name']
                    except:
                        pass
        
        return span_name
    
    def _extract_pure_llm_durations(self) -> List[float]:
        """提取真实的LLM调用耗时（排除工具调用子span）"""
        pure_llm_durations = []
        
        for span in self.spans:
            if not self._is_llm_call_span(span):
                continue
                
            # 获取LLM span的总耗时
            start_time = span.get('start_time') or span.get('startTimeUnixNano')
            end_time = span.get('end_time') or span.get('endTimeUnixNano')
            
            if not start_time or not end_time:
                continue
                
            total_llm_duration = (int(end_time) - int(start_time)) / 1e9
            
            # 获取该LLM span的子span（工具调用）
            llm_span_id = span.get('span_id')
            if not llm_span_id:
                continue
                
            # 查找所有子span（通过parent_span_id关联）
            child_tool_spans = []
            for child_span in self.spans:
                if child_span.get('parent_span_id') == llm_span_id and self._is_tool_call_span(child_span):
                    child_tool_spans.append(child_span)
            
            # 计算工具调用的总耗时
            total_tool_duration = 0
            for tool_span in child_tool_spans:
                tool_start = tool_span.get('start_time') or tool_span.get('startTimeUnixNano')
                tool_end = tool_span.get('end_time') or tool_span.get('endTimeUnixNano')
                
                if tool_start and tool_end:
                    tool_duration = (int(tool_end) - int(tool_start)) / 1e9
                    total_tool_duration += tool_duration
            
            # 真实的LLM耗时 = 总耗时 - 工具调用耗时
            pure_llm_duration = total_llm_duration - total_tool_duration
            
            # 确保不为负数
            pure_llm_duration = max(0, pure_llm_duration)
            pure_llm_durations.append(pure_llm_duration)
        
        return pure_llm_durations
    
    def _get_trace_summary(self) -> Dict[str, Any]:
        """获取trace的基本信息汇总"""
        if not self.spans:
            return {}
            
        summary = {
            'total_spans': len(self.spans),
            'span_types': {},
            'time_range': None
        }
        
        # 统计span类型
        for span in self.spans:
            span_name = span.get('name', 'unknown')
            if span_name not in summary['span_types']:
                summary['span_types'][span_name] = 0
            summary['span_types'][span_name] += 1
        
        # 时间范围
        start_times = []
        end_times = []
        
        for span in self.spans:
            start_time = span.get('start_time') or span.get('startTimeUnixNano')
            end_time = span.get('end_time') or span.get('endTimeUnixNano')
            
            if start_time:
                start_times.append(int(start_time))
            if end_time:
                end_times.append(int(end_time))
        
        if start_times and end_times:
            summary['time_range'] = {
                'start': min(start_times),
                'end': max(end_times),
                'duration': (max(end_times) - min(start_times)) / 1e9
            }
        
        return summary


def extract_performance_data_from_trace_file(file_path: str) -> Dict[str, Any]:
    """从trace文件中提取性能数据"""
    extractor = PerformanceDataExtractor()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        
        extractor.load_trace_data(trace_data)
        return extractor.extract_performance_summary()
        
    except Exception as e:
        return {
            'error': f"Failed to extract performance data: {str(e)}",
            'end_to_end_duration': None,
            'tool_call_stats': {},
            'llm_call_stats': {},
            'agent_run_stats': {},
            'total_tool_calls': 0,
            'total_llm_calls': 0,
            'total_agent_runs': 0,
            'trace_summary': {}
        }


def extract_performance_data_from_trace(trace_data: Union[str, Dict, List]) -> Dict[str, Any]:
    """从trace数据中提取性能数据"""
    extractor = PerformanceDataExtractor()
    extractor.load_trace_data(trace_data)
    return extractor.extract_performance_summary()


# 使用示例
if __name__ == "__main__":
    # 示例：从文件提取
    # file_path = ".cache/veagentbench/金融分析场景__financial_deep_research__1c23368b85a244eb0799271e.json"
    # performance_data = extract_performance_data_from_trace_file(file_path)
    # print(json.dumps(performance_data, indent=2, ensure_ascii=False))
    
    # 示例：从内存数据提取
    # trace_data = {...}  # 你的trace数据
    # performance_data = extract_performance_data_from_trace(trace_data)
    # print(json.dumps(performance_data, indent=2, ensure_ascii=False))
    pass
