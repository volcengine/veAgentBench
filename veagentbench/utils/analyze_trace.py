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

import json
from typing import Dict, List, Any, Union


def extract_tool_calls_from_trace(trace: Union[Dict, List]) -> List[Dict[str, Any]]:
    """
    从trace数据中提取agent工具调用信息
    
    Args:
        trace: trace数据，可能是字典或列表格式，支持OpenTelemetry格式
        
    Returns:
        list: 包含工具调用信息的列表，每个元素包含：
            - tool_name: 工具名称
            - input_params: 输入参数、
            - discription: 描述
            - available_functions: 可用的函数列表（如果有）
            - output_result: 输出结果
            - timestamp: 调用时间戳（如果有）
            - span_id: span标识符（如果有）
            - trace_id: trace标识符（如果有）
            - duration: 执行时长（如果有）
    """
    tool_calls = []
    
    # 处理不同的trace数据格式
    spans = []
    if isinstance(trace, dict):
        # 如果是OpenTelemetry格式，可能有resourceSpans结构
        if 'resourceSpans' in trace:
            for resource_span in trace['resourceSpans']:
                if 'scopeSpans' in resource_span:
                    for scope_span in resource_span['scopeSpans']:
                        if 'spans' in scope_span:
                            spans.extend(scope_span['spans'])
        elif 'spans' in trace:
            spans = trace['spans']
        else:
            # 如果trace本身就是一个span
            spans = [trace]
    elif isinstance(trace, list):
        spans = trace
    
    # 首先提取所有可用的functions信息
    functions_map = _extract_functions_from_spans(spans)
    available_functions = []
    # 分析每个span
    for span in spans:
        if not isinstance(span, dict):
            continue
            
        # 检查是否是工具调用相关的span
        span_name = span.get('name', '')
        attributes = span.get('attributes', {})
        events = span.get('events', [])
        
        # 初始化工具调用信息
        tool_info = {
            'tool_name': None,
            'input_params': None,
            'output_result': None,
            'timestamp': span.get('startTimeUnixNano') or span.get('start_time'),
            'span_id': span.get('spanId') or span.get('span_id'),
            'trace_id': span.get('traceId') or span.get('trace_id'),
            'duration': None,
            'available_functions': None  # 新增：可用的函数列表
        }
        
        # 计算执行时长
        start_time = span.get('startTimeUnixNano') or span.get('start_time')
        end_time = span.get('endTimeUnixNano') or span.get('end_time')
        if start_time and end_time:
            tool_info['duration'] = int(end_time) - int(start_time)
        
        # 精确识别实际的工具调用span
        is_tool_call = _is_actual_tool_call_span(span, span_name, attributes, events)
        
        if is_tool_call:
            # 提取工具调用的详细信息
            _extract_tool_call_details(span, span_name, attributes, events, tool_info)
            
            # 查找对应的functions列表
            trace_id = tool_info['trace_id']
            span_id = tool_info['span_id']
            
            # 优先使用trace_id匹配
            if trace_id and trace_id in functions_map:
                available_functions = functions_map[trace_id]
            # 如果没有找到，尝试使用span_id或者查找最近的functions
            elif not available_functions:
                # 查找同一trace中的任何functions
                for tid, funcs in functions_map.items():
                    if funcs:  # 如果有functions，就使用（假设同一个trace中的functions是相同的）
                        available_functions = funcs
                        break
            
            # 只有当成功提取到工具名称时才添加
            if (tool_info['tool_name']) and (tool_info['tool_name'] != '(merged tools)'):
                tool_calls.append(tool_info)
    
    # 去重处理
    deduplicated_calls = _deduplicate_tool_calls(tool_calls)
    
    # 按执行时间排序并添加执行顺序index
    return _add_execution_order(deduplicated_calls), available_functions


def _extract_functions_from_spans(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    从spans中提取LLM请求时的functions列表
    
    Args:
        spans: 所有span的列表
        
    Returns:
        dict: trace_id -> functions列表的映射
    """
    functions_map = {}
    
    for span in spans:
        if not isinstance(span, dict):
            continue
            
        span_name = span.get('name', '').lower()
        attributes = span.get('attributes', {})
        events = span.get('events', [])
        trace_id = span.get('traceId') or span.get('trace_id')
        
        # 识别LLM调用相关的span或包含functions的span
        llm_indicators = [
            'llm', 'chat', 'completion', 'openai', 'anthropic', 'claude',
            'generate', 'model', 'inference', 'ai_model', 'assistant', 'agent'
        ]
        
        is_llm_span = any(indicator in span_name for indicator in llm_indicators)
        
        # 也检查attributes中是否有LLM相关标识
        if not is_llm_span:
            for key, value in attributes.items():
                key_lower = key.lower()
                value_str = str(value).lower()
                if any(indicator in key_lower or indicator in value_str for indicator in llm_indicators):
                    is_llm_span = True
                    break
        
        # 即使不是LLM span，如果包含functions相关信息也要处理
        has_functions_info = False
        for key, value in attributes.items():
            key_lower = key.lower()
            if any(func_key in key_lower for func_key in ['function', 'tool', 'schema']):
                has_functions_info = True
                break
        
        # 检查events中是否有functions信息
        if not has_functions_info:
            for event in events:
                event_name = event.get('name', '').lower()
                event_attrs = event.get('attributes', {})
                if any(func_key in event_name for func_key in ['function', 'tool', 'schema']):
                    has_functions_info = True
                    break
                for attr_key in event_attrs.keys():
                    if any(func_key in attr_key.lower() for func_key in ['function', 'tool', 'schema']):
                        has_functions_info = True
                        break
                if has_functions_info:
                    break
        
        if (is_llm_span or has_functions_info) and trace_id:
            functions = None
            
            # 1. 优先从gen_ai.request.functions.*字段中提取（这是标准的OpenTelemetry格式）
            functions_dict = {}
            for key, value in attributes.items():
                key_lower = key.lower()
                
                # 检查是否是gen_ai.request.functions.N.xxx格式
                if key_lower.startswith('gen_ai.request.functions.'):
                    parts = key.split('.')
                    if len(parts) >= 4:
                        try:
                            func_index = int(parts[3])  # 获取函数索引
                            func_field = parts[4] if len(parts) > 4 else None  # 获取字段名
                            
                            if func_index not in functions_dict:
                                functions_dict[func_index] = {}
                            
                            if func_field:
                                functions_dict[func_index][func_field] = value
                        except (ValueError, IndexError):
                            continue
            
            # 如果找到了functions字段，构建functions列表
            if functions_dict:
                functions_list = []
                for i in sorted(functions_dict.keys()):
                    func_data = functions_dict[i]
                    if 'name' in func_data:  # 至少要有name字段
                        functions_list.append(func_data)
                
                if functions_list:
                    functions = functions_list
            
            # 2. 如果没有找到，使用原来的宽泛匹配方法
            if not functions:
                for key, value in attributes.items():
                    key_lower = key.lower()
                    value_str = str(value)
                    
                    # 检查key中是否包含functions相关关键词
                    function_keywords = ['function', 'tool', 'schema', 'available', 'definition']
                    if any(keyword in key_lower for keyword in function_keywords):
                        parsed_value = _parse_json_if_string(value)
                        if isinstance(parsed_value, (list, dict)):
                            # 进一步验证是否真的是functions定义
                            if _is_functions_definition(parsed_value):
                                functions = parsed_value
                                break
                    
                    # 检查value中是否包含functions定义的特征
                    if len(value_str) > 50 and ('"name"' in value_str or '"function"' in value_str):
                        parsed_value = _parse_json_if_string(value)
                        if _is_functions_definition(parsed_value):
                            functions = parsed_value
                            break
            
            # 2. 从events中查找functions（更全面的搜索）
            if not functions:
                for event in events:
                    event_name = event.get('name', '').lower()
                    event_attrs = event.get('attributes', {})
                    
                    # 检查所有event，不仅仅是request/input
                    for attr_key, attr_value in event_attrs.items():
                        attr_key_lower = attr_key.lower()
                        
                        # 更宽泛的关键词匹配
                        if any(keyword in attr_key_lower for keyword in ['function', 'tool', 'schema', 'available']):
                            parsed_value = _parse_json_if_string(attr_value)
                            if _is_functions_definition(parsed_value):
                                functions = parsed_value
                                break
                        
                        # 检查较长的字符串值是否包含functions定义
                        if isinstance(attr_value, str) and len(attr_value) > 100:
                            if '"name"' in attr_value and ('"function"' in attr_value or '"parameters"' in attr_value):
                                parsed_value = _parse_json_if_string(attr_value)
                                if _is_functions_definition(parsed_value):
                                    functions = parsed_value
                                    break
                    
                    if functions:
                        break
            
            # 3. 优先从请求体的functions字段中提取（这里包含完整的参数定义）
            if not functions:
                # 搜索所有可能包含请求数据的字段
                request_candidates = []
                for key, value in attributes.items():
                    key_lower = key.lower()
                    if any(req_key in key_lower for req_key in ['request', 'body', 'payload', 'input', 'message']):
                        request_candidates.append(value)
                
                for candidate in request_candidates:
                    request_data = _parse_json_if_string(candidate)
                    if isinstance(request_data, dict):
                        # 优先检查request.functions字段（这里通常包含完整的参数定义）
                        if 'functions' in request_data:
                            potential_functions = request_data['functions']
                            if _is_functions_definition(potential_functions):
                                functions = potential_functions
                                break
                        
                        # 然后检查其他常见的functions字段
                        for func_key in ['tools', 'function_call', 'tool_choice', 'available_functions']:
                            if func_key in request_data:
                                potential_functions = request_data[func_key]
                                if _is_functions_definition(potential_functions):
                                    functions = potential_functions
                                    break
                        
                        # 递归搜索嵌套的字段
                        if not functions:
                            functions = _search_nested_functions(request_data)
                        
                        if functions:
                            break
            
            # 如果找到了functions，解析并添加到映射中
            if functions:
                parsed_functions = _parse_functions_details(functions)
                if parsed_functions:
                    functions_map[trace_id] = parsed_functions
    
    return functions_map


def _is_actual_tool_call_span(span: Dict[str, Any], span_name: str, attributes: Dict[str, Any], events: List[Dict[str, Any]]) -> bool:
    """
    精确判断span是否为实际的工具调用
    
    Args:
        span: span数据
        span_name: span名称
        attributes: span属性
        events: span事件列表
        
    Returns:
        bool: 是否为工具调用span
    """
    # 1. 检查span名称是否明确表示工具调用
    tool_call_patterns = [
        # 'execute_command',
        # 'read_file', 
        # 'write_to_file',
        # 'replace_in_file',
        # 'grep',
        # 'list_files',
        # 'glob',
        # 'use_mcp_tool',
        # 'ask_followup_question',
        # 'attempt_completion',
        # 'tool_call',
        # 'function_call',
        'execute_tool'
    ]
    
    span_name_lower = span_name.lower()
    if any(pattern in span_name_lower for pattern in tool_call_patterns):
        return True
    
    # # 2. 检查attributes中是否有明确的工具调用标识
    # for key, value in attributes.items():
    #     key_lower = key.lower()
    #     if key_lower in ['tool_name', 'function_name', 'tool_type']:
    #         return True
    #     if 'tool' in key_lower and 'call' in key_lower:
    #         return True
    
    # # 3. 检查events中是否有工具调用相关事件
    # for event in events:
    #     event_name = event.get('name', '').lower()
    #     if any(pattern in event_name for pattern in ['tool_start', 'tool_end', 'function_start', 'function_end']):
    #         return True
    
    # # 4. 检查span的kind是否表示客户端调用（通常工具调用是客户端调用）
    # span_kind = span.get('kind') or attributes.get('span.kind')
    # if span_kind in ['CLIENT', 'PRODUCER', 3]:  # 3 是 CLIENT 的数值表示
    #     # 进一步检查是否有工具相关的属性
    #     if any('tool' in str(v).lower() for v in attributes.values()):
    #         return True
    
    return False


def _extract_tool_call_details(span: Dict[str, Any], span_name: str, attributes: Dict[str, Any], events: List[Dict[str, Any]], tool_info: Dict[str, Any]) -> None:
    """
    从span中提取工具调用的详细信息
    
    Args:
        span: span数据
        span_name: span名称
        attributes: span属性
        events: span事件列表
        tool_info: 要填充的工具信息字典
    """
    # 1. 提取工具名称
    if not tool_info['tool_name']:
        # 优先从attributes中获取
        for key, value in attributes.items():
            key_lower = key.lower()
            if 'tool.name' in key_lower :
                tool_info['tool_name'] = str(value)
                break
        
        # 如果attributes中没有，使用span名称
        if not tool_info['tool_name']:
            tool_info['tool_name'] = span_name
    
    # 2. 提取输入参数
    input_sources = ['input', 'params', 'arguments', 'args', 'request', 'command']
    for key, value in attributes.items():
        key_lower = key.lower()
        if ('tool' in key_lower) and any(source in key_lower for source in input_sources):
            print(value)
            tool_info['input_params'] = _parse_json_if_string(value).get('parameters', {}) if isinstance(_parse_json_if_string(value), dict) else _parse_json_if_string(value)
            break
    
    # 3. 提取输出结果
    output_sources = ['output', 'result', 'response', 'return', 'stdout']
    for key, value in attributes.items():
        key_lower = key.lower()
        if ('tool' in key_lower) and any(source in key_lower for source in output_sources):
            tool_info['output_result'] = _parse_json_if_string(value)['response'].get('result', {}) if isinstance(_parse_json_if_string(value), dict) else _parse_json_if_string(value)
            break
    
    # 4. 从events中提取更详细的信息
    for event in events:
        event_name = event.get('name', '').lower()
        event_attrs = event.get('attributes', {})
        
        # 工具开始事件
        if 'start' in event_name or 'begin' in event_name:
            for attr_key, attr_value in event_attrs.items():
                attr_key_lower = attr_key.lower()
                if any(source in attr_key_lower for source in input_sources) and not tool_info['input_params']:
                    tool_info['input_params'] = _parse_json_if_string(attr_value)
        
        # 工具结束事件
        elif 'end' in event_name or 'complete' in event_name or 'finish' in event_name:
            for attr_key, attr_value in event_attrs.items():
                attr_key_lower = attr_key.lower()
                if any(source in attr_key_lower for source in output_sources) and not tool_info['output_result']:
                    tool_info['output_result'] = _parse_json_if_string(attr_value)
    
    # 5. 清理工具名称（移除常见的前缀后缀）
    if tool_info['tool_name']:
        tool_name = tool_info['tool_name']
        # 移除常见的前缀
        prefixes_to_remove = ['span:', 'trace:', 'operation:', 'call:', 'execute:']
        for prefix in prefixes_to_remove:
            if tool_name.lower().startswith(prefix):
                tool_name = tool_name[len(prefix):]
                break
        tool_info['tool_name'] = tool_name.strip()


def _deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对工具调用进行去重处理
    
    Args:
        tool_calls: 原始工具调用列表
        
    Returns:
        list: 去重后的工具调用列表
    """
    if not tool_calls:
        return tool_calls
    
    seen = set()
    deduplicated = []
    
    for call in tool_calls:
        # 创建唯一标识符，基于关键信息
        tool_name = call.get('tool_name', '')
        span_id = call.get('span_id', '')
        timestamp = call.get('timestamp', '')
        
        # 如果有span_id，优先使用span_id作为唯一标识
        if span_id:
            unique_key = f"{tool_name}_{span_id}"
        else:
            # 如果没有span_id，使用工具名称、时间戳和输入参数的组合
            input_str = str(call.get('input_params', ''))[:100]  # 限制长度避免过长
            unique_key = f"{tool_name}_{timestamp}_{hash(input_str)}"
        
        if unique_key not in seen:
            seen.add(unique_key)
            deduplicated.append(call)
    
    return deduplicated


def _parse_functions_details(functions_data: Any) -> List[Dict[str, Any]]:
    """
    解析functions数据，提取每个function的名字、描述和参数定义
    
    Args:
        functions_data: 原始的functions数据
        
    Returns:
        list: 解析后的functions列表，每个元素包含name、description、parameters
    """
    if not functions_data:
        return []
    
    parsed_functions = []
    
    # 确保functions_data是列表格式
    if isinstance(functions_data, dict):
        # 检查是否是单个function定义
        if 'name' in functions_data:
            functions_list = [functions_data]
        # 检查是否包含functions字段
        elif 'functions' in functions_data:
            functions_list = functions_data['functions']
        elif 'tools' in functions_data:
            functions_list = functions_data['tools']
        else:
            # 尝试将整个dict作为单个function处理
            functions_list = [functions_data]
    elif isinstance(functions_data, list):
        functions_list = functions_data
    else:
        return []
    
    for func_data in functions_list:
        if not isinstance(func_data, dict):
            continue
        
        parsed_func = {
            'name': None,
            'description': None,
            'parameters': None
        }
        
        # 提取function名称
        if 'name' in func_data:
            parsed_func['name'] = func_data['name']
        elif 'function' in func_data and isinstance(func_data['function'], dict):
            # OpenAI格式：{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
            func_def = func_data['function']
            parsed_func['name'] = func_def.get('name')
            parsed_func['description'] = func_def.get('description')
            parsed_func['parameters'] = func_def.get('parameters')
        elif 'type' in func_data and func_data['type'] == 'function':
            # 可能是工具格式
            parsed_func['name'] = func_data.get('name')
        
        # 提取描述
        if not parsed_func['description']:
            for desc_key in ['description', 'desc', 'summary', 'doc']:
                if desc_key in func_data:
                    parsed_func['description'] = func_data[desc_key]
                    break
        
        # 提取参数定义（更全面的提取）
        if not parsed_func['parameters']:
            # 优先从标准字段提取
            for param_key in ['parameters', 'params', 'arguments', 'args', 'schema']:
                if param_key in func_data:
                    param_value = func_data[param_key]
                    # 确保参数定义是完整的
                    if isinstance(param_value, dict):
                        parsed_func['parameters'] = param_value
                        break
                    elif isinstance(param_value, str):
                        # 先尝试特殊格式解析
                        parsed_param = _parse_special_parameters(param_value)
                        if isinstance(parsed_param, dict):
                            parsed_func['parameters'] = parsed_param
                            break
                        # 再尝试JSON字符串解析
                        parsed_param = _parse_json_if_string(param_value)
                        if isinstance(parsed_param, dict):
                            parsed_func['parameters'] = parsed_param
                            break
        
        # 如果仍然没有参数定义，尝试从properties中构建
        if not parsed_func['parameters'] and 'properties' in func_data:
            parsed_func['parameters'] = {
                'type': 'object',
                'properties': func_data['properties'],
                'required': func_data.get('required', [])
            }
        
        # 如果是OpenAI格式但参数在function字段中，确保提取完整
        if 'function' in func_data and isinstance(func_data['function'], dict):
            func_def = func_data['function']
            if 'parameters' in func_def and not parsed_func['parameters']:
                parsed_func['parameters'] = func_def['parameters']
        
        # 只有当至少有名称时才添加
        if parsed_func['name']:
            parsed_functions.append(parsed_func)
    
    return parsed_functions


def _search_nested_functions(data: Dict[str, Any], max_depth: int = 3) -> Any:
    """
    递归搜索嵌套字典中的functions定义
    
    Args:
        data: 要搜索的字典
        max_depth: 最大搜索深度
        
    Returns:
        找到的functions定义，如果没找到返回None
    """
    if max_depth <= 0 or not isinstance(data, dict):
        return None
    
    # 直接检查当前层级
    function_keys = ['functions', 'tools', 'available_functions', 'tool_schemas', 'function_definitions']
    for key in function_keys:
        if key in data:
            potential_functions = data[key]
            if _is_functions_definition(potential_functions):
                return potential_functions
    
    # 递归搜索子字典
    for key, value in data.items():
        if isinstance(value, dict):
            result = _search_nested_functions(value, max_depth - 1)
            if result:
                return result
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result = _search_nested_functions(item, max_depth - 1)
                    if result:
                        return result
    
    return None


def _is_functions_definition(data: Any) -> bool:
    """
    验证数据是否为functions定义
    
    Args:
        data: 要验证的数据
        
    Returns:
        bool: 是否为functions定义
    """
    if not data:
        return False
    
    # 如果是列表，检查列表中的元素
    if isinstance(data, list):
        if not data:
            return False
        # 检查第一个元素是否像function定义
        first_item = data[0]
        if isinstance(first_item, dict):
            # 检查是否有function定义的常见字段
            function_fields = ['name', 'description', 'parameters', 'function', 'type']
            return any(field in first_item for field in function_fields)
    
    # 如果是字典，检查是否有function相关字段
    elif isinstance(data, dict):
        function_fields = ['name', 'description', 'parameters', 'function', 'type']
        tools_fields = ['tools', 'functions', 'function_call']
        
        # 直接包含function字段
        if any(field in data for field in function_fields):
            return True
        
        # 包含tools或functions字段，且值是列表
        for field in tools_fields:
            if field in data and isinstance(data[field], (list, dict)):
                return True
        
        # 检查是否是OpenAI工具格式
        if 'type' in data and data['type'] == 'function' and 'function' in data:
            return True
    
    return False


def _parse_special_parameters(param_str: str) -> Dict[str, Any]:
    """
    解析特殊格式的参数字符串，如包含<Type.STRING: 'STRING'>的格式
    
    Args:
        param_str: 参数字符串
        
    Returns:
        解析后的参数字典
    """
    if not isinstance(param_str, str):
        return param_str
    
    try:
        # 替换特殊的Type格式
        cleaned_str = param_str
        
        # 替换 <Type.STRING: 'STRING'> 为 'string'
        cleaned_str = cleaned_str.replace("<Type.STRING: 'STRING'>", "'string'")
        cleaned_str = cleaned_str.replace("<Type.OBJECT: 'OBJECT'>", "'object'")
        cleaned_str = cleaned_str.replace("<Type.INTEGER: 'INTEGER'>", "'integer'")
        cleaned_str = cleaned_str.replace("<Type.NUMBER: 'NUMBER'>", "'number'")
        cleaned_str = cleaned_str.replace("<Type.BOOLEAN: 'BOOLEAN'>", "'boolean'")
        cleaned_str = cleaned_str.replace("<Type.ARRAY: 'ARRAY'>", "'array'")
        
        # 尝试使用eval解析（因为这是Python字典格式，不是JSON）
        result = eval(cleaned_str)
        return result
    except:
        # 如果解析失败，返回原字符串
        return param_str


def _parse_json_if_string(value: Any) -> Any:
    """
    如果值是JSON字符串，尝试解析为对象；否则返回原值
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def debug_functions_extraction(trace: Union[Dict, List]) -> Dict[str, Any]:
    """
    调试functions提取过程，帮助理解为什么没有提取到functions
    
    Args:
        trace: trace数据
        
    Returns:
        dict: 调试信息
    """
    debug_info = {
        'total_spans': 0,
        'llm_spans': [],
        'spans_with_functions_keywords': [],
        'all_attribute_keys': set(),
        'all_event_names': set(),
        'potential_functions_data': [],
        'parsed_functions': []
    }
    
    # 处理不同的trace数据格式
    spans = []
    if isinstance(trace, dict):
        if 'resourceSpans' in trace:
            for resource_span in trace['resourceSpans']:
                if 'scopeSpans' in resource_span:
                    for scope_span in resource_span['scopeSpans']:
                        if 'spans' in scope_span:
                            spans.extend(scope_span['spans'])
        elif 'spans' in trace:
            spans = trace['spans']
        else:
            spans = [trace]
    elif isinstance(trace, list):
        spans = trace
    
    debug_info['total_spans'] = len(spans)
    
    for i, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
            
        span_name = span.get('name', '')
        attributes = span.get('attributes', {})
        events = span.get('events', [])
        
        # 收集所有attribute keys
        debug_info['all_attribute_keys'].update(attributes.keys())
        
        # 收集所有event names
        for event in events:
            debug_info['all_event_names'].add(event.get('name', ''))
        
        # 检查是否是LLM span
        llm_indicators = ['llm', 'chat', 'completion', 'openai', 'anthropic', 'claude', 'generate', 'model', 'inference', 'ai_model', 'assistant', 'agent']
        if any(indicator in span_name.lower() for indicator in llm_indicators):
            debug_info['llm_spans'].append({
                'index': i,
                'name': span_name,
                'attributes_count': len(attributes),
                'events_count': len(events)
            })
        
        # 检查是否包含functions相关关键词
        function_keywords = ['function', 'tool', 'schema']
        has_function_keywords = False
        
        for key, value in attributes.items():
            if any(keyword in key.lower() for keyword in function_keywords):
                has_function_keywords = True
                debug_info['potential_functions_data'].append({
                    'span_index': i,
                    'span_name': span_name,
                    'attribute_key': key,
                    'value_type': type(value).__name__,
                    'value_length': len(str(value)) if value else 0,
                    'value_preview': str(value)[:200] if value else ''
                })
                
                # 特别检查request相关字段中的functions
                if 'request' in key.lower():
                    request_data = _parse_json_if_string(value)
                    if isinstance(request_data, dict) and 'functions' in request_data:
                        debug_info['parsed_functions'].append({
                            'span_index': i,
                            'span_name': span_name,
                            'source': f'request_attribute:{key}',
                            'raw_functions': request_data['functions'],
                            'parsed_functions': _parse_functions_details(request_data['functions'])
                        })
        
        for event in events:
            for attr_key, attr_value in event.get('attributes', {}).items():
                if any(keyword in attr_key.lower() for keyword in function_keywords):
                    has_function_keywords = True
                    debug_info['potential_functions_data'].append({
                        'span_index': i,
                        'span_name': span_name,
                        'event_name': event.get('name', ''),
                        'attribute_key': attr_key,
                        'value_type': type(attr_value).__name__,
                        'value_length': len(str(attr_value)) if attr_value else 0,
                        'value_preview': str(attr_value)[:200] if attr_value else ''
                    })
        
        if has_function_keywords:
            debug_info['spans_with_functions_keywords'].append({
                'index': i,
                'name': span_name
            })
    
    # 转换set为list以便JSON序列化
    debug_info['all_attribute_keys'] = list(debug_info['all_attribute_keys'])
    debug_info['all_event_names'] = list(debug_info['all_event_names'])
    
    return debug_info


def analyze_tool_usage_summary(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析工具使用情况的汇总信息
    
    Args:
        tool_calls: extract_tool_calls_from_trace返回的工具调用列表
        
    Returns:
        dict: 包含工具使用统计信息
    """
    if not tool_calls:
        return {
            'total_calls': 0,
            'unique_tools': 0,
            'tool_frequency': {},
            'average_duration': None,
            'total_duration': None
        }
    
    tool_frequency = {}
    durations = []
    
    for call in tool_calls:
        tool_name = call.get('tool_name', 'unknown')
        tool_frequency[tool_name] = tool_frequency.get(tool_name, 0) + 1
        
        if call.get('duration') is not None:
            durations.append(call['duration'])
    
    total_duration = sum(durations) if durations else None
    average_duration = total_duration / len(durations) if durations and total_duration is not None else None
    
    return {
        'total_calls': len(tool_calls),
        'unique_tools': len(tool_frequency),
        'tool_frequency': tool_frequency,
        'average_duration': average_duration,
        'total_duration': total_duration,
        'tools_with_errors': len([call for call in tool_calls if call.get('output_result') and 'error' in str(call['output_result']).lower()])
    }


def get_context_from_trace(trace: Union[Dict, List]) -> str:
    """
    从trace中提取load_knowledge工具检索的context，帮助理解工具调用的背景
    
    Args:
        trace: trace数据
        
    Returns:
        str: 提取的上下文信息字符串
    """
    context_pieces = []
    
    # 处理不同的trace数据格式
    spans = []
    if isinstance(trace, dict):
        if 'resourceSpans' in trace:
            for resource_span in trace['resourceSpans']:
                if 'scopeSpans' in resource_span:
                    for scope_span in resource_span['scopeSpans']:
                        if 'spans' in scope_span:
                            spans.extend(scope_span['spans'])
        elif 'spans' in trace:
            spans = trace['spans']
        else:
            spans = [trace]
    elif isinstance(trace, list):
        spans = trace
    
    for span in spans:
        if not isinstance(span, dict):
            continue
            
        span_name = span.get('name', '').lower()
        attributes = span.get('attributes', {})
        events = span.get('events', [])
        
        # 识别load_knowledge相关的span
        if 'load_knowledgebase' in span_name or 'knowledge_retrieval' in span_name or 'retrieve_context' in span_name:
            # 尝试从attributes中提取context
            for key, value in attributes.items():
                key_lower = key.lower()
                if 'tool.output' in key_lower:
                    output = _parse_json_if_string(value)
                    context_pieces = output['response']['result']['knowledges'] if isinstance(output, dict) and 'response' in output and 'result' in output['response'] and 'knowledges' in output['response']['result'] else []
            
    return [c['content'] for c in context_pieces if 'content' in c]


def _add_execution_order(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据工具执行时间按先后顺序标记工具的执行顺序index
    
    Args:
        tool_calls: 工具调用列表
        
    Returns:
        list: 添加了execution_order字段的工具调用列表，按时间顺序排序
    """
    if not tool_calls:
        return tool_calls
    
    # 为每个工具调用添加排序键
    for call in tool_calls:
        timestamp = call.get('timestamp')
        if timestamp:
            # 尝试将timestamp转换为数值进行排序
            try:
                # 如果是字符串格式的数字，转换为整数
                if isinstance(timestamp, str) and timestamp.isdigit():
                    call['_sort_key'] = int(timestamp)
                # 如果已经是数字，直接使用
                elif isinstance(timestamp, (int, float)):
                    call['_sort_key'] = int(timestamp)
                else:
                    # 如果是其他格式，尝试解析
                    call['_sort_key'] = int(str(timestamp))
            except (ValueError, TypeError):
                # 如果无法转换，使用0作为默认值
                call['_sort_key'] = 0
        else:
            # 如果没有timestamp，使用0
            call['_sort_key'] = 0
    
    # 按时间戳排序
    sorted_calls = sorted(tool_calls, key=lambda x: x.get('_sort_key', 0))
    
    # 添加执行顺序index并移除临时排序键
    for i, call in enumerate(sorted_calls, 1):
        call['execution_order'] = i
        # 移除临时的排序键
        call.pop('_sort_key', None)
    
    return sorted_calls


if __name__ == "__main__":
    with open('dataset/mcp/tmp/veadk_opentelemetry_tracer_veadk_default_user_financial_deep_research_1_425992b64225f91fb71ae9d611817b98.json', encoding='utf8', mode='r') as f:
        trace_data = json.load(f)
        tools_result = extract_tool_calls_from_trace(trace_data)
        print(json.dumps(tools_result[1], indent=2, ensure_ascii=False))