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


from analyze_trace import *
import sys

trace_file = sys.argv[1]
with open(trace_file) as f:
    trace_data = json.load(f)

tool_calls, avalible = extract_tool_calls_from_trace(trace_data)
print("=== 工具调用信息 ===")
for tool_call in tool_calls:
    execution_order = tool_call.get('execution_order', 'N/A')
    print(f"[执行顺序 {execution_order}] 工具名称: {tool_call.get('tool_name', 'N/A')}")
    print(f"   输入参数: {tool_call.get('input_params', {})}")
    print(f"   输出结果: {tool_call.get('output_result', 'N/A')}")
    print(f"   执行时长: {tool_call.get('duration', 'N/A')}")
    print(f"   时间戳: {tool_call.get('timestamp', 'N/A')}")
    print(f"   Span ID: {tool_call.get('span_id', 'N/A')}")
    print(f"   Trace ID: {tool_call.get('trace_id', 'N/A')}")
print(f"   可用函数: {avalible}")
print()
summary = analyze_tool_usage_summary(tool_calls)
print(summary)



context = get_context_from_trace(trace_data)
print(context)