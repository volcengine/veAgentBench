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

from typing import Optional, Dict, Any

class PerformanceTemplate:
    @staticmethod
    def generate_reason(
        total_calls: int,
        success_rate: float,
        avg_duration: float,
        rounds: int,
        parallel_efficiency: float,
        end_to_end_duration: Optional[float] = None,
        tool_call_duration: Optional[float] = None,
        llm_call_duration: Optional[float] = None,
    ) -> str:
        # 基础性能信息
        reason_parts = [
            f"共执行 {total_calls} 次工具调用，成功率 {success_rate:.2f}。",
            f"平均耗时 {avg_duration:.3f}s，轮次 {rounds}，并发效率约 {parallel_efficiency:.3f}。"
        ]
        
        # 添加端到端耗时信息
        if end_to_end_duration is not None:
            reason_parts.append(f"端到端总耗时 {end_to_end_duration:.3f}s。")
        
        # 添加工具调用耗时信息
        if tool_call_duration is not None and tool_call_duration > 0:
            reason_parts.append(f"工具调用总耗时 {tool_call_duration:.3f}s。")
        
        # 添加LLM调用耗时信息
        if llm_call_duration is not None and llm_call_duration > 0:
            reason_parts.append(f"LLM调用总耗时 {llm_call_duration:.3f}s。")
        
        return " ".join(reason_parts)
