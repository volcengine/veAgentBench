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

from typing import Optional
from pydantic import BaseModel

class PerformanceSummary(BaseModel):
    # 为了兼容性保留的基础字段
    total_calls: int
    success_calls: int
    success_rate: float
    fail_rate: float
    total_duration: float
    avg_duration: float
    rounds: int
    parallel_efficiency: float
    reason: Optional[str] = None
    
    # 核心时延指标 - 只保留这三个关键指标
    end_to_end_duration: Optional[float] = None  # 端到端总耗时
    tool_call_duration: Optional[float] = None     # 工具调用总耗时
    llm_call_duration: Optional[float] = None      # LLM调用总耗时
