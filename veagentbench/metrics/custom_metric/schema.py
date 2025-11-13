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

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class CustomEvaluationResult(BaseModel):
    """Schema for custom evaluation result"""
    score: float
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class CustomScoreReason(BaseModel):
    """Schema for custom score reason"""
    reason: str
    details: Optional[Dict[str, Any]] = None


class CustomEvaluationParams(BaseModel):
    """Schema for custom evaluation parameters"""
    template_params: Dict[str, Any]
    evaluation_context: Optional[Dict[str, Any]] = None
