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

from typing import List
from pydantic import BaseModel


class LongMemEvalVerdict(BaseModel):
    question: str
    correct_answer: str
    model_response: str
    verdict: str  # "yes" or "no"
    reason: str


class LongMemEvalResult(BaseModel):
    verdicts: List[LongMemEvalVerdict]


class LongMemEvalScoreReason(BaseModel):
    reason: str


class LongMemEvalEvaluationOutput(BaseModel):
    is_correct: bool
    explanation: str
