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

# 基础度量指标类
from .base import BaseMetric

# 自定义度量指标
from .custom_metric import (
    CustomMetric,
    CustomEvaluationResult,
    CustomScoreReason,
    CustomMetricTemplate,
)

# 答案正确性度量指标
from .answer_correctness import (
    AnswerCorrectnessMetric,
    StatementVerdict,
    AnswerCorrectnessVerdicts,
    AnswerCorrectnessScoreReason,
    StatementGenerationOutput,
    AnswerCorrectnessTemplate,
)

# MCP Bench 度量指标
from .mcp_bench import (
    MCPToolMetric,
    ToolCall,
    MCPToolEvaluationResult,
    MCPToolVerdicts,
    MCPToolScoreReason,
    ToolCallAnalysis,
    MCPToolTemplate,
)

# 性能度量指标
from .performance_metric import (
    PerformanceMetric,
)

# 从 deepeval 合并的度量指标
from veagentbench.evals.deepeval.metrics.base_metric import (
    BaseMetric as DeepEvalBaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
    BaseArenaMetric,
)

from veagentbench.evals.deepeval.metrics.dag.dag import DAGMetric, DeepAcyclicGraph
from veagentbench.evals.deepeval.metrics.conversational_dag.conversational_dag import ConversationalDAGMetric
from veagentbench.evals.deepeval.metrics.bias.bias import BiasMetric
from veagentbench.evals.deepeval.metrics.toxicity.toxicity import ToxicityMetric
from veagentbench.evals.deepeval.metrics.pii_leakage.pii_leakage import PIILeakageMetric
from veagentbench.evals.deepeval.metrics.non_advice.non_advice import NonAdviceMetric
from veagentbench.evals.deepeval.metrics.misuse.misuse import MisuseMetric
from veagentbench.evals.deepeval.metrics.role_violation.role_violation import RoleViolationMetric
from veagentbench.evals.deepeval.metrics.hallucination.hallucination import HallucinationMetric
from veagentbench.evals.deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from veagentbench.evals.deepeval.metrics.summarization.summarization import SummarizationMetric
from veagentbench.evals.deepeval.metrics.g_eval.g_eval import GEval
from veagentbench.evals.deepeval.metrics.arena_g_eval.arena_g_eval import ArenaGEval
from veagentbench.evals.deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from veagentbench.evals.deepeval.metrics.contextual_recall.contextual_recall import ContextualRecallMetric
from veagentbench.evals.deepeval.metrics.contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric
from veagentbench.evals.deepeval.metrics.contextual_precision.contextual_precision import ContextualPrecisionMetric
from veagentbench.evals.deepeval.metrics.knowledge_retention.knowledge_retention import KnowledgeRetentionMetric
from veagentbench.evals.deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from veagentbench.evals.deepeval.metrics.json_correctness.json_correctness import JsonCorrectnessMetric
from veagentbench.evals.deepeval.metrics.prompt_alignment.prompt_alignment import PromptAlignmentMetric
from veagentbench.evals.deepeval.metrics.task_completion.task_completion import TaskCompletionMetric
from veagentbench.evals.deepeval.metrics.argument_correctness.argument_correctness import ArgumentCorrectnessMetric
from veagentbench.evals.deepeval.metrics.mcp.mcp_task_completion import MCPTaskCompletionMetric
from veagentbench.evals.deepeval.metrics.mcp.multi_turn_mcp_use_metric import MultiTurnMCPUseMetric
from veagentbench.evals.deepeval.metrics.mcp_use_metric.mcp_use_metric import MCPUseMetric
from veagentbench.evals.deepeval.metrics.turn_relevancy.turn_relevancy import TurnRelevancyMetric
from veagentbench.evals.deepeval.metrics.conversation_completeness.conversation_completeness import ConversationCompletenessMetric
from veagentbench.evals.deepeval.metrics.role_adherence.role_adherence import RoleAdherenceMetric
from veagentbench.evals.deepeval.metrics.conversational_g_eval.conversational_g_eval import ConversationalGEval
from veagentbench.evals.deepeval.metrics.multimodal_metrics import (
    TextToImageMetric,
    ImageEditingMetric,
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
    MultimodalContextualRecallMetric,
    MultimodalContextualRelevancyMetric,
    MultimodalContextualPrecisionMetric,
    MultimodalAnswerRelevancyMetric,
    MultimodalFaithfulnessMetric,
    MultimodalToolCorrectnessMetric,
    MultimodalGEval,
)

__all__ = [
    # 基础类
    "BaseMetric",
    "DeepEvalBaseMetric",
    "BaseConversationalMetric",
    "BaseMultimodalMetric",
    "BaseArenaMetric",
    
    # VeAgentBench 自定义度量指标
    "CustomMetric",
    "CustomEvaluationResult",
    "CustomScoreReason",
    "CustomMetricTemplate",
    "AnswerCorrectnessMetric",
    "StatementVerdict",
    "AnswerCorrectnessVerdicts",
    "AnswerCorrectnessScoreReason",
    "StatementGenerationOutput",
    "AnswerCorrectnessTemplate",
    "MCPToolMetric",
    "ToolCall",
    "MCPToolEvaluationResult",
    "MCPToolVerdicts",
    "MCPToolScoreReason",
    "ToolCallAnalysis",
    "MCPToolTemplate",
    "PerformanceMetric",

    
    # DeepEval 核心度量指标
    "GEval",
    "ArenaGEval",
    "ConversationalGEval",
    "DAGMetric",
    "DeepAcyclicGraph",
    "ConversationalDAGMetric",
    
    # RAG 度量指标
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    
    # MCP 度量指标
    "MCPTaskCompletionMetric",
    "MultiTurnMCPUseMetric",
    "MCPUseMetric",
    
    # 内容质量度量指标
    "HallucinationMetric",
    "BiasMetric",
    "ToxicityMetric",
    "SummarizationMetric",
    
    # 安全和合规度量指标
    "PIILeakageMetric",
    "NonAdviceMetric",
    "MisuseMetric",
    "RoleViolationMetric",
    "RoleAdherenceMetric",
    
    # 任务特定度量指标
    "ToolCorrectnessMetric",
    "JsonCorrectnessMetric",
    "PromptAlignmentMetric",
    "TaskCompletionMetric",
    "ArgumentCorrectnessMetric",
    "KnowledgeRetentionMetric",
    
    # 对话度量指标
    "TurnRelevancyMetric",
    "ConversationCompletenessMetric",
    
    # 多模态度量指标
    "TextToImageMetric",
    "ImageEditingMetric",
    "ImageCoherenceMetric",
    "ImageHelpfulnessMetric",
    "ImageReferenceMetric",
    "MultimodalContextualRecallMetric",
    "MultimodalContextualRelevancyMetric",
    "MultimodalContextualPrecisionMetric",
    "MultimodalAnswerRelevancyMetric",
    "MultimodalFaithfulnessMetric",
    "MultimodalToolCorrectnessMetric",
    "MultimodalGEval",
]
