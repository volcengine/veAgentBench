
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

from veagentbench.task.base import BaseTask
from agents.base_agent import BaseAgent
from veagentbench.metrics.base import BaseMetric
from veagentbench.evals.deepeval.metrics import GEval,ToolCorrectnessMetric, ArgumentCorrectnessMetric, ContextualRecallMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, AnswerRelevancyMetric, FaithfulnessMetric
from veagentbench.metrics.answer_correctness._answer_correctness import AnswerCorrectnessMetric
import pandas as pd
from veagentbench.evals.deepeval.test_case import LLMTestCase, MCPServer, ToolCall
from veagentbench.utils.analyze_trace import *
from typing import List, Union
from veagentbench.utils.eval_set import get_output_from_eval_case
from veagentbench.models.models import VolceOpenAI
from veagentbench.evals.deepeval import evaluate
import os
from veagentbench.evals.deepeval.evaluate import AsyncConfig, ErrorConfig, CacheConfig
from veagentbench.evals.deepeval.models import GPTModel
from veagentbench.models.models import VolceOpenAI
from veagentbench.evals.deepeval.test_case import LLMTestCaseParams


class RAGTask(BaseTask):

    def __init__(self, task_name: str, agent: BaseAgent):
        super().__init__(task_name)
        self.metrics = []
        self.dataset = None
        self.output_dir = None
        self.agent = agent
        # self.llm_mode = VolceOpenAI(model="ep-20250618145117-ps5rq")
        self.llm_model = VolceOpenAI(
            # model="ep-20250618145117-ps5rq",
            model = 'ep-20250509115145-gtm8g',
            temperature=0,
            base_url="https://ark-cn-beijing.bytedance.net/api/v3",
            cost_per_input_token=0.000002,
            cost_per_output_token=0.000008,
            _openai_api_key=os.environ.get("ARK_API_KEY"),
            
            # api_key=os.environ.get("ARK_API_KEY"),
        )
    
    def add_metric(self, metric):
        self.metrics.append(metric)

    
    def evaluate(self, predictions, references):
        # Implement evaluation logic specific to RAG tasks
        pass

    
    def load_data(self, data_path: str):
        # Implement data loading logic specific to RAG tasks

        csv_reader = pd.read_csv(data_path)
        self.dataset = csv_reader.to_dict()
        # print(self.dataset)
        
        

    
    def run(self):
        # Implement the main execution logic for RAG tasks
        # self.add_metric(ArgumentCorrectnessMetric(model=self.llm_model))
        # self.add_metric(ContextualRecallMetric(model=self.llm_model))
        # self.add_metric(ToolCorrectnessMetric())
        # self.add_metric(AnswerCorrectnessMetric(model=self.llm_model, threshold=0.3))
        # self.add_metric(ContextualPrecisionMetric(model=self.llm_model))
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            # evaluation_steps=[
            #     "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            #     "You should also heavily penalize omission of detail",
            #     "Vague language, or contradicting OPINIONS, are OK"
            # ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=self.llm_model
        )
        self.add_metric(correctness_metric)
        # self.add_metric(FaithfulnessMetric(model=self.llm_model))
        testcase = []
        for i in range(0, len(self.dataset['序号'])):
            print('index: %s'%i)
            input_text = self.dataset['input'][i]
            reference = self.dataset['expect_output'][i]
            index = self.dataset['序号'][i]-1
            # prediction = self.agent.generate_response(input_text)
            trace_file = os.path.join('dataset/tmp', self.dataset.get('trace_id', None)[i])
            eva_set_file = self.dataset.get('eval_set_id', None)[i]
            expected_tool = self.dataset.get('expect_tools', '')[0]
            expected_tool_calls_name = expected_tool.split(',') if expected_tool else []
            expected_tool_calls_name.remove('load_knowledgebase')
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
            tool_calls = extract_tool_calls_from_trace(trace_data)
            expected_tool_calls = []
            def turn_toolcalls(tool_calls) -> Union[List[ToolCall], List[ToolCall]]:
                tool_use = []
                avalible_tool_use = []
                expected_tool_calls = []
                for call in tool_calls:
                    tool_name = call.get('tool_name', 'unknown_tool')
                    parameters = call.get('input_params', {})
                    output_result = call.get('output_result', {})
                    if tool_name in ['load_knowledgebase', '(merged tools)']:
                        continue
                    tool_use.append(ToolCall(
                        name=tool_name,
                        input_parameters=parameters,
                        output=output_result,
                        discription=call.get('description', '')
                    ))
                    avalible_tools = call.get('available_functions', [])
                    for avalible_tool in avalible_tools:
                        if isinstance(avalible_tool, dict):
                            if avalible_tool.get('name', 'unknown_tool') not in  ['load_knowledgebase', '(merged tools)']:
                                
                                avalible_tool_use.append(ToolCall(
                                    name=avalible_tool.get('name', 'unknown_tool'),
                                    input_parameters=avalible_tool.get('parameters', {}),
                                    discription=avalible_tool.get('description', ''),
                                    output={}
                                ))
                                if  avalible_tool.get('name', 'unknown_tool') in expected_tool_calls_name:
                                    expected_tool_calls.append(ToolCall(
                                        name=avalible_tool.get('name', 'unknown_tool'),
                                        input_parameters=avalible_tool.get('parameters', {}),
                                        description=avalible_tool.get('description', ''),
                                        output={}
                                    ))
                return tool_use, avalible_tool_use, expected_tool_calls


            tools_called , avalible_toole_use, expected_tool_calls = turn_toolcalls(tool_calls)
            
            with open(eva_set_file, 'r', encoding='utf-8') as f:
                eval_case = json.load(f)
            # print(eval_case)
            output = get_output_from_eval_case(eval_case['eval_cases'][index])
            contexts = get_context_from_trace(trace_data)
            testcase.append(LLMTestCase(
                input=input_text,
                actual_output=output,
                expected_output=reference,
                tools_called=tools_called,
                retrieval_context=contexts,
                expected_tools=expected_tool_calls
            ))
        
        evaluate(testcase, self.metrics, async_config=AsyncConfig(max_concurrent=100), \
                 error_config=ErrorConfig(ignore_errors=False), cache_config=CacheConfig(use_cache=False, write_cache=True))

    def post_process(self, results):
        # 增加对badcase的一些分析处理，输出当前agent在该任务下的badcase以及存在的主要问题
        pass

