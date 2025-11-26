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

from typing import List, Optional, Any
from veagentbench.dataset.dataset import Dataset
from veagentbench.agents.base_agent import BaseAgent
from veagentbench.evals.deepeval.metrics.base_metric import BaseMetric
class LLMAgentTestCaseRecord:
    def __init__(self, test_case_id: str, trace_file_path: str, resoponse: str):
        self.test_case_id = test_case_id
        self.trace_file_path = trace_file_path
        self.response = resoponse

class BaseTask:

    def __init__(
        self, 
        task_name: str,
        metrics: List[BaseMetric],
        datasets: List[Dataset],
        agent: Optional[BaseAgent]=None,
        
        
    ):
        self.task_name = task_name
        self.metrics = metrics
        self.dataset : List[str]= None
        self.output_dir : str = './output'
        self.agent = agent
        self.datasets = datasets
        self.test_env = None   #评测环境，预留

    @classmethod
    def load_data(self, data_path: str):
        raise NotImplementedError("Subclasses should implement load_data method.")
    
    @classmethod
    def run(self):
        raise NotImplementedError("Subclasses should implement run method.")
    
    @classmethod
    def post_process(self, results):
        raise NotImplementedError("Subclasses should implement post_process method.")