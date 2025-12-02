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

import pandas as pd
import json
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from datasets import load_dataset, DatasetDict

class Dataset():
    """
    数据集基类，用于加载和管理不同类型的数据集
    """
    
    def __init__(
        self,
        name: str,
        description: str
    ):
        self.name = name
        self.description = description


    def load(
        self,
        load_type: str = Field(
            default='csv',
            examples = ['csv', 'trace_local', 'trace_cloud', 'huggingface'],
            description='数据载入类型'
        ),
        **kwargs
    ):
        """
        加载数据集的基本方法
        """
        if load_type == 'csv':
            self.testcases = self.from_csv(**kwargs)
        elif load_type == 'huggingface':
            self.testcases = self.from_huggingface(**kwargs)
            

    def get_testcase(self):
        for testcase in self.testcases:
            yield testcase

    def from_csv(
        self,
        csv_file: str,
        avalible_tools_column: str="avalible_tools",
        input_column: str='input',
        mulit_turn_input_column: str='input_list',
        expected_column: Optional[str]='expected_output',
        context_retrieved_column: Optional[str]='context_retrieved_column',
        expected_tool_call_column: Optional[str] = Field(
            default='',
            description='''
            预期的工具调用；格式如下：
            {
                "tool_name":
                "server_name":(默认defaul)
                "discription":
                "parameters":
                "expected_output":
            }
            '''
        )
    ):
        """
        从CSV文件加载数据集
        
        Args:
            csv_file: CSV文件路径
            input_column: 输入列名
            expected_column: 预期结果列名
            expected_tool_call_column: 预期工具调用列名（可选）
            
        Returns:
            List[Dict]: 包含测试用例的列表
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 获取所有列名
            all_columns = df.columns.tolist()
            
            # 如果输入列不存在，尝试使用第一个非索引列
            if input_column not in df.columns:
                # 尝试使用常见的输入列名
                possible_input_cols = ['input', 'Input', 'INPUT', 'prompt', 'Prompt', 'PROMPT']
                for col in possible_input_cols:
                    if col in df.columns:
                        input_column = col
                        break
                else:
                    # 使用第一个非索引列
                    non_index_cols = [col for col in df.columns if col not in ['index', 'Index', 'INDEX', '序号', 'id', 'ID']]
                    if non_index_cols:
                        input_column = non_index_cols[0]
                    else:
                        raise ValueError(f"输入列 '{input_column}' 不存在于CSV文件中")
            
            
            
            
            # 如果预期列不存在，尝试使用常见的预期列名
            if expected_column not in df.columns:
                possible_expected_cols = ['expect', 'Expect', 'EXPECTED', 'expected', 'expectation', 'Expectation']
                for col in possible_expected_cols:
                    if col in df.columns:
                        expected_column = col
                        break
                else:
                    # 使用第二个非索引列
                    non_index_cols = [col for col in df.columns if col not in ['index', 'Index', 'INDEX', '序号', 'id', 'ID']]
                    if len(non_index_cols) > 1:
                        expected_column = non_index_cols[1]
                    else:
                        raise ValueError(f"预期列 '{expected_column}' 不存在于CSV文件中")
            
            # 如果工具调用列存在但为空字符串，尝试使用常见的工具调用列名
            if expected_tool_call_column == '' or expected_tool_call_column not in df.columns:
                possible_tool_cols = ['expect_tools_calls', 'expected_tools_calls', 'tool_calls', 'ToolCalls']
                for col in possible_tool_cols:
                    if col in df.columns:
                        expected_tool_call_column = col
                        break
            
            test_cases = []
            
            # 遍历每一行数据
            for index, row in df.iterrows():
                test_case = {
                    'id': index + 1,
                    'input': str(row[input_column]) if pd.notna(row[input_column]) else '',
                    'expected_output': str(row[expected_column]) if pd.notna(row[expected_column]) else '',
                }
                #如果多轮对话输入不为空
                if mulit_turn_input_column and mulit_turn_input_column in df.columns and pd.notna(row[mulit_turn_input_column]):
                    mulit_turn_input = str(row[mulit_turn_input_column])
                    try:
                        # 尝试解析为JSON格式
                        if mulit_turn_input.strip():
                            mulit_turn_input = json.loads(mulit_turn_input)
                            test_case['input_list'] = [x[0]['content'] for x in mulit_turn_input]
                        else:
                            test_case['input_list'] = []
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是有效的JSON，则作为字符串处理
                        test_case['input_list'] = mulit_turn_input.strip()
                else:
                    test_case['input_list'] = []
                
                #如果期望的调用工具存在且不为空
                if avalible_tools_column and avalible_tools_column in df.columns and pd.notna(row[avalible_tools_column]):
                    avalible_tools = str(row[avalible_tools_column])
                    try:
                        # 尝试解析为JSON格式
                        if avalible_tools.strip():
                            avalible_tools = json.loads(avalible_tools)
                            test_case['available_tools'] = avalible_tools
                        else:
                            test_case['available_tools'] = []
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是有效的JSON，则作为字符串处理
                        test_case['available_tools'] = avalible_tools.strip()
                else:
                    test_case['expected_tools'] = []
                
                # 如果工具调用列存在且不为空
                if expected_tool_call_column and expected_tool_call_column in df.columns and pd.notna(row[expected_tool_call_column]):
                    tool_call_data = str(row[expected_tool_call_column])
                    try:
                        # 尝试解析为JSON格式
                        if tool_call_data.strip():
                            tool_calls = json.loads(tool_call_data)
                            test_case['expected_tools'] = tool_calls
                        else:
                            test_case['expected_tools'] = []
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是有效的JSON，则作为字符串处理
                        test_case['expected_tools'] = tool_call_data.strip()
                else:
                    test_case['expected_tools'] = []
                
                # 添加其他可能的列
                for col in df.columns:
                    if col not in [input_column, expected_column, expected_tool_call_column]:
                        col_value = row[col]
                        if pd.notna(col_value):
                            test_case[col] = str(col_value)
                
                test_cases.append(test_case)
            

            
            return test_cases
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV文件 '{csv_file}' 不存在")
        except Exception as e:
            raise Exception(f"读取CSV文件时发生错误: {str(e)}")

    @abstractmethod
    def from_trace_local(self):
        """
        从本地trace文件加载数据集
        """
        pass

    @abstractmethod
    def from_cloud(self):
        """
        从云端加载数据集
        """
        pass

    def from_huggingface(
        self,
        config_name: str,
        split: str='test',
        input_column: str = "input",
        expected_column: Optional[str] = "expect_output",
        expected_tool_call_column: Optional[str] = "expected_tool_calls",
        available_tools_column: Optional[str] = "available_tools",
        context_column: Optional[str] = "context",
        id_column: Optional[str] = None,
        **load_kwargs
    ) -> List[Dict[str, Any]]:
        """
        从Hugging Face数据集加载测试用例
        
        Args:
            input_column: 输入列名
            expected_column: 预期输出列名
            expected_tool_call_column: 预期工具调用列名
            available_tools_column: 可用工具列名
            context_column: 上下文列名
            id_column: ID列名（如果不提供，将使用索引）
            **load_kwargs: 传递给load_dataset的其他参数
            
        Returns:
            List[Dict]: 包含测试用例的列表
        """
        try:
            # 从Hugging Face Hub加载数据集
            if config_name:
                dataset_dict = load_dataset(
                    self.name, 
                    config_name, 
                    split=split,
                )
            else:
                dataset_dict = load_dataset(
                    self.name, 
                    split=split,
                )
            
            # 确保我们有一个Dataset对象
            if isinstance(dataset_dict, DatasetDict):
                self.hf_dataset = dataset_dict[split]
            else:
                self.hf_dataset = dataset_dict
            
            # 获取数据集的特征（列名）
            features = self.hf_dataset.features
            def _find_column( features, preferred_column: str, fallback_columns: List[str]) -> Optional[str]:
                """
                查找列名，如果首选列不存在，则尝试备选列
                
                Args:
                    features: 数据集的特征
                    preferred_column: 首选列名
                    fallback_columns: 备选列名列表
                    
                Returns:
                    找到的列名，如果都找不到则返回None
                """
                # 获取实际的列名列表
                actual_columns = list(features.keys())
                
                # 检查首选列
                if preferred_column in actual_columns:
                    return preferred_column
                
                # 尝试备选列
                for col in fallback_columns:
                    if col in actual_columns:
                        return col
                
                return None

            def _parse_tools_field( tools_data: Any) -> Union[List[Dict], str, List]:
                """
                解析工具字段数据
                
                Args:
                    tools_data: 工具数据（可能是字符串、列表、字典等）
                    
                Returns:
                    解析后的工具数据
                """
                if tools_data is None:
                    return []
                
                if isinstance(tools_data, str):
                    # 如果是字符串，尝试解析为JSON
                    try:
                        if tools_data.strip():
                            parsed = json.loads(tools_data)
                            return parsed if isinstance(parsed, list) else [parsed]
                        else:
                            return []
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是有效的JSON，返回原始字符串
                        return tools_data.strip()
                elif isinstance(tools_data, (list, dict)):
                    # 如果是列表或字典，直接返回
                    return tools_data if isinstance(tools_data, list) else [tools_data]
                else:
                    # 其他类型，转换为字符串
                    return str(tools_data)
    
   
            # 自动检测列名（如果提供的列名不存在）
            input_column = _find_column(features, input_column, ["input", "question", "text", "prompt"])
            expected_column = _find_column(features, expected_column, ["answer", "output", "target", "label"])
            expected_tool_call_column = _find_column(features, expected_tool_call_column, 
                                                         ["tool_calls", "expected_tools", "expected_tool_calls"])
            available_tools_column = _find_column(features, available_tools_column, ["tools", "available_tools"])
            context_column = _find_column(features, context_column, ["context", "passage", "document"])
            
            test_cases = []
            
            # 遍历数据集中的每个样本
            for idx, sample in enumerate(self.hf_dataset):
                test_case = {
                    'id': sample[id_column] if id_column and id_column in sample else idx + 1,
                    'input': str(sample.get(input_column, "")) if input_column in sample else "",
                    'expected_output': str(sample.get(expected_column, "")) if expected_column and expected_column in sample else "",
                }
                
                # 处理可用工具
                if available_tools_column and available_tools_column in sample:
                    available_tools = sample[available_tools_column]
                    test_case['available_tools'] = _parse_tools_field(available_tools)
                
                # 处理预期工具调用
                if expected_tool_call_column and expected_tool_call_column in sample:
                    expected_tools = sample[expected_tool_call_column]
                    test_case['expected_tools'] = _parse_tools_field(expected_tools)
                
                # 处理上下文
                if context_column and context_column in sample:
                    context = sample[context_column]
                    test_case['context'] = str(context) if context is not None else ""
                
                # 添加其他字段
                for key, value in sample.items():
                    if key not in [input_column, expected_column, expected_tool_call_column, 
                                 available_tools_column, context_column, id_column]:
                        if key not in test_case:
                            test_case[key] = str(value) if value is not None else ""
                
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            raise Exception(f"加载Hugging Face数据集时发生错误: {str(e)}")
    



if __name__ == "__main__":
    
    dataset = Dataset(name='bytedance-research/veAgentBench', description='test')
    dataset.load(load_type='huggingface', split='test[:1]',config_name='financial_analysis', input_column='input', expected_column='expect_output', expected_tool_call_column='expected_tool_calls')
    for test in dataset.get_testcase():
        print(test)