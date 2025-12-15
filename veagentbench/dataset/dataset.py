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
import traceback
from enum import Enum
import logging

# 设置日志
logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """数据格式枚举"""
    CSV = "csv"
    JSONL = "jsonl"
    HUGGINGFACE = "huggingface"
    TRACE_LOCAL = "trace_local"
    TRACE_CLOUD = "trace_cloud"

class FieldMapping:
    """字段映射配置"""
    def __init__(
        self,
        input_column: str = "input",
        expected_column: str = "expected_output",
        expected_tool_call_column: str = "expected_tool_calls",
        available_tools_column: str = "available_tools",
        context_column: str = "context",
        id_column: str = "id",
        multi_turn_input_column: str = "input_list",
        case_name_column: str = "name",
        extra_fields: List[str] = None,
        **kwargs
    ):
        self.input_column = input_column
        self.expected_column = expected_column
        self.expected_tool_call_column = expected_tool_call_column
        self.available_tools_column = available_tools_column
        self.context_column = context_column
        self.id_column = id_column
        self.multi_turn_input_column = multi_turn_input_column
        self.case_name_column = case_name_column
        self.extra_fields = extra_fields or []

class DataProcessor:
    """统一数据处理器"""
    
    # 列名备选方案
    INPUT_ALTERNATIVES = ['input', 'Input', 'INPUT', 'prompt', 'Prompt', 'PROMPT', 'question', 'Question', 'QUESTION']
    EXPECTED_ALTERNATIVES = ['answer', 'Answer', 'ANSWER', 'output', 'Output', 'OUTPUT', 'expected', 'Expected', 'EXPECTED', 
                             'expect_output', 'target', 'Target', 'TARGET', 'expect', 'Expect']
    TOOL_CALLS_ALTERNATIVES = ['tool_calls', 'expected_tools', 'expected_tool_calls', 'expect_tools_calls', 'ToolCalls']
    TOOLS_ALTERNATIVES = ['tools', 'available_tools', 'avalible_tools']
    CONTEXT_ALTERNATIVES = ['context', 'passage', 'document', 'context_retrieved_column']
    
    @staticmethod
    def find_column(columns: List[str], preferred: str, alternatives: List[str]) -> str:
        """查找列名，如果首选列不存在，则尝试备选列"""
        if preferred in columns:
            return preferred
        
        for alt in alternatives:
            if alt in columns:
                return alt
        
        # 如果都找不到，返回首选列名（会触发后续的错误处理）
        return preferred
    
    @staticmethod
    def parse_tools_field(tools_data: Any) -> Union[List[Dict], str, List]:
        """解析工具字段数据"""
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
    
    @staticmethod
    def safe_get_value(data: Dict[str, Any], key: str, default: Any = "") -> Any:
        """安全获取字典值"""
        value = data.get(key, default)
        return str(value) if value is not None else default
    
    @staticmethod
    def parse_multi_turn_input(multi_turn_data: Any) -> List[str]:
        """解析多轮对话输入"""
        if not multi_turn_data:
            return []
        
        if isinstance(multi_turn_data, str):
            try:
                if multi_turn_data.strip():
                    parsed = json.loads(multi_turn_data)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], list) and parsed[0]:
                        return [x['content'] for x in parsed[0]]    #暂不支持多list
                        
                    else:
                        return [multi_turn_data.strip()]
                else:
                    return []
            except (json.JSONDecodeError, ValueError):
                return [multi_turn_data.strip()]
        elif isinstance(multi_turn_data, list):
            return multi_turn_data
        else:
            return [str(multi_turn_data)]

class BaseDataLoader(ABC):
    """数据加载器基类"""
    
    def __init__(self, field_mapping: FieldMapping):
        self.field_mapping = field_mapping
    
    @abstractmethod
    def load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """加载数据的抽象方法"""
        pass
    
    def process_record(self, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        """处理单条记录"""
        mapping = self.field_mapping
        
        test_case = {
            'id': self._get_id(record, index),
            'input': DataProcessor.safe_get_value(record, mapping.input_column),
            'expected_output': DataProcessor.safe_get_value(record, mapping.expected_column)
        }
        
        # 处理用例名称
        if mapping.case_name_column and mapping.case_name_column in record:
            test_case['name'] = DataProcessor.safe_get_value(record, mapping.case_name_column)
        
        # 处理多轮对话输入
        if mapping.multi_turn_input_column and mapping.multi_turn_input_column in record:
            multi_turn_input = record[mapping.multi_turn_input_column]
            test_case['input_list'] = DataProcessor.parse_multi_turn_input(multi_turn_input)
        else:
            test_case['input_list'] = []
        
        # 处理可用工具
        if mapping.available_tools_column and mapping.available_tools_column in record:
            available_tools = record[mapping.available_tools_column]
            test_case['available_tools'] = DataProcessor.parse_tools_field(available_tools)
        else:
            test_case['available_tools'] = []
        
        # 处理预期工具调用
        if mapping.expected_tool_call_column and mapping.expected_tool_call_column in record:
            expected_tools = record[mapping.expected_tool_call_column]
            test_case['expected_tools'] = DataProcessor.parse_tools_field(expected_tools)
        else:
            test_case['expected_tools'] = []
        
        # 处理上下文
        if mapping.context_column and mapping.context_column in record:
            test_case['context'] = DataProcessor.safe_get_value(record, mapping.context_column)
        
        # 处理额外字段
        if mapping.extra_fields:
            test_case['extra_fields'] = {}
            for field in mapping.extra_fields:
                if field in record:
                    test_case['extra_fields'][field] = DataProcessor.safe_get_value(record, field)
        
        # 添加其他未明确处理的字段
        # processed_keys = {mapping.input_column, mapping.expected_column, mapping.expected_tool_call_column,
        #                  mapping.available_tools_column, mapping.context_column, mapping.id_column,
        #                  mapping.multi_turn_input_column, mapping.case_name_column} | set(mapping.extra_fields)
        
        # for key, value in record.items():
        #     if key not in processed_keys and key not in test_case:
        #         test_case[key] = DataProcessor.safe_get_value(record, key)
        
        return test_case
    
    def _get_id(self, record: Dict[str, Any], index: int) -> Any:
        """获取记录ID"""
        if self.field_mapping.id_column and self.field_mapping.id_column in record:
            return record[self.field_mapping.id_column]
        return index + 1

class CSVDataLoader(BaseDataLoader):
    """CSV数据加载器"""
    
    def load_data(self, file_path: str = None, csv_file: str = None, **kwargs) -> List[Dict[str, Any]]:
        """加载CSV数据"""
        try:
            # 确定文件路径
            file_path = csv_file or file_path
            if not file_path:
                raise ValueError("必须提供CSV文件路径")
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            
            # 自动检测列名
            self._auto_detect_columns(columns)
            
            # 处理每条记录
            test_cases = []
            for index, row in df.iterrows():
                record = {col: str(row[col]) if pd.notna(row[col]) else "" for col in columns}
                test_case = self.process_record(record, index)
                test_cases.append(test_case)
            
            return test_cases
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV文件 '{file_path}' 不存在")
        except Exception as e:
            logger.error(f"读取CSV文件时发生错误: {str(e)}")
            raise Exception(f"读取CSV文件时发生错误: {str(e)}")
    
    def _auto_detect_columns(self, columns: List[str]):
        """自动检测CSV列名"""
        mapping = self.field_mapping
        
        mapping.input_column = DataProcessor.find_column(columns, mapping.input_column, DataProcessor.INPUT_ALTERNATIVES)
        mapping.expected_column = DataProcessor.find_column(columns, mapping.expected_column, DataProcessor.EXPECTED_ALTERNATIVES)
        mapping.expected_tool_call_column = DataProcessor.find_column(columns, mapping.expected_tool_call_column, DataProcessor.TOOL_CALLS_ALTERNATIVES)
        mapping.available_tools_column = DataProcessor.find_column(columns, mapping.available_tools_column, DataProcessor.TOOLS_ALTERNATIVES)
        mapping.context_column = DataProcessor.find_column(columns, mapping.context_column, DataProcessor.CONTEXT_ALTERNATIVES)

class JSONLDataLoader(BaseDataLoader):
    """JSONL数据加载器"""
    
    def load_data(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """加载JSONL数据"""
        try:
            test_cases = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    try:
                        # 解析JSON行
                        record = json.loads(line)
                        
                        # 自动检测列名（对于第一条记录）
                        if line_num == 1:
                            self._auto_detect_columns(list(record.keys()))
                        
                        test_case = self.process_record(record, line_num - 1)
                        test_cases.append(test_case)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理第{line_num}行时发生错误: {e}")
                        continue
            
            return test_cases
            
        except FileNotFoundError:
            raise FileNotFoundError(f"JSONL文件 '{file_path}' 不存在")
        except Exception as e:
            logger.error(f"读取JSONL文件时发生错误: {str(e)}")
            raise Exception(f"读取JSONL文件时发生错误: {str(e)}")
    
    def _auto_detect_columns(self, columns: List[str]):
        """自动检测JSONL列名"""
        mapping = self.field_mapping
        
        mapping.input_column = DataProcessor.find_column(columns, mapping.input_column, DataProcessor.INPUT_ALTERNATIVES)
        mapping.expected_column = DataProcessor.find_column(columns, mapping.expected_column, DataProcessor.EXPECTED_ALTERNATIVES)
        mapping.expected_tool_call_column = DataProcessor.find_column(columns, mapping.expected_tool_call_column, DataProcessor.TOOL_CALLS_ALTERNATIVES)
        mapping.available_tools_column = DataProcessor.find_column(columns, mapping.available_tools_column, DataProcessor.TOOLS_ALTERNATIVES)
        mapping.context_column = DataProcessor.find_column(columns, mapping.context_column, DataProcessor.CONTEXT_ALTERNATIVES)

class HuggingFaceDataLoader(BaseDataLoader):
    """HuggingFace数据加载器"""
    
    def load_data(self, huggingface_repo: str, config_name: str = None, split: str = 'test', **kwargs) -> List[Dict[str, Any]]:
        """加载HuggingFace数据"""
        try:
            # 从Hugging Face Hub加载数据集
            load_params = {'split': split}
            if config_name:
                load_params['name'] = config_name
            
            dataset_dict = load_dataset(huggingface_repo, **load_params)
            hf_dataset = []
            # 确保我们有一个Dataset对象
            if isinstance(dataset_dict, DatasetDict):
                hf_dataset.append(dataset_dict[split])
            else:
                hf_dataset = dataset_dict
            
            # 处理每条记录
            test_cases = []
            # 获取数据集的特征（列名）
            for hf_dataset in hf_dataset:
                features = hf_dataset.features
                columns = list(features.keys())
            
                # 自动检测列名
                self._auto_detect_columns(columns)
                
                
                for idx, sample in enumerate(hf_dataset):
                    record = {key: sample.get(key) for key in columns}
                    test_case = self.process_record(record, idx)
                    test_cases.append(test_case)
                
            return test_cases
            
        except Exception as e:
            logger.error(f"加载Hugging Face数据集时发生错误: {str(e)}")
            raise Exception(f"加载Hugging Face数据集时发生错误: {str(e)}")
    
    def _auto_detect_columns(self, columns: List[str]):
        """自动检测HuggingFace列名"""
        mapping = self.field_mapping
        
        mapping.input_column = DataProcessor.find_column(columns, mapping.input_column, DataProcessor.INPUT_ALTERNATIVES)
        mapping.expected_column = DataProcessor.find_column(columns, mapping.expected_column, DataProcessor.EXPECTED_ALTERNATIVES)
        mapping.expected_tool_call_column = DataProcessor.find_column(columns, mapping.expected_tool_call_column, DataProcessor.TOOL_CALLS_ALTERNATIVES)
        mapping.available_tools_column = DataProcessor.find_column(columns, mapping.available_tools_column, DataProcessor.TOOLS_ALTERNATIVES)
        mapping.context_column = DataProcessor.find_column(columns, mapping.context_column, DataProcessor.CONTEXT_ALTERNATIVES)

class Dataset:
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
        self.testcases = []
        self._loaders = {
            DataFormat.CSV: CSVDataLoader,
            DataFormat.JSONL: JSONLDataLoader,
            DataFormat.HUGGINGFACE: HuggingFaceDataLoader,
        }

    def load(
        self,
        load_type: str = Field(
            default='csv',
            examples=['csv', 'jsonl', 'trace_local', 'trace_cloud', 'huggingface'],
            description='数据载入类型'
        ),
        **kwargs
    ):
        """
        加载数据集的基本方法
        """
        try:
            # 转换load_type为枚举
            if isinstance(load_type, str):
                load_type = DataFormat(load_type)
            
            columns_mapping = kwargs.get('columns_name', {})
            

            field_mapping = FieldMapping(**columns_mapping)
            
            # 获取对应的数据加载器
            loader_class = self._loaders.get(load_type)
            if not loader_class:
                raise ValueError(f"不支持的数据加载类型: {load_type}")
            
            # 创建加载器实例并加载数据
            loader = loader_class(field_mapping)
            
            
            self.testcases = loader.load_data(**kwargs)
            logger.info(f"成功加载 {len(self.testcases)} 条测试用例")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def get_testcase(self):
        """获取测试用例生成器"""
        for testcase in self.testcases:
            yield testcase

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

if __name__ == "__main__":
    # 测试代码
    dataset = Dataset(name='bytedance-research/veAgentBench', description='test')
    dataset.load(load_type='huggingface', config_name='financial_analysis', split='test[:1]')
    for test in dataset.get_testcase():
        print(test)
