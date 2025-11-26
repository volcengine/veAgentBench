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


import json
from typing import Optional, List, Dict, Any, Union
from datasets import load_dataset, DatasetDict
from veagentbench.dataset.dataset import Dataset


class HuggingFaceDataset(Dataset):
    """
    通过Hugging Face datasets库加载数据集的类
    专门用于从Hugging Face Hub加载数据集
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        dataset_name: str,
        split: str = "train",
        config_name: Optional[str] = None
    ):
        """
        初始化HuggingFace数据集
        
        Args:
            name: 数据集名称
            description: 数据集描述
            dataset_name: Hugging Face Hub上的数据集名称 (如: 'squad', 'glue', 'cnndm')
            split: 数据集分割 (如: 'train', 'test', 'validation')
            config_name: 数据集配置名称，这里指代为subsets名称
        """
        super().__init__(name, description)
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.hf_dataset = None
    
    def load(
        self,
        load_type: str = "huggingface",
        **kwargs
    ):
        """
        加载数据集
        
        Args:
            load_type: 加载类型，默认为'huggingface'
            **kwargs: 传递给Hugging Face load_dataset的其他参数
        """
        if load_type == "huggingface":
            self.testcases = self.from_huggingface(**kwargs)
        else:
            # 如果load_type不是huggingface，调用父类的load方法
            super().load(load_type=load_type, **kwargs)
    
    def from_huggingface(
        self,
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
            if self.config_name:
                dataset_dict = load_dataset(
                    self.dataset_name, 
                    self.config_name, 
                    split=self.split,
                    **load_kwargs
                )
            else:
                dataset_dict = load_dataset(
                    self.dataset_name, 
                    split=self.split,
                    **load_kwargs
                )
            
            # 确保我们有一个Dataset对象
            if isinstance(dataset_dict, DatasetDict):
                self.hf_dataset = dataset_dict[self.split]
            else:
                self.hf_dataset = dataset_dict
            
            # 获取数据集的特征（列名）
            features = self.hf_dataset.features
            
            # 自动检测列名（如果提供的列名不存在）
            input_column = self._find_column(features, input_column, ["input", "question", "text", "prompt"])
            expected_column = self._find_column(features, expected_column, ["answer", "output", "target", "label"])
            expected_tool_call_column = self._find_column(features, expected_tool_call_column, 
                                                         ["tool_calls", "expected_tools", "expected_tool_calls"])
            available_tools_column = self._find_column(features, available_tools_column, ["tools", "available_tools"])
            context_column = self._find_column(features, context_column, ["context", "passage", "document"])
            
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
                    test_case['available_tools'] = self._parse_tools_field(available_tools)
                
                # 处理预期工具调用
                if expected_tool_call_column and expected_tool_call_column in sample:
                    expected_tools = sample[expected_tool_call_column]
                    test_case['expected_tools'] = self._parse_tools_field(expected_tools)
                
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
    
    def _find_column(self, features, preferred_column: str, fallback_columns: List[str]) -> Optional[str]:
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
    
    def _parse_tools_field(self, tools_data: Any) -> Union[List[Dict], str, List]:
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
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            包含数据集信息的字典
        """
        if self.hf_dataset is None:
            return {"error": "数据集尚未加载"}
        
        info = {
            "dataset_name": self.dataset_name,
            "num_examples": len(self.hf_dataset),
            "features": list(self.hf_dataset.features.keys()),
            "split": self.split
        }
        
        # 添加数据集描述（如果可用）
        if hasattr(self.hf_dataset, 'info') and self.hf_dataset.info.description:
            info["description"] = self.hf_dataset.info.description
        
        return info
    
    def get_testcase(self):
        """
        生成器方法，逐个返回测试用例
        
        Yields:
            测试用例字典
        """
        if not hasattr(self, 'testcases') or not self.testcases:
            raise ValueError("数据集尚未加载，请先调用load()方法")
        
        for testcase in self.testcases:
            yield testcase
    
    def filter_testcases(self, condition: callable) -> List[Dict[str, Any]]:
        """
        根据条件过滤测试用例
        
        Args:
            condition: 过滤函数，接收测试用例字典，返回True/False
            
        Returns:
            符合条件的测试用例列表
        """
        if not hasattr(self, 'testcases') or not self.testcases:
            return []
        
        return [tc for tc in self.testcases if condition(tc)]


# 使用示例
if __name__ == "__main__":
    # 示例1: 从Hugging Face Hub加载SQuAD数据集
    try:
        dataset = HuggingFaceDataset(
            name="bytedance-research/veAgentBench",
            description="SQuAD问答数据集",
            dataset_name="bytedance-research/veAgentBench",
            config_name='educational_tutoring'
            # split="validation"
        )
        dataset.load()
        
        info = dataset.get_dataset_info()
        print(f"数据集信息:")
        print(f"  数据集名称: {info['dataset_name']}")
        print(f"  样本数量: {info['num_examples']}")
        print(f"  特征列: {info['features']}")
        print(f"  分割: {info['split']}")
        
        # 获取前2个测试用例
        print(f"\n前2个测试用例:")
        for i, testcase in enumerate(dataset.get_testcase()):
            if i >= 2:
                break
            print(f"测试用例 {i+1}:")
            print(f"  输入: {testcase.get('input', '')[:100]}...")
            print(f"  预期输出: {testcase.get('expected_output', '')[:100]}...")
            print()
            
    except Exception as e:
        print(f"加载SQuAD数据集失败: {e}")
    
    # 示例2: 从Hugging Face Hub加载GLUE数据集（需要配置）
    try:
        dataset = HuggingFaceDataset(
            name="glue_cola",
            description="GLUE CoLA数据集",
            dataset_name="glue",
            config_name="cola",
            split="validation"
        )
        dataset.load(
            input_column="sentence",
            expected_column="label"
        )
        
        info = dataset.get_dataset_info()
        print(f"GLUE CoLA数据集信息:")
        print(f"  数据集名称: {info['dataset_name']}")
        print(f"  配置名称: cola")
        print(f"  样本数量: {info['num_examples']}")
        print(f"  分割: {info['split']}")
        
    except Exception as e:
        print(f"加载GLUE数据集失败: {e}")
