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


"""
HuggingFaceDataset 使用示例

这个模块展示了如何使用 HuggingFaceDataset 类从 Hugging Face Hub 加载数据集
"""

from veagentbench.dataset.huggingface_dataset import HuggingFaceDataset


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 加载SQuAD问答数据集
    dataset = HuggingFaceDataset(
        name="squad_qa",
        description="SQuAD问答数据集",
        dataset_name="squad",
        split="validation"
    )
    
    # 加载数据集
    dataset.load(
        input_column="question",  # 使用question作为输入
        expected_column="answers"  # 使用answers作为预期输出
    )
    
    # 获取数据集信息
    info = dataset.get_dataset_info()
    print(f"数据集名称: {info['dataset_name']}")
    print(f"样本数量: {info['num_examples']}")
    print(f"特征列: {info['features']}")
    print()
    
    # 获取前3个测试用例
    print("前3个测试用例:")
    for i, testcase in enumerate(dataset.get_testcase()):
        if i >= 3:
            break
        print(f"测试用例 {i+1}:")
        print(f"  问题: {testcase.get('input', '')[:100]}...")
        print(f"  答案: {str(testcase.get('expected_output', ''))[:100]}...")
        print()


def example_with_config():
    """使用配置文件的数据集示例"""
    print("=== 使用配置文件的数据集示例 ===")
    
    # 加载GLUE数据集的CoLA子集
    dataset = HuggingFaceDataset(
        name="glue_cola",
        description="GLUE CoLA语法正确性数据集",
        dataset_name="glue",
        config_name="cola",  # CoLA是GLUE的一个子任务
        split="validation"
    )
    
    # 加载数据集
    dataset.load(
        input_column="sentence",
        expected_column="label"
    )
    
    info = dataset.get_dataset_info()
    print(f"数据集: {info['dataset_name']} - cola")
    print(f"样本数量: {info['num_examples']}")
    print()
    
    # 获取前2个测试用例
    print("前2个测试用例:")
    for i, testcase in enumerate(dataset.get_testcase()):
        if i >= 2:
            break
        print(f"测试用例 {i+1}:")
        print(f"  句子: {testcase.get('input', '')}")
        print(f"  标签: {testcase.get('expected_output', '')} (0=不正确, 1=正确)")
        print()


def example_custom_columns():
    """自定义列映射示例"""
    print("=== 自定义列映射示例 ===")
    
    # 加载AG News数据集
    dataset = HuggingFaceDataset(
        name="ag_news_classification",
        description="AG News新闻分类数据集",
        dataset_name="ag_news",
        split="test[:20]"  # 只加载前20个测试样本
    )
    
    # 加载数据集，自定义列映射
    dataset.load(
        input_column="text",           # 新闻文本作为输入
        expected_column="label",       # 类别标签作为预期输出
        id_column=None                 # 使用默认的索引作为ID
    )
    
    info = dataset.get_dataset_info()
    print(f"数据集: {info['dataset_name']}")
    print(f"样本数量: {info['num_examples']}")
    print()
    
    # 获取前2个测试用例
    print("前2个测试用例:")
    for i, testcase in enumerate(dataset.get_testcase()):
        if i >= 2:
            break
        print(f"测试用例 {i+1}:")
        print(f"  新闻: {testcase.get('input', '')[:100]}...")
        print(f"  类别: {testcase.get('expected_output', '')} (1=World, 2=Sports, 3=Business, 4=Sci/Tech)")
        print()


def example_filtering():
    """数据过滤示例"""
    print("=== 数据过滤示例 ===")
    
    # 加载一个小型数据集
    dataset = HuggingFaceDataset(
        name="small_dataset",
        description="小型测试数据集",
        dataset_name="ag_news",
        split="test[:50]"  # 加载前50个样本
    )
    
    dataset.load(
        input_column="text",
        expected_column="label"
    )
    
    # 过滤出特定类别的样本（例如科技新闻，类别4）
    def is_tech_news(testcase):
        return testcase.get('expected_output') == '4'
    
    tech_news = dataset.filter_testcases(is_tech_news)
    
    print(f"总样本数: {len(list(dataset.get_testcase()))}")
    print(f"科技新闻数量: {len(tech_news)}")
    print()
    
    # 显示前2个科技新闻
    print("前2个科技新闻:")
    for i, testcase in enumerate(tech_news):
        if i >= 2:
            break
        print(f"新闻 {i+1}: {testcase.get('input', '')[:100]}...")
        print()


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_with_config()
        example_custom_columns()
        example_filtering()
        
        print("=== 所有示例执行完成 ===")
        print("HuggingFaceDataset 类可以方便地从 Hugging Face Hub 加载各种数据集")
        print("支持自动列检测、自定义列映射、数据过滤等功能")
        
    except Exception as e:
        print(f"执行示例时出错: {e}")
        print("请确保网络连接正常，并且已安装必要的依赖包:")
        print("pip install datasets")
