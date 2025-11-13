import pandas as pd
from typing import List, Optional
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

def _load_prompts(
        excel_path: str,
        sheet_name: str,
        input_column: str,
        limit: Optional[int] = None
) -> List[str]:
    """通用的prompt加载函数"""
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 提取input列并去重
    prompts = df[input_column].dropna().unique().tolist()

    # 应用数量限制
    if limit is not None and limit > 0:
        prompts = prompts[:limit]

    return prompts


def load_law_prompts(
        excel_path: str = os.path.join(current_dir, "../../../dataset/agentkit_test_cases.xlsx"),  # 替换为实际Excel路径
        sheet_name: str = "法律援助客服",
        input_column: str = "input",  # Excel中input字段的列名
        limit: Optional[int] = None
) -> List[str]:
    """
    从Excel文件读取法律援助prompt列表

    Args:
        excel_path: Excel文件路径
        sheet_name: 工作表名称
        input_column: 存储prompt的列名
        limit: 限制返回数量，None表示返回全部

    Returns:
        prompt列表
    """
    return _load_prompts(excel_path, sheet_name, input_column, limit)


def load_deep_research_prompts(
        excel_path: str = os.path.join(current_dir, "../dataset/agentkit_test_cases_1.xlsx"),  # 替换为实际Excel路径
        sheet_name: str = "财务分析",
        input_column: str = "输入（用户查询指令）",  # Excel中input字段的列名
        limit: Optional[int] = None
) -> List[str]:
    """
    从Excel文件读取深度研究prompt列表

    Args:
        excel_path: Excel文件路径
        sheet_name: 工作表名称
        input_column: 存储prompt的列名
        limit: 限制返回数量，None表示返回全部

    Returns:
        prompt列表
    """
    return _load_prompts(excel_path, sheet_name, input_column, limit)


# 测试代码（可选）
if __name__ == "__main__":
    test_prompts = load_deep_research_prompts(limit=5)
    print(f"Loaded {len(test_prompts)} prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. {prompt[:50]}...")  # 打印前50个字符预览