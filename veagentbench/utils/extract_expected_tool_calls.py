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
import re
from veagentbench.evals.deepeval.test_case import ToolCall


def parse_expected_tool_calls(text: str, text_expect_output: str) -> List[Dict[str, Any]]:
    """
    从自由格式字符串中解析 expected_tool_calls 列表。
    支持：
    - 每行一个工具调用，行首可有编号："1. "、"2. " 等
    - 工具名与内容用中文冒号 "：" 或英文冒号 ":" 分隔
    - 键值参数格式：key="value"，兼容中文引号 “ ” 与英文引号 "
    - 括号说明（如：（提取 “行业”））解析为 extraction_hint
    - 非键值参数的自然语言作为 description/query（当没有键值时作为 query）
    返回结构：
    [
      {
        "tool_name": str,
        "parameters": dict,
        "description": Optional[str],
        "extraction_hint": Optional[str]
      },
      ...
    ]
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    calls: List[Dict[str, Any]] = []

    # 统一引号为英文，便于正则解析
    def normalize_quotes(s: str) -> str:
        return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # 提取括号说明（中文全角括号），如：（提取 “行业”）
    bracket_hint_pattern = re.compile(r"（\s*提取\s*[\"']?([^\"'）]+)[\"']?\s*）")

    # 提取键值参数 key="value"（兼容字母数字下划线key）
    kv_pattern = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")

    for raw_line in lines:
        line = normalize_quotes(raw_line)

        # 去除前缀编号，如 "1. "、"2. "
        line = re.sub(r"^\s*\d+\.\s*", "", line)

        # 按中文冒号或英文冒号分隔工具名与内容
        # 优先中文冒号
        if "：" in line:
            tool_name, rest = line.split("：", 1)
        elif ":" in line:
            tool_name, rest = line.split(":", 1)
        else:
            # 无冒号无法解析为工具调用，跳过
            # 也可能是纯描述，不作为工具
            continue

        tool_name = tool_name.strip()
        rest = rest.strip()

        # 提取括号说明（extraction_hint）
        extraction_hint: Optional[str] = None
        m_hint = bracket_hint_pattern.search(rest)
        if m_hint:
            extraction_hint = m_hint.group(1).strip()
            # 将括号说明从文本中移除，避免干扰参数解析
            rest = bracket_hint_pattern.sub("", rest).strip()

        # 解析键值参数
        parameters: Dict[str, Any] = {}
        for m in kv_pattern.finditer(rest):
            key = m.group(1)
            val = m.group(2)
            parameters[key] = val

        # 移除已匹配键值对后的残余文本，用于 description/query
        rest_without_kv = kv_pattern.sub("", rest).strip(" ，,;")

        description: Optional[str] = None

        if parameters:
            # 存在参数时，剩余文本作为补充描述
            if rest_without_kv:
                description = rest_without_kv
        else:
            # 不存在参数时，将剩余文本作为 query
            if rest_without_kv:
                parameters["query"] = rest_without_kv

        call: Dict[str, Any] = {
            "tool_name": tool_name,
            "parameters": parameters or {},
        }
        if description:
            call["description"] = description
        if extraction_hint:
            call["extraction_hint"] = extraction_hint

        calls.append(call)

    # 合并预期输出：优先通过序号匹配，其次通过工具名匹配
    try:
        expected_map = _parse_expected_outputs(text_expect_output)
    except Exception:
        expected_map = {}

    for idx, call in enumerate(calls, start=1):
        tname = str(call.get("tool_name") or "").strip()
        exp = expected_map.get(idx)
        if exp is None and tname:
            exp = expected_map.get(tname)
        if exp is not None:
            call["expected_output_raw"] = exp

    return calls


def _parse_expected_outputs(text_expect_output: str) -> Dict[Any, str]:
    """
    将“预期工具调用结果”解析为映射：
      - key: 行序号(int) 或 工具名(str)
      - value: 对应的原始文本片段（保持原样，不做结构化解析）
    支持格式：
      1. tool_name：<文本或列表/字典样式字符串>
      2. tool_name: <...>
    """
    mapping: Dict[Any, str] = {}
    if not text_expect_output:
        return mapping

    lines = [ln.strip() for ln in text_expect_output.splitlines() if ln.strip()]

    for raw in lines:
        line = raw
        idx_key = None
        name_key = None
        body = ""

        # 提取前缀编号（如 "1. "）
        if ". " in line:
            num_part, rest = line.split(". ", 1)
        elif "." in line and line.split(".", 1)[0].isdigit():
            num_part, rest = line.split(".", 1)
        else:
            rest = line
            num_part = None

        if num_part and str(num_part).strip().isdigit():
            try:
                idx_key = int(str(num_part).strip())
            except Exception:
                idx_key = None

        # 提取工具名与内容（中英文冒号）
        if "：" in rest:
            name_part, body = rest.split("：", 1)
        elif ":" in rest:
            name_part, body = rest.split(":", 1)
        else:
            name_part, body = rest, ""

        name_key = name_part.strip()
        body = body.strip()

        # 不做 eval，直接保留原始文本，安全且通用
        if idx_key is not None:
            mapping[idx_key] = body
        if name_key:
            mapping[name_key] = body

    return mapping


def turn_tool_dict2toolcall(tool_dict_list: List[Dict[str, Any]]) -> List[ToolCall]:
    """
    将工具调用的字典形式转换为标准的 ToolCall 结构
    """
    tool_list = []
    for tool in tool_dict_list:
        tool_list.append(
            ToolCall(
                name= tool.get("tool_name", "unknown_tool"),
                input_parameters= tool.get("parameters", {}),
                description= tool.get("description", ""),
                output= tool.get("expected_output_raw", ""),
                )
            )
    
    return tool_list

def turn_tool_toolcall2dict(tool_call_list: List[ToolCall]) -> List[Dict[str, Any]]:
    tool_dict_list = []
    for tool in tool_call_list:
        tool_dict_list.append(
            {
                'name': tool.name,
                'input_parameters': tool.input_parameters,
                'description': tool.description,
                'expected_output': tool.output
                
            })
    return tool_dict_list


# 简单自测
if __name__ == "__main__":
    sample = """1. vesearch：2023 年 4 月，国内互联网安全龙头（核心产品安全卫士，参与国家网络安全应急响应）创始人离婚分割近 90 亿元股权，查公司名称、股票代码、事件日期
2. stock_zh_a_hist：symbol="601360"，period="daily"，start_date="20230404"，end_date="20230407"，adjust=""
3. stock_individual_info_em：symbol="601360"（提取 “行业”）
4. stock_board_industry_hist_em：symbol="软件开发"，start_date="20230404"，end_date="20230407"，period="daily"，adjust=""
"""
    sample_expect = """1. vesearch：2023 年 4 月 3 日，三六零（601360）披露《关于股东权益变动的提示性公告》，创始人离婚，分割 8.97 亿股（市值近 90 亿元），公司为国内互联网安全龙头，核心产品含 360 安全卫士
2. stock_zh_a_hist：[{'日期': '2023-04-04', '股票代码': '601360', '开盘': 19.71, '收盘': 20.08}, {'日期': '2023-04-06', '股票代码': '601360', '开盘': 19.2, '收盘': 18.97}]
3. stock_individual_info_em：[{'item': '行业', 'value': '软件开发'}]
4. stock_board_industry_hist_em：[{'日期': '2023-04-04', '开盘': 959.09, '收盘': 960.94}, {'日期': '2023-04-06', '开盘': 946.1, '收盘': 947.17}]
"""
    import json
    print(json.dumps(parse_expected_tool_calls(sample, sample_expect), ensure_ascii=False, indent=2))