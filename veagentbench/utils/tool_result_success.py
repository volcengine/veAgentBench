from typing import Any


ERROR_KEYS = {"error", "err", "exception", "traceback", "message"}
NEGATIVE_TOKENS = {
    "invalid", "not found", "permission denied", "timeout", "error",
    "异常", "失败"
}
NEGATIVE_STATUS = {"error", "failed", "fail", "failure"}
PAYLOAD_KEYS = {"data", "records", "result", "output", "items", "content"}


def _contains_negative_signal(payload: Any) -> bool:
    try:
        # 字典：错误字段、失败状态、success=False、负面词
        if isinstance(payload, dict):
            # 显式错误字段
            if any(k in payload for k in ERROR_KEYS):
                return True
            # 状态字段负面
            status = str(payload.get("status", "")).lower()
            if status in NEGATIVE_STATUS:
                return True
            
            #error字段
            iserror = payload.get("isError", "")
            if iserror == False:
                return False
            elif iserror == True:
                return True
            # success 显式为 False
            if payload.get("success") is False:
                return True
            # 字典文本中包含负面词
            if any(tok in str(payload).lower() for tok in NEGATIVE_TOKENS):
                return True
            return False

        # 列表：若文本中包含负面词，视为失败；否则不直接判负
        if isinstance(payload, list):
            text = " ".join(str(x) for x in payload)
            return any(tok in text.lower() for tok in NEGATIVE_TOKENS)

        # 字符串：包含负面词即失败
        if isinstance(payload, str):
            return any(tok in payload.lower() for tok in NEGATIVE_TOKENS)

        # 其他类型：不判负
        return False
    except Exception:
        # 解析过程中异常，保守视为含负面信号
        return True


def _contains_valid_payload(payload: Any) -> bool:
    # 有效载荷判断：存在非空的典型数据字段或非空列表/字符串
    if payload is None:
        return False

    if isinstance(payload, dict):
        # 典型载荷键存在且非空
        for key in PAYLOAD_KEYS:
            if key in payload and payload.get(key) not in (None, [], {}, "", 0):
                return True
        # 字典本身非空也可视作有效（兜底）
        return bool(payload)

    if isinstance(payload, list):
        return len(payload) > 0

    if isinstance(payload, str):
        return payload.strip() != ""

    # 其他类型：按布尔值兜底
    return bool(payload)


def is_tool_execution_success(output_result: Any) -> bool:
    """
    根据工具执行结果判断是否成功：
    - 若包含错误/失败信号，则为失败
    - 若包含有效载荷，则为成功
    - 否则按内容布尔值兜底
    """
    # 负面信号优先失败
    if _contains_negative_signal(output_result):
        return False

    # 有效载荷即成功
    if _contains_valid_payload(output_result):
        return True

    # 兜底：内容为真视为成功
    return bool(output_result)

