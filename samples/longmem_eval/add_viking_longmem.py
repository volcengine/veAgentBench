#!/usr/bin/env python3
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

import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from vikingdb.memory import VikingMem
from vikingdb import APIKey
from threading import Lock
import re
import datetime

API_KEY = os.getenv("VIKING_MEMORY_API_KEY", "2GFTZH07D00ER9ZXRFZTJGZPWRYSS1ADJDR03S55RN4M1RVWM5T060R30C1G6CTK0")
client = VikingMem(
    host = "api-knowledgebase.mlp.cn-beijing.volces.com",
    region = "cn-beijing",
    auth= APIKey(api_key=API_KEY),
    scheme = "http",
)



collection = client.get_collection(
    collection_name=os.getenv("DATABASE_VIKINGMEM_COLLECTION"),  # 替换为你的记忆库名称
    project_name="default"
)

def add_session(session_id, messages, timestamp=None):
    # Use the provided timestamp if available, otherwise use current time
    now_ts = timestamp if timestamp is not None else int(time.time() * 1000)
    result = collection.add_session(
        session_id=session_id,
        messages=messages,
        metadata = {
            "default_user_id": "user",
            "default_user_name": "user",
            "default_assistant_id": "assistant",
            "default_assistant_name": "assistant",
            "time": now_ts,
        }
    )
    return result

class QPSController:
    """QPS 控制器，用于限制每秒请求数量"""
    def __init__(self, max_qps=10):
        self.max_qps = max_qps
        self.request_times = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        """如果需要，等待以保持 QPS 限制"""
        with self.lock:
            now = time.time()
            # 清理超过1秒的旧请求记录
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # 如果当前 QPS 已达到限制，等待
            if len(self.request_times) >= self.max_qps:
                # 计算需要等待的时间
                oldest_request = min(self.request_times)
                wait_time = 1.0 - (now - oldest_request)
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    # 重新清理请求记录
                    self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # 记录当前请求时间
            self.request_times.append(now)

def add_session_with_error_handling(session_data):
    """包装add_session函数，添加错误处理、重试机制和结果返回"""
    session_id, messages, timestamp, qps_controller = session_data
    
    # 重试配置
    max_retries = 3  # 最大重试次数
    base_retry_delay = 1.0  # 基础重试延迟（秒）
    max_retry_delay = 30.0  # 最大重试延迟（秒）
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # QPS 控制
            if qps_controller:
                qps_controller.wait_if_needed()
            
            result = add_session(session_id, messages, timestamp)
            if retry_count > 0:
                print(f"✓ Session {session_id} succeeded after {retry_count} retries")
            
            return {
                'success': True,
                'session_id': session_id,
                'message_count': len(messages),
                'result': result,
                'retries': retry_count
            }
            
        except Exception as e:
            last_error = str(e)
            error_type = type(e).__name__
            
            # 判断是否需要重试
            should_retry = True
            
            # 对于某些特定错误类型，不重试
            non_retryable_errors = [
                'ValidationError',  # 参数验证错误
                'AuthenticationError',  # 认证错误
                'AuthorizationError',  # 授权错误
                'NotFound',  # 资源不存在
                'BadRequest',  # 请求格式错误
                'PermissionError'  # 权限错误（Python中的认证错误）
            ]
            
            if error_type in non_retryable_errors:
                should_retry = False
                print(f"✗ Session {session_id} failed with non-retryable error: {error_type} - {last_error}")
            
            if should_retry and retry_count < max_retries:
                # 指数退避延迟
                retry_delay = min(
                    base_retry_delay * (2 ** retry_count),  # 指数增长
                    max_retry_delay  # 不超过最大延迟
                )
                
                print(f"⚠ Session {session_id} failed (attempt {retry_count + 1}/{max_retries + 1}): {last_error}")
                print(f"  Retrying in {retry_delay:.1f} seconds...")
                
                time.sleep(retry_delay)
                retry_count += 1
            else:
                # 达到最大重试次数或不需要重试
                if should_retry:
                    print(f"✗ Session {session_id} failed after {max_retries + 1} attempts: {last_error}")
                break
    
    # 所有重试都失败
    return {
        'success': False,
        'session_id': session_id,
        'error': last_error,
        'retries': retry_count
    }

def parse_longmemeval_timestamp(timestamp_str):
    """
    解析LongMemEval格式的时间戳
    格式: "2023/05/20 (Sat) 02:21"
    返回: datetime对象
    """
    # 移除星期信息，只保留日期和时间
    # 使用正则表达式提取日期和时间部分
    pattern = r'(\d{4}/\d{2}/\d{2}) \([A-Za-z]+\) (\d{2}:\d{2})'
    match = re.match(pattern, timestamp_str)
    
    if not match:
        raise ValueError(f"无法解析时间戳格式: {timestamp_str}")
    
    date_part = match.group(1)  # "2023/05/20"
    time_part = match.group(2)  # "02:21"
    
    # 组合成完整的日期时间字符串
    datetime_str = f"{date_part} {time_part}"
    
    # 解析为datetime对象
    dt = datetime.datetime.strptime(datetime_str, "%Y/%m/%d %H:%M")
    
    
    
    return timestamp_to_milliseconds(dt)

def timestamp_to_milliseconds(dt):
    """将datetime对象转换为Unix时间戳（毫秒）"""
    return int(dt.timestamp() * 1000)

def prepare_session_data(data, qps_controller=None):
    """准备所有会话数据用于并发处理"""
    all_sessions = []
    
    for root_idx, root_obj in enumerate(data):
        conversation = root_obj.get("haystack_sessions", None)
        conversation_ids = root_obj.get("haystack_session_ids", None)
        conversation_dates = root_obj.get("haystack_dates", None)
        question_id = root_obj.get("question_id", None)
        if not conversation:
            continue  # Skip objects without conversation
        
        # Iterate through all sessions in the conversation
        for idx, session in enumerate(conversation):
            session_id = conversation_ids[idx]
            date_time_str = conversation_dates[idx]
            timestamp = parse_longmemeval_timestamp(date_time_str)
            messages = []
            for msg in session:
                text = msg.get("content", "")
                role = msg.get("role", "")
                if role == "user":
                    speaker = question_id
                else:
                    speaker = "assistant"
                # Skip messages without speaker or text
                if not speaker or not text:
                    continue
                
                # Set all messages to user role, use speaker name for role_id and role_name
                messages.append({
                    "role": role,
                    "role_id": speaker,
                    "role_name": speaker,
                    "content": text
                })
            
            # 添加到会话列表中，包含 QPS 控制器
            all_sessions.append((session_id, messages, timestamp, qps_controller))
    
    return all_sessions

# 重试配置参数（可以根据需要调整）
RETRY_CONFIG = {
    'max_retries': 3,           # 最大重试次数
    'base_retry_delay': 1.0,    # 基础重试延迟（秒）
    'max_retry_delay': 30.0,    # 最大重试延迟（秒）
    'enable_retry': True        # 是否启用重试机制
}

def add_session_with_error_handling_and_retry(session_data, retry_config=None):
    """包装add_session函数，添加错误处理、重试机制和结果返回"""
    if retry_config is None:
        retry_config = RETRY_CONFIG
    
    session_id, messages, timestamp, qps_controller = session_data
    
    # 获取重试配置
    max_retries = retry_config.get('max_retries', 3)
    base_retry_delay = retry_config.get('base_retry_delay', 1.0)
    max_retry_delay = retry_config.get('max_retry_delay', 30.0)
    enable_retry = retry_config.get('enable_retry', True)
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # QPS 控制
            if qps_controller:
                qps_controller.wait_if_needed()
            
            result = add_session(session_id, messages, timestamp)
            if retry_count > 0:
                print(f"✓ Session {session_id} succeeded after {retry_count} retries")
            
            return {
                'success': True,
                'session_id': session_id,
                'message_count': len(messages),
                'result': result,
                'retries': retry_count
            }
            
        except Exception as e:
            last_error = str(e)
            error_type = type(e).__name__
            
            # 判断是否需要重试
            should_retry = enable_retry
            
            # 对于某些特定错误类型，不重试
            non_retryable_errors = [
                'ValidationError',  # 参数验证错误
                'AuthenticationError',  # 认证错误
                'AuthorizationError',  # 授权错误
                'NotFound',  # 资源不存在
                'BadRequest'  # 请求格式错误
            ]
            
            if error_type in non_retryable_errors:
                should_retry = False
                print(f"✗ Session {session_id} failed with non-retryable error: {error_type} - {last_error}")
            
            if should_retry and retry_count < max_retries:
                # 指数退避延迟
                retry_delay = min(
                    base_retry_delay * (2 ** retry_count),  # 指数增长
                    max_retry_delay  # 不超过最大延迟
                )
                
                print(f"⚠ Session {session_id} failed (attempt {retry_count + 1}/{max_retries + 1}): {last_error}")
                print(f"  Retrying in {retry_delay:.1f} seconds...")
                
                time.sleep(retry_delay)
                retry_count += 1
            else:
                # 达到最大重试次数或不需要重试
                if should_retry:
                    print(f"✗ Session {session_id} failed after {max_retries + 1} attempts: {last_error}")
                break
    
    # 所有重试都失败
    return {
        'success': False,
        'session_id': session_id,
        'error': last_error,
        'retries': retry_count
    }

# 保持向后兼容性
def add_session_with_error_handling(session_data):
    """向后兼容的函数，使用默认重试配置"""
    return add_session_with_error_handling_and_retry(session_data, RETRY_CONFIG)

if __name__ == "__main__":
    print("Viking Memory Add Session Messages Example")
    print("Reading locomo10.json file...")
    
    # Read the locomo10.json file
    json_file_path = os.getenv('PATH_TO_LONGMEM_S_DATA'), 
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Option to add only a single session for testing
    test_single = False  # Set to True to test with only one conversation
    
    # QPS 控制参数
    max_qps = 9  # 每秒最大请求数，可以根据需要调整
    qps_controller = QPSController(max_qps=max_qps)
    
    # 重试配置（可以根据需要调整）
    retry_config = RETRY_CONFIG.copy()
    # retry_config['max_retries'] = 5  # 可以在这里修改重试次数
    # retry_config['enable_retry'] = False  # 可以禁用重试
    
    print(f"QPS limit set to: {max_qps} requests per second")
    print(f"Retry enabled: {retry_config['enable_retry']}")
    print(f"Max retries: {retry_config['max_retries']}")
    
    # 准备所有会话数据
    print("Preparing session data for concurrent processing...")
    all_sessions = prepare_session_data(data, qps_controller)
    total_sessions = len(all_sessions)
    print(f"Total sessions to process: {total_sessions}")
    
    # 测试单个会话
    if test_single and all_sessions:
        print("Testing with single session...")
        session_data = all_sessions[0]
        result = add_session_with_error_handling(session_data)
        if result['success']:
            print(f"Single session test completed! Added session {result['session_id']} with {result['message_count']} messages")
        else:
            print(f"Single session test failed: {result['error']}")
        exit()
    
    # 使用线程池进行并发处理
    print("Starting concurrent processing with QPS control...")
    successful_sessions = 0
    failed_sessions = 0
    max_workers = 3  # 可以根据需要调整并发线程数，QPS 控制会确保总体速率
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，使用新的重试函数
        future_to_session = {
            executor.submit(add_session_with_error_handling_and_retry, session_data, retry_config): session_data[0] 
            for session_data in all_sessions
        }
        
        # 处理完成的任务
        start_time = time.time()
        total_retries = 0  # 统计总重试次数
        
        for future in as_completed(future_to_session):
            session_id = future_to_session[future]
            try:
                result = future.result()
                print('reqid: %s'%result.get('result').get('request_id'))
                if result['success']:
                    successful_sessions += 1
                    retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
                    print(f"✓ Added session {result['session_id']} with {result['message_count']} messages{retry_info}")
                    total_retries += result['retries']
                else:
                    failed_sessions += 1
                    retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
                    print(f"✗ Failed to add session {result['session_id']}: {result['error']}{retry_info}")
                
                # 进度报告
                total_processed = successful_sessions + failed_sessions
                if total_processed % 100 == 0:
                    elapsed_time = time.time() - start_time
                    actual_qps = total_processed / elapsed_time if elapsed_time > 0 else 0
                    print(f"Progress: {total_processed}/{total_sessions} sessions processed "
                          f"({successful_sessions} successful, {failed_sessions} failed)")
                    print(f"Current QPS: {actual_qps:.2f}, Target QPS: {max_qps}")
                    if total_retries > 0:
                        print(f"Total retries so far: {total_retries}")
                    
            except Exception as e:
                failed_sessions += 1
                print(f"✗ Exception occurred for session {session_id}: {str(e)}")
    
    total_time = time.time() - start_time
    actual_qps = total_sessions / total_time if total_time > 0 else 0
    
    print(f"\nConcurrent processing completed!")
    print(f"Total sessions: {total_sessions}")
    print(f"Successful: {successful_sessions}")
    print(f"Failed: {failed_sessions}")
    print(f"Total retries: {total_retries}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Actual QPS: {actual_qps:.2f}")
    print(f"Target QPS: {max_qps}")
    
    # 计算重试成功率
    if total_retries > 0:
        retry_success_rate = (total_retries / (total_sessions + total_retries)) * 100
        print(f"Retry success rate: {retry_success_rate:.1f}%")
    
    print("All sessions processed!")
