import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from vikingdb.memory import VikingMem
from vikingdb import APIKey
from threading import Lock

API_KEY = os.getenv("MEMORY_API_KEY", "")
client = VikingMem(
    host = "api-knowledgebase.mlp.cn-beijing.volces.com",
    region = "cn-beijing",
    auth= APIKey(api_key=API_KEY),
    scheme = "http",
)

collection = client.get_collection(
    collection_name="locomo_pe",  # 替换为你的记忆库名称
    project_name="default"
)

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

def add_session(session_id, messages, timestamp=None):
    """原始的 add_session 函数"""
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

def process_conversation_with_retry(item, idx, qps_controller=None, retry_config=None):
    """处理对话，使用重试机制和QPS控制"""
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    speaker_a_user_id = f"{speaker_a}_{idx}"
    speaker_b_user_id = f"{speaker_b}_{idx}"

    # delete all memories for the two users
    # self.mem0_client.delete_all(user_id=speaker_a_user_id)
    # self.mem0_client.delete_all(user_id=speaker_b_user_id)

    for key in conversation.keys():
        if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
            continue

        date_time_key = key + "_date_time"
        
        date_time_str = conversation[date_time_key]
        dt_part = date_time_str.replace("on ", "")
        # Then parse using strptime
        dt = datetime.strptime(dt_part, "%I:%M %p %d %B, %Y")
        # Convert to timestamp (milliseconds)
        timestamp = int(dt.timestamp() * 1000)
        
        chats = conversation[key]
        session_id =  f"session_{idx:02d}_{int(key.split('_')[-1]):02d}"
        messages = []
        for chat in chats:
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            
            # Skip messages without speaker or text
            if not speaker or not text:
                continue
            
            # Set all messages to user role, use speaker name for role_id and role_name
            messages.append({
                "role": "user",
                "role_id": speaker,
                "role_name": speaker,
                "content": text
            })
        
        # 使用带重试的函数添加会话
        session_data = (session_id, messages, timestamp, qps_controller)
        result = add_session_with_error_handling_and_retry(session_data, retry_config)
        
        if result['success']:
            retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
            print(f"✓ Added session {session_id} with {len(messages)} messages{retry_info}")
        else:
            retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
            print(f"✗ Failed to add session {session_id}: {result['error']}{retry_info}")

# 保持向后兼容性
def process_conversation(item, idx):
    """向后兼容的函数，使用默认配置"""
    return process_conversation_with_retry(item, idx)



def process_all_conversations_with_qps_and_retry(data, max_workers=3, max_qps=5, retry_config=None):
    """使用并发、QPS控制和重试机制处理所有对话"""
    if retry_config is None:
        retry_config = RETRY_CONFIG.copy()
    
    # QPS 控制器
    qps_controller = QPSController(max_qps=max_qps)
    
    print(f"QPS limit set to: {max_qps} requests per second")
    print(f"Retry enabled: {retry_config['enable_retry']}")
    print(f"Max retries: {retry_config['max_retries']}")
    print(f"Concurrent workers: {max_workers}")
    
    # 准备所有会话数据
    all_sessions = []
    for root_idx, root_obj in enumerate(data):
        conversation = root_obj.get("conversation", None)
        if not conversation:
            continue
        
        # 为每个对话创建会话数据
        idx = root_idx
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            date_time_str = conversation[date_time_key]
            dt_part = date_time_str.replace("on ", "")
            dt = datetime.strptime(dt_part, "%I:%M %p %d %B, %Y")
            timestamp = int(dt.timestamp() * 1000)
            
            chats = conversation[key]
            session_id = f"session_{idx:02d}_{int(key.split('_')[-1]):02d}"
            
            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}", "role_id": speaker_a_user_id, "role_name": speaker_a})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}", "role_id": speaker_a_user_id, "role_name": speaker_a})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}", "role_id": speaker_b_user_id, "role_name": speaker_b})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}", "role_id": speaker_b_user_id, "role_name": speaker_b})
            
            # 添加到会话列表
            all_sessions.append((session_id, messages, timestamp, qps_controller))
            all_sessions.append((session_id + "_reverse", messages_reverse, timestamp, qps_controller))
    
    total_sessions = len(all_sessions)
    print(f"Total sessions to process: {total_sessions}")
    
    if total_sessions == 0:
        print("No sessions to process")
        return 0
    
    # 使用线程池进行并发处理
    print("Starting concurrent processing with QPS control and retry mechanism...")
    successful_sessions = 0
    failed_sessions = 0
    total_retries = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_session = {
            executor.submit(add_session_with_error_handling_and_retry, session_data, retry_config): session_data[0] 
            for session_data in all_sessions
        }
        
        # 处理完成的任务
        start_time = time.time()
        
        for future in as_completed(future_to_session):
            session_id = future_to_session[future]
            try:
                result = future.result()
                if result['success']:
                    successful_sessions += 1
                    retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
                    print(f"✓ Added session {result['session_id']} with {result['message_count']} messages{retry_info}")
                    total_retries += result['retries']
                else:
                    failed_sessions += 1
                    retry_info = f" (retried {result['retries']} times)" if result['retries'] > 0 else ""
                    print(f"✗ Failed to add session {result['session_id']}: {result['error']}{retry_info}")
                    total_retries += result['retries']
                
                # 进度报告
                total_processed = successful_sessions + failed_sessions
                if total_processed % 50 == 0:
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
    
    return successful_sessions

if __name__ == "__main__":
    print("Viking Memory Add Session Messages Example")
    print("Reading locomo10.json file...")
    
    # Read the locomo10.json file
    json_file_path = os.path.join(os.path.dirname(__file__), "locomo10.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 配置参数
    max_workers = 3  # 并发线程数
    max_qps = 9  # QPS限制
    retry_config = RETRY_CONFIG.copy()
    # retry_config['max_retries'] = 5  # 可以修改重试次数
    # retry_config['enable_retry'] = False  # 可以禁用重试
    
    # 处理所有对话
    successful_count = process_all_conversations_with_qps_and_retry(
        data, 
        max_workers=max_workers, 
        max_qps=max_qps, 
        retry_config=retry_config
    )
    
    print(f"Total successful sessions: {successful_count}")
    print("All sessions processed!")
