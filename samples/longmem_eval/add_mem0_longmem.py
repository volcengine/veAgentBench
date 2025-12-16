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



import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from threading import Lock
from collections import deque

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient
from mem0.utils.comm import LLMClientFactory,load_json


load_dotenv()


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""
# g_config = load_json(os.getenv("CONFIG_PATH", "config.json"))

HOST=os.getenv("DATABASE_MEM0_BASE_URL")

class QPSLimiter:
    """QPS限制器，用于控制每秒请求数量"""
    def __init__(self, qps_limit=10):
        self.qps_limit = qps_limit
        self.request_times = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """等待直到可以发送请求"""
        with self.lock:
            now = time.time()
            
            # 清理超过1秒的请求时间记录
            while self.request_times and now - self.request_times[0] > 1:
                self.request_times.popleft()
            
            # 如果当前请求数已达到QPS限制，等待
            if len(self.request_times) >= self.qps_limit:
                sleep_time = 1 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    # 再次清理
                    while self.request_times and now - self.request_times[0] > 1:
                        self.request_times.popleft()
            
            # 记录当前请求时间
            self.request_times.append(now)

class MemoryADD:
    def __init__(self, data_path=None, batch_size=50, is_graph=False, qps_limit=10):
        print(f"{batch_size=},{is_graph=},{qps_limit=}")
        # benchmark_config = g_config["benchmark"]
        self.mem0_client = MemoryClient(host=HOST, api_key=os.getenv("MEM0_API_KEY"))
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.qps_limiter = QPSLimiter(qps_limit=qps_limit)
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        # 使用QPS限制器控制请求速率
        self.qps_limiter.wait_if_needed()
        print(user_id)
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    message, user_id=user_id, metadata=metadata, enable_graph=self.is_graph
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, pbar, pbar_lock):
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i: i + self.batch_size]
            res = self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

            with pbar_lock:
                pbar.update(1)

    def process_conversation(self, root_obj, idx, pbar, pbar_lock):
        conversation = root_obj.get("haystack_sessions", None)
        conversation_ids = root_obj.get("haystack_session_ids", None)
        conversation_dates = root_obj.get("haystack_dates", None)
        question_id = root_obj.get("question_id", None)        
        # Iterate through all sessions in the conversation
        for idx, session in enumerate(conversation):
            session_id = conversation_ids[idx]
            date_time_str = conversation_dates[idx]
            timestamp = date_time_str
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
                    "content": text
                })
            
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(question_id, messages, timestamp, pbar, pbar_lock),
            )

            thread_a.start()
            thread_a.join()

    def count_total_batches(self):
        total_batches = 0
        for item in self.data:
            conversation = item["haystack_sessions"]
            for session in conversation:
                num_messages = len(session)
                batches = ceil(num_messages / self.batch_size)
                total_batches += batches  # speaker_a 和 speaker_b 各有一份
        return total_batches

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        total_batches = self.count_total_batches()
        pbar = tqdm(total=total_batches, desc="Adding all memories")
        pbar_lock = Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, item in enumerate(self.data):
                futures.append(
                    executor.submit(self.process_conversation, item, idx, pbar, pbar_lock)
                )
            for future in futures:
                future.result()
        pbar.close()
        
if __name__ == "__main__":
    # 示例：使用QPS限制为5的MemoryADD实例
    qps_limit = 7
    memory_manager = MemoryADD(
        data_path=os.getenv('PATH_TO_LONGMEM_S_DATA'), 
        is_graph=False,
        qps_limit=qps_limit,  # 设置QPS限制为5个请求/秒
        batch_size=os.getenv("MEM0_BATCHSIZE", 4)
        )
    
    print("开始处理对话，QPS限制为%s个请求/秒..."%qps_limit)
    memory_manager.load_data()
    memory_manager.process_all_conversations(max_workers=1)
    
    # # 导入完成后测试搜索功能，建议等待一个晚上的时间，导入的耗时依赖知识抽取的速度，现在mem0线上抽取模型的资源较小，入库较慢
    res = memory_manager.mem0_client.search('''I was looking back at our previous conversation about Caribbean dishes and I was wondering, what was the name of that Jamaican dish you recommended I try with snapper that has fruit in it?''', user_id='778164c6', output_format="v1.1", top_k=5)
    print(res)
