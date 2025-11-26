from veagentbench.agents.base_agent import BaseAgent, AgentOutPut
from veadk.utils.logger import get_logger
from veagentbench.utils.tool_result_success import is_tool_execution_success
from pydantic import BaseModel
logger = get_logger(__name__)

class LocalAdkAgent(BaseAgent):
    """
    本地ADK Agent，基于给定的目录中的agent对象，通过runner实现generate_output函数
    """
    
    def __init__(self, agent_dir_path: str, agent_name: str = "local_adk_agent"):
        """
        初始化本地ADK Agent
        
        Args:
            agent_dir_path: 包含agent定义的目录路径
            app_name: 应用名称
        """
        self.agent_dir_path = agent_dir_path
        self.agent_name = agent_name
        self.agent = None
        self.runner = None
        self._load_agent_from_directory()
        
    def _load_agent_from_directory(self):
        """从目录中加载agent对象"""
        import importlib.util
        import sys
        from pathlib import Path
        import os
        
        try:
            # 将目录添加到sys.path
            if str(self.agent_dir_path) not in sys.path:
                sys.path.insert(0, str(self.agent_dir_path))
                logger.info(f"添加路径到sys.path: {self.agent_dir_path}")
            
            # 查找目录中的Python文件
            py_files = list(Path(self.agent_dir_path).glob("*.py"))
            if not py_files:
                raise FileNotFoundError(f"目录 {self.agent_dir_path} 中没有找到Python文件")
            
            # 优先查找包含agent定义的文件
            agent_module = None
            for py_file in py_files:
                if py_file.name.startswith('test_') or py_file.name == '__init__.py':
                    continue  # 跳过测试文件和初始化文件
                    
                try:
                    # 加载模块
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None or spec.loader is None:
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # 检查是否包含agent对象
                    if hasattr(module, 'agent'):
                        self.agent = module.agent
                        logger.info(f"成功从 {py_file} 加载agent: {self.agent.name}")
                        agent_module = module
                        break
                        
                except Exception as e:
                    logger.warning(f"加载文件 {py_file} 失败: {e}")
                    continue
            
            if self.agent is None:
                # 如果没有找到agent对象，尝试第一个成功的模块
                if agent_module is None:
                    raise AttributeError(f"目录 {self.agent_dir_path} 中没有找到包含 'agent' 对象的Python文件")
            
            # 创建runner
            from veadk import Runner
            self.runner = Runner(agent=self.agent, app_name=self.agent_name)
            logger.info("成功创建runner")
            
        except Exception as e:
            logger.error(f"加载agent目录失败: {e}")
            raise
    
    def get_session(self):
        """获取会话ID"""
        from uuid import uuid4
        return str(uuid4())
    
    async def generate_output(
        self,
        prompt: str,
        user_id: str,
        stream: bool = False,
        **kwargs
    ) -> AgentOutPut:
        """通过runner生成输出，使用流式处理"""
        
        session_id = self.get_session()
        tool_called = {}
        final_response = ""
        first_token_duration = 0.0
        end2end_duration = 0.0
        
        import time
        from google.genai.types import Content, Part
        from google.adk.agents import RunConfig
        from google.adk.agents.run_config import StreamingMode
        
        time_start = time.time()
        
        try:
            # 获取会话服务
            session_service = self.runner.short_term_memory.session_service
            
            # 防止会话重新创建
            session = await session_service.get_session(
                app_name=self.agent_name, user_id=user_id, session_id=session_id
            )
            if not session:
                await session_service.create_session(
                    app_name=self.agent_name, user_id=user_id, session_id=session_id
                )
            
            # 创建新消息
            new_message = Content(role="user", parts=[Part(text=prompt)])
            
            # 使用流式处理运行agent
            full_response_text = ""
            first_token_received = False
            
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                # 处理工具调用
                calls = event.get_function_calls()
                if calls:
                    for call in calls:
                        tool_name = call.name
                        arguments = call.args
                        tool_id = call.id
                        tool_called[tool_id] = {
                            "name": tool_name,
                            "input_parameters": arguments,
                        }
                
                # 处理工具响应
                responses = event.get_function_responses()
                if responses:
                    for response in responses:
                        tool_name = response.name
                        result_dict = response.response
                        tool_id = response.id
                        if tool_id in tool_called:
                            if tool_name == 'load_knowledgebase':
                                result = result_dict.get('result', None)
                                if isinstance(result, BaseModel):
                                    result = result.model_dump_json()
                                tool_called[tool_id]["output"] = {'result':result}
                            else:
                                tool_called[tool_id]["output"] = result_dict
                            tool_called[tool_id]["success"] = is_tool_execution_success(result_dict)
                
                # 处理文本内容
                if event.partial and event.content and event.content.parts and event.content.parts[0].text:
                    if not first_token_received:
                        first_token_duration = time.time() - time_start
                        first_token_received = True
                    full_response_text += event.content.parts[0].text
                
                # 处理最终响应
                if event.is_final_response():
                    end2end_duration = time.time() - time_start
                    
                    if event.content and event.content.parts and event.content.parts[0].text:
                        # 如果是流的最后部分，使用累积的文本
                        final_response = full_response_text + (event.content.parts[0].text if not event.partial else "")
                    elif event.actions and event.actions.skip_summarization and event.get_function_responses():
                        # 处理原始工具结果的显示
                        response_data = event.get_function_responses()[0].response
                        final_response = str(response_data)
                    else:
                        final_response = full_response_text
            
            logger.info(f"Agent执行完成，响应长度: {len(final_response)} 字符")
            
            return AgentOutPut(
                first_token_duration=first_token_duration,
                end2end_duration=end2end_duration,
                tool_called=tool_called,
                final_response=final_response,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            import traceback
            traceback.print_exc()
            end2end_duration = time.time() - time_start
            return AgentOutPut(
                first_token_duration=first_token_duration,
                end2end_duration=end2end_duration,
                tool_called=tool_called,
                final_response=f"执行失败: {str(e)}",
                success=False
            )

