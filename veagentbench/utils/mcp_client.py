import asyncio
import json
import sys
from typing import Dict, Any, Optional
import argparse
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
import httpx
import traceback
import anyio
class HttpStreambleMCPClient:
    def __init__(self):
        self.session = None
        self.endpoint_url = None
        self.api_key = None
        self.headers = {}
        self.read_stream = None
        self.write_stream = None
        self.get_session_id = None
        
    def setup(self, endpoint_url: str, api_key: str = None):
        """设置云端MCP服务器连接参数"""
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        
        # 设置HTTP头
        self.headers = {
            'User-Agent': 'MCP-Cloud-Client/1.0',
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache'
        }
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
            self.headers['X-API-Key'] = api_key
    
    async def connect(self) -> bool:
        """连接到云端MCP服务器"""
        try:
            print(f"正在连接到云端MCP服务器: {self.endpoint_url}")
            
            # 使用SSE客户端连接
            self.streamablehttp_context = streamablehttp_client(self.endpoint_url, headers=self.headers)
            self.read_stream, self.write_stream, self.get_session_id = await self.streamablehttp_context.__aenter__()
            
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            
            # 获取服务器信息
            server_info = await self.session.initialize()
            print(f"成功连接到云端MCP服务器!")
            print(f"服务器名称: {server_info.serverInfo.name}")
            print(f"服务器版本: {server_info.serverInfo.version}")
            
            return True
                
        except Exception as e:
            traceback.print_exc()
            print(f"连接云端MCP服务器失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            if self.streamablehttp_context:
                await self.streamablehttp_context.__aexit__(None, None, None)
                self.streamablehttp_context = None
        except Exception as e:
            print(f"断开连接时出错: {e}")
    
    async def list_tools(self) -> list:
        """获取可用工具列表"""
        if not self.session:
            print("错误: 未连接到服务器")
            return []
            
        try:
            tools = await self.session.list_tools()
            print(tools)
            return tools
        except Exception as e:
            traceback.print_exc()
            print(f"获取工具列表失败: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Optional[types.CallToolResult]:
        """调用云端MCP服务器的工具"""
        if arguments is None:
            arguments = {}
            
        # 检查连接状态，如果已断开则重新连接
        if not self.session or not self.read_stream or not self.write_stream:
            print("检测到连接已断开，正在重新连接...")
            await self.disconnect()  # 确保清理旧连接
            connected = await self.connect()
            if not connected:
                print("错误: 重新连接失败")
                return None
            
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except anyio.ClosedResourceError:
            print("检测到连接已关闭，正在重新连接...")
            await self.disconnect()  # 清理旧连接
            connected = await self.connect()
            if not connected:
                print("错误: 重新连接失败")
                return None
            # 重试一次
            try:
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                print(f"重试调用工具失败: {e}")
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"调用工具失败: {e}")
            traceback.print_exc()
            return None
    
    async def get_tool_info(self, tool_name: str) -> Optional[types.Tool]:
        """获取特定工具的信息"""
        tools = await self.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def format_tool_result(self, result: types.CallToolResult) -> str:
        """格式化工具调用结果"""
        if not result:
            return "无结果"
            
        output_parts = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                output_parts.append(content.text)
            elif isinstance(content, types.ImageContent):
                output_parts.append(f"[图片: {content.mime_type}]")
            else:
                output_parts.append(str(content))
                
        return "\n".join(output_parts)


async def interactive_standard_cloud_client(endpoint_url: str, api_key: str = None)-> HttpStreambleMCPClient:
    """标准云端MCP服务器的交互式客户端"""
    client = HttpStreambleMCPClient()
    client.setup(endpoint_url, api_key)
    
    print(f"标准MCP云端客户端 - 交互式模式")
    print(f"正在连接到云端服务器: {endpoint_url}")
    
    connected = await client.connect()
    if not connected:
        return
    
    print(f"\n已连接到标准云端MCP服务器")
    
    return client
