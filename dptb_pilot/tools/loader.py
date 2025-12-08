import asyncio
import json
from mcp import ClientSession, McpError
from mcp.client.sse import sse_client
from typing import List, Dict, Any


async def get_mcp_server_tools(server_url: str) -> List[Dict[str, Any]]:
    """
    从 MCP 服务器获取所有工具信息

    Args:
        server_url: MCP 服务器的 SSE 端点 URL

    Returns:
        工具信息列表
    """
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化会话
                init_result = await session.initialize()
                print(f"✅ 成功连接到 MCP 服务器")
                print(f"   服务器信息: {init_result}")

                # 获取工具列表
                tools_result = await session.list_tools()
                tools = tools_result.tools

                print(f"📋 发现 {len(tools)} 个可用工具")

                # 转换为字典格式便于处理
                tools_info = []
                for tool in tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": getattr(tool, 'inputSchema', {}),
                        "parameters": getattr(tool, 'parameters', {})
                    }
                    tools_info.append(tool_info)

                return tools_info

    except McpError as e:
        print(f"❌ MCP 协议错误: {e}")
        raise
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        raise


def display_tools_info(tools_info: List[Dict[str, Any]]):
    """格式化显示工具信息"""
    print("\n" + "=" * 80)
    print("MCP 服务器工具详情")
    print("=" * 80)

    for i, tool in enumerate(tools_info, 1):
        print(f"\n{i}. 🛠️  {tool['name']}")
        print(f"   📝 {tool['description']}")

        # 显示输入模式
        if tool['input_schema']:
            print(f"   📋 输入模式:")
            print(f"      {json.dumps(tool['input_schema'], indent=6)}")

        print("-" * 80)

