import asyncio

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Optional, Dict, Any


from dptb_pilot.core.logger import get_logger

logger = get_logger(__name__)
async def tool_modify_guardrail(
        tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    from dptb_pilot.core.legacy_main import unmodified_schema_store, pending_events, modified_args_store, target_tools, tools_info
    global unmodified_schema_store
    tool_name = tool.name
    agent_name = tool_context.agent_name # Agent attempting the tool call
    logger.debug(f"--- Callback: tool_modify_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    logger.debug(f"--- Callback: Inspecting args: {args} ---")

    session_id = agent_name[-32:]
    logger.debug(target_tools)

    if tool_name in target_tools:
        schema = zip_tool_schema(tool_name=tool_name,
                                 arguments=args,
                                 tools_dict=tools_info)

        unmodified_schema_store[session_id] = schema

        pending_events[session_id] = asyncio.Event()
        pending_events[session_id] = asyncio.Event()
        logger.info("--- Callback: Wait for the user to click the button to continue execution... ---")
        await pending_events[session_id].wait()  # ⏸ pause until clicked
        logger.info("--- Callback: The user has clicked the button and continues to execute. ---")

        unmodified_schema_store[session_id] = ""
        for k, v in modified_args_store[session_id].items():
            args[k] = v

        logger.info(f"--- Callback: Tool '{tool_name}' Running with modified args: {args}. ---")
    else:
        logger.debug(f"--- Callback: Tool '{tool_name}' is not in the target tools. Allowing. ---")


    # If the checks above didn't return a dictionary, allow the tool to execute
    logger.debug(f"--- Callback: Allowing tool '{tool_name}' to proceed. ---")
    return None # Returning None allows the actual tool function to run

def zip_tool_schema(tool_name, arguments, tools_dict):
    """
    根据工具名称和参数准备工具模式

    Args:
        tool_name: 工具名称
        arguments: 参数字典
        tools_dict: 工具字典列表

    Returns:
        包含agent_input的更新后的工具模式字典，如果未找到工具则返回None
    """
    # 在工具字典中查找匹配的工具
    tool_info = None
    for tool in tools_dict:
        if tool.get('name') == tool_name:
            tool_info = tool.copy()  # 创建副本以避免修改原始数据
            break

    if tool_info is None:
        return None

    # 如果有input_schema且包含properties，则插入agent_input字段
    if 'input_schema' in tool_info and 'properties' in tool_info['input_schema']:
        properties = tool_info['input_schema']['properties']

        for prop_name, prop_value in properties.items():
            if prop_name in arguments:
                # 在属性字典中插入agent_input，保持原有结构
                prop_value = prop_value.copy()  # 创建属性副本
                prop_value['agent_input'] = arguments[prop_name]
                properties[prop_name] = prop_value

    return tool_info

def collect_inputs(schema, _session_id, *values):
    from dptb_pilot.core.legacy_main import pending_events, modified_args_store, modified_schema_store
    if _session_id not in modified_schema_store.keys():
        modified_schema_store[_session_id] = ""
    if len(schema['input_schema']['properties']) == len(values):
        # 根据用户输入生成输出字典，结构与输入类似，并新增 'user_input'
        output = {
            'name': schema['name'],
            'description': schema['description'],
            'input_schema': {'properties': {}},
            'parameters': schema.get('parameters', {})
        }
        for (key, prop), val in zip(schema['input_schema']['properties'].items(), values):
            new_prop = prop.copy()
            new_prop['user_input'] = val
            output['input_schema']['properties'][key] = new_prop

        modified_schema_store[_session_id] = output
        modified_args_store[_session_id] = extract_arguments_from_schema(output)
        pending_events[_session_id].set()
    return modified_schema_store[_session_id]

def extract_arguments_from_schema(tool_schema):
    """
    从工具模式中提取user_input并生成arguments字典

    Args:
        tool_schema: 包含input_schema的工具模式字典

    Returns:
        包含所有user_input值的参数字典
    """
    try:
        arguments = {}

        # 检查输入有效性
        if not tool_schema or not isinstance(tool_schema, dict):
            return arguments

        if 'input_schema' not in tool_schema or not isinstance(tool_schema['input_schema'], dict):
            return arguments

        input_schema = tool_schema['input_schema']

        if 'properties' not in input_schema or not isinstance(input_schema['properties'], dict):
            return arguments

        properties = input_schema['properties']

        logger.info("=" * 80)
        logger.info("[ExtractArguments] 开始提取参数")
        logger.info(f"[ExtractArguments] 工具名称: {tool_schema.get('name', 'unknown')}")

        # 遍历所有属性，提取user_input
        for prop_name, prop_value in properties.items():
            if (isinstance(prop_value, dict) and
                    'user_input' in prop_value):
                user_input_value = prop_value['user_input']
                arguments[prop_name] = user_input_value
                # 对于 Executor 和 Storage 参数，记录详细信息
                if prop_name in ['Executor', 'Storage']:
                    logger.info(f"[ExtractArguments] {prop_name} 参数:")
                    logger.info(f"  类型: {type(user_input_value).__name__}")
                    logger.info(f"  内容: {json.dumps(user_input_value, ensure_ascii=False, indent=2)}")

        logger.info(f"[ExtractArguments] 最终参数: {arguments}")
        logger.info("=" * 80)

        return arguments

    except Exception as e:
        logger.error(f"提取参数时发生错误: {e}")
        return {}