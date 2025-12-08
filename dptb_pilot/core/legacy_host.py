import time
import functools

import gradio as gr
import json
import os
from typing import Dict, List, Tuple, Any, AsyncGenerator

from gradio.components.chatbot import ExampleMessage

from dptb_pilot.core.agent import create_llm_agent
from dptb_pilot.core.session import pop_event
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

from better_aim.utils import generate_random_string, hash_dict


def get_chat_history_file_path(sha_id: str, work_path: str) -> str:
    """获取聊天历史文件路径"""
    # 确保文件路径存在
    history_file_path = os.path.join(work_path, "chat_history")
    os.makedirs(history_file_path, exist_ok=True)
    return os.path.join(history_file_path, f"{sha_id}.json")


def load_chat_history(sha_id: str, work_path: str) -> List[List[str]]:
    """加载聊天历史记录"""
    history_file = get_chat_history_file_path(sha_id, work_path)

    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_history(sha_id: str, history: List[List[str]], work_path: str):
    """保存聊天历史记录"""
    history_file = get_chat_history_file_path(sha_id, work_path)

    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存聊天历史失败: {e}")


def login(session_id: str,
          mcp_tools_url: str,
          agent_info: dict,
          work_path: str,
          model_config: dict) -> Tuple[
    gr.update, gr.update, str]:
    """处理登录逻辑"""

    print(work_path)

    if not session_id:
        return gr.update(visible=True), gr.update(visible=False), "请填写或自动生成会话ID", []
    elif len(session_id) != 32:
        return gr.update(visible=True), gr.update(visible=False), f"会话ID需要为长度为32的任意字符，目前长度：{len(session_id)}", []

    # 生成SHA ID
    sha_id = session_id

    # 创建或获取agent
    from dptb_pilot.core.legacy_main import active_agents
    if sha_id not in active_agents:
        try:
            agent = create_llm_agent(session_id=session_id,
                                     mcp_tools_url=mcp_tools_url,
                                     agent_info=agent_info,
                                     model_config=model_config)
            active_agents[sha_id] = agent
        except Exception as e:
            return gr.update(visible=True), gr.update(visible=False), f"创建Agent失败: {str(e)}", []
    else:
        agent = active_agents[sha_id]

    # 加载聊天历史
    chat_history = load_chat_history(sha_id, work_path)

    # 返回更新后的界面和状态
    return (
        gr.update(visible=False),  # 隐藏登录界面
        gr.update(visible=True),  # 显示聊天界面
        f"登录成功! 会话 ID: {'*' * 16 + sha_id[:4]}"  # 状态消息
    )

# modified from https://google.github.io/adk-docs/tutorials/agent-team/#step-1-your-first-agent-basic-weather-lookup
async def call_agent_async(query: str, runner, user_id, session_id, tools_info: List[Dict[str, Any]]):
    """Sends a query to the agent and prints the final response."""
    #print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        #print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}, a:{event.content.parts}")
        """
        # 检测 function call
        # https://google.github.io/adk-docs/events/#identifying-event-origin-and-type
        if event.content and event.content.parts:
            calls = event.get_function_calls()
            if calls:
                for call in calls:
                    tool_name = call.name
                    arguments = call.args  # This is usually a dictionary
                    print(f"  Tool: {tool_name}, Args: {arguments}")
                    # Application might dispatch execution based on this

                    session = session_service.get_session_sync(app_name=runner.app_name,
                                                               user_id=user_id,
                                                               session_id=session_id)

                    last_event = await pop_event(session_service=session_service, session=session)
                    print(last_event)

                    schema = zip_tool_schema(tool_name=tool_name,
                                             arguments=arguments,
                                             tools_dict=tools_info)

                    yield event.content.parts[0].text, schema, False

                    pending_events[session_id] = asyncio.Event()
                    print("等待用户点击按钮以继续执行...")
                    await pending_events[session_id].wait()  # ⏸ 暂停在这里直到按钮被点击
                    print("✅ 用户已点击按钮，继续执行")

                    last_event.content.parts[1].function_call.args = modified_args_store[session_id]
                    print(modified_args_store)
                    print("----")
                    print(last_event.content.parts[1].function_call.args)
                    await session_service.append_event(session=session, event=last_event)
                    print(session)
                # 继续对话
                continue

        session = session_service.get_session_sync(app_name=runner.app_name,
                                                   user_id=user_id,
                                                   session_id=session_id)
        print(f"----------------------{session}")
        """
        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                yield event.content.parts[0].text, True
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                yield f"Agent escalated: {event.error_message or 'No specific message.'}", True
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found
        else:
            if event.content and event.content.parts:
                yield event.content.parts[0].text, False

    #print(f"<<< Agent Response: {final_response_text}")


async def chat_with_agent(message: str,
                          history: List[List[str]],
                          session_id: str,
                          agent_info: dict,
                          work_path: str,
                          tools_info: List[Dict[str, Any]]) -> \
AsyncGenerator[tuple[list[list[str]], str, bool], Any]:
    """处理与agent的聊天"""
    from dptb_pilot.core.legacy_main import active_agents
    if session_id not in active_agents:
        yield history, "Agent未找到，请重新登录", True
        return

    agent = active_agents[session_id]
    from dptb_pilot.core.legacy_main import session_service
    session = await session_service.create_session(app_name=agent_info["name"],
                                   user_id=session_id[:4],
                                   session_id=session_id)

    runner = Runner(agent=agent,
                    app_name=agent_info["name"],
                    session_service=session_service)

    responses = []
    last_yielded_history = None

    async for response, is_final in call_agent_async(query=message,
                                                             runner=runner,
                                                             user_id=session_id[:4],
                                                             session_id=session_id,
                                                             tools_info=tools_info):
        # 更新聊天历史
        if response:
            if len(responses) == 0:
                responses.append([message, response])
            else:
                responses.append([None, response])
            new_history = history + responses
        else:
            new_history = history

        # 保存聊天历史
        save_chat_history(session_id, new_history, work_path)

        # 只有当历史记录发生变化时才yield，避免重复发送相同数据
        if new_history != last_yielded_history:
            last_yielded_history = new_history
            if is_final:
                yield new_history, "待机中。", True
            else:
                yield new_history, "正在等待LLM回复……", False


def logout() -> Tuple[gr.update, gr.update, str, str]:
    """处理登出逻辑"""
    return (
        gr.update(visible=True),  # 显示登录界面
        gr.update(visible=False),  # 隐藏聊天界面
        "已登出",  # 状态消息
        ""# 清空登录表单
    )

def handle_refresh(work_path: str, session_id):
    session_dir = os.path.join(work_path, session_id)
    os.makedirs(session_dir, exist_ok=True)
    # 返回更新后的文件列表（绝对路径）
    all_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir)
        if os.path.isfile(os.path.join(session_dir, f))
    ])
    return all_files

def handle_upload(files, work_path: str, session_id):
    session_dir = os.path.join(work_path, session_id)
    os.makedirs(session_dir, exist_ok=True)
    saved_files = []
    if files is None:
        # 无文件上传时，只返回当前文件列表
        files = []
    # 处理每个上传文件
    for file_obj in files:
        file_path = file_obj.name  # 上传后 Gradio 存储的临时文件路径
        # 检查文件大小 (最大 10MB)
        if os.path.getsize(file_path) > 10 * 1024 * 1024:
            continue  # 跳过超限文件，可改为报错提示
        filename = os.path.basename(file_path)
        dest_path = os.path.join(session_dir, filename)
        # 复制文件到会话目录（覆盖同名文件）
        with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        saved_files.append(dest_path)
    # 返回更新后的文件列表（绝对路径）
    all_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir)
        if os.path.isfile(os.path.join(session_dir, f))
    ])
    return all_files


def update_interface(selected_value):
    if selected_value == "玻尔(作为任务提交到Bohrium)":
        return [
            "Bohr",
            gr.Textbox(visible=True),
            gr.Textbox(visible=True),
            gr.Textbox(visible=True)
        ]
    elif selected_value == "在线(在agent部署服务器运行)":
        return [
            "Local",
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False)
        ]
    else:
        return [
            "None",
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False)
        ]


def update_executor_storage_state(mode, username, password, project_id):
    if mode == "在线(在agent部署服务器运行)":
        return [
            {
                "type": "local"
            },
            {
                "type": "https"
            }
        ]
    elif mode == "玻尔(作为任务提交到Bohrium)":
        return [
            {
                "type": "dispatcher",
                "machine": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "email": username,
                        "password": password,
                        "program_id": int(project_id),
                        "input_data": {
                            "image_name": "registry.dp.tech/dptech/dp/native/prod-19853/dpa-mcp:0.0.0",
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "1 * NVIDIA V100_32g"
                        }
                    }
                }
            },
            {
                "type": "bohrium",
                "username": username,
                "password": password,
                "program_id": int(project_id),
            }
        ]




def create_interface(mcp_server_url: str,
                     agent_info: dict,
                     work_path: str,
                     tools_info: List[Dict[str, Any]],
                     model_config: dict,
                     mcp_server_mode: str):
    # 发送消息事件
    async def handle_send_message(message, _session_id, _tools_info):
        from dptb_pilot.core.legacy_main import history_pool
        history = history_pool[_session_id]
        if not message.strip():
            yield history, "消息不能为空", {}, True
        else:
            # 使用一个累积变量来存储所有响应
            accumulated_history = history.copy() if history else []

            async for new_history, status, finish in chat_with_agent(message,
                                                                     history,
                                                                     session_id=_session_id,
                                                                     agent_info=agent_info,
                                                                     work_path=work_path,
                                                                     tools_info=tools_info):
                # 更新累积的历史记录
                if new_history and len(new_history) > 0:
                    accumulated_history.append(new_history)

                print(accumulated_history)

                history_pool[_session_id] = accumulated_history

                # 每次都yield最新的状态，确保Gradio能接收到所有更新
                yield status, finish

    """创建Gradio界面"""
    with (gr.Blocks(title=agent_info["name"], theme=gr.themes.Soft()) as demo):
        # 状态变量
        session_id_state = gr.State("")
        mcp_server_url_state = gr.State(mcp_server_url)
        agent_info_state = gr.State(agent_info)
        model_config_state = gr.State(model_config)
        work_path_state = gr.State(work_path)
        tools_info_state = gr.State(tools_info)
        finish_state = gr.State(True)
        task_submit_mode_state = gr.State("None")
        executor_state = gr.State()
        storage_state = gr.State()
        schema_state = gr.State()
        schema_hash_store_state = gr.State({})

        gr.Markdown(f"# {agent_info['name']}")

        with gr.Column(visible=True) as login_section:
            gr.Markdown("## 创建会话")

            with gr.Row(equal_height=True):
                session_id = gr.Textbox(label="会话ID", placeholder="请输入32位任意字符串", value="", scale=4)
                generate_btn = gr.Button("随机生成", scale=1, min_width=100, variant="primary")

            gr.Markdown("输入或自动生成32位任意字符串作为您的专属会话ID，使用相同ID可以访问此前的历史记录，历史记录在一小时后会被自动清除，请不要传播您专属的ID！")

            login_btn = gr.Button("进入会话", variant="primary")

            status_msg = gr.Textbox(label="状态", interactive=False, value="")

        with gr.Column(visible=False) as chat_section:
            with gr.Row():
                with gr.Column(scale=1) as files_column:
                    if mcp_server_mode == "bohr-agent-sdk":
                        with gr.Row():
                            gr.Markdown(
                                f"**工具执行位置**")
                            gr.HTML("""
                                                            <div style="margin-left: 10px; margin-top: 0px;">
                                                                <span title="在线模式：将会在agent部署的服务器上运行，所有数据将会在一小时后自动清除，且可能无法访问完整文件。
玻尔模式：将会在你的Bohrium存储中运行，任务会被提交为Bohrium任务，可能会产生一定的开销，但数据不会保存在当前服务器上。
某些任务不能在线运行（如需要大量算力或时间的任务）将会自动提交为Bohrium任务。" style="cursor: help; font-size: 16px;">❓</span>
                                                            </div>
                                                            """)

                        mode_choice = gr.Dropdown(
                            choices=["请选择", "在线(在agent部署服务器运行)", "玻尔(作为任务提交到Bohrium)"],
                            value="请选择",
                            label="选择运行模式"
                        )

                        bohrium_username = gr.Textbox(visible=False, label="Bohrium用户名")
                        bohrium_password = gr.Textbox(visible=False, label="Bohrium密码")
                        bohrium_project_id = gr.Textbox(visible=False, label="Bohrium项目ID")

                        mode_choice.change(
                            fn=update_interface,
                            inputs=mode_choice,
                            outputs=[task_submit_mode_state, bohrium_username, bohrium_password, bohrium_project_id]
                        ).then(
                            fn=update_executor_storage_state,
                            inputs=[mode_choice, bohrium_username, bohrium_password, bohrium_project_id],
                            outputs=[executor_state, storage_state]
                        )

                        bohrium_username.change(
                            fn=update_executor_storage_state,
                            inputs=[mode_choice, bohrium_username, bohrium_password, bohrium_project_id],
                            outputs=[executor_state, storage_state]
                        )

                        bohrium_password.change(
                            fn=update_executor_storage_state,
                            inputs=[mode_choice, bohrium_username, bohrium_password, bohrium_project_id],
                            outputs=[executor_state, storage_state]
                        )

                        bohrium_project_id.change(
                            fn=update_executor_storage_state,
                            inputs=[mode_choice, bohrium_username, bohrium_password, bohrium_project_id],
                            outputs=[executor_state, storage_state]
                        )

                    with gr.Row(equal_height=True):
                        with gr.Column():
                            file_list = gr.File(label="您的专属工作区（在线模式）", file_count="multiple")

                            with gr.Row():
                                refresh_btn = gr.Button("刷新目录")
                                upload_btn = gr.UploadButton("上传文件", file_count="multiple", file_types=None)

                                refresh_btn.click(
                                    fn=handle_refresh,
                                    inputs=[work_path_state, session_id_state],
                                    outputs=[file_list]
                                )

                                upload_btn.upload(
                                    fn=handle_upload,
                                    inputs=[upload_btn, work_path_state, session_id_state],
                                    outputs=[file_list]
                                )

                with gr.Column(scale=3) as main_column:
                    gr.Markdown(f"## 与{agent_info['name']}协作")

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            clear_btn = gr.Button("清空对话")
                            logout_btn = gr.Button("离开会话", variant="secondary")
                        # 显示当前用户和项目信息
                        current_info = gr.Textbox(label="当前会话信息", interactive=False, value="", scale=2)
                        chat_status = gr.Textbox(label="聊天状态", interactive=False, scale=2)

                    from dptb_pilot.core.legacy_main import history_pool

                    @gr.render(inputs=session_id_state)
                    def render_chatbot(_session_id):
                        if _session_id:
                            if not history_pool[session_id]:
                                history_pool[session_id] = {"role": "assistant",
                                                            "content": "I am happy to provide you that report and plot."}
                            chatbot = gr.Chatbot(
                                value=history_pool[session_id],
                                label="聊天记录",
                                height=700,
                                show_copy_button=True,
                                examples=[
                                    {"text":'请帮我生成碳的训练输入配置文件，基组为{"C":"2s1p"}，截断半径{"C":6.0}，训练数据路径"my_data"，前缀"C16"，其余按默认配置',
                                     "display_text":"生成训练输入配置文件"},
                                    {"text":"使用poly4基准模型绘制能带图，结构文件为xxx",
                                     "display_text":"使用基准模型绘制能带图"},
                                    {"text":"请帮我生成sp轨道的Si的ploy4基准模型",
                                     "display_text":"生成基准模型"},
                                    {"text":"请使用我的模型预测并绘制能带图",
                                     "display_text":"使用模型预测并绘制能带图"}
                                ]
                            )

                    with gr.Row(equal_height=True):
                        example = gr.Dropdown(
                            choices=["-",
                                     '请帮我生成碳的训练输入配置文件，基组为{"C":"2s1p"}，截断半径{"C":6.0}，训练数据路径"my_data"，前缀"C16"，其余按默认配置',
                                     "使用poly4基准模型绘制能带图，结构文件为xxx",
                                     "请帮我生成sp轨道的Si的ploy4基准模型",
                                     "请使用我的模型预测并绘制能带图"],
                            value="-",
                            label="输入对话示例"
                        )

                        send_btn = gr.Button("发送", variant="primary", scale=1)
                        @gr.render(inputs=finish_state)
                        def render_input_box(finish):
                            send_btn.interactive = finish

                        msg = gr.Textbox(
                            label="输入消息",
                            placeholder=f"输入你想对{agent_info['name']}说的话...",
                            scale=4
                        )

                        send_btn.click(
                            fn=handle_send_message,
                            inputs=[msg, session_id_state, tools_info_state],
                            outputs=[chat_status, finish_state]
                        ).then(
                            lambda: "",  # 清空输入框
                            outputs=msg
                        )

                        # 回车发送消息
                        """msg.submit(
                            fn=handle_send_message,
                            inputs=[msg, chatbot, session_id_state, tools_info_state],
                            outputs=[chatbot, chat_status, schema_state]
                        ).then(
                            lambda: "",  # 清空输入框
                            outputs=msg
                        )"""

                        example.change(
                            fn=lambda m:m,
                            inputs=example,
                            outputs=[msg]
                        )

                with gr.Column(scale=1) as values_column:
                    gr.Markdown("## 修改运行参数")
                    gr.Markdown("当调用mcp工具时，将会弹出参数的确认或手动修改")

                    # 尝试更新
                    def check_update_schema(_session_id, hash_store):
                        print("Checking update...")
                        from dptb_pilot.core.legacy_main import unmodified_schema_store
                        if _session_id not in unmodified_schema_store.keys():
                            return {_: ""}
                        new_hash = hash_dict(unmodified_schema_store[_session_id])
                        if _session_id not in hash_store.keys() or new_hash != hash_store[_session_id]:
                            hash_store[_session_id] = new_hash
                            return {schema_state: unmodified_schema_store[_session_id]}
                        return {_: ""}  # 无变化时跳过更新

                    _ = gr.State()

                    timer = gr.Timer(1)
                    timer.tick(fn=check_update_schema,
                               inputs=[session_id_state, schema_hash_store_state],
                               outputs=[schema_state, _])

                    @gr.render(inputs=[schema_state, session_id_state, task_submit_mode_state])
                    def render_form(schema:dict, _session_id, task_submit_mode):
                        print("Updating...")
                        gr.Markdown("### 使用的运行参数：")

                        if schema and len(schema.keys()) != 0:
                            output_json = gr.JSON(value=schema)

                            submit_json_button = gr.Button("通过JSON快速提交")
                            input_json = gr.Textbox(label="请输入JSON")

                            submit_json_button.click(lambda json_input: json.loads(json_input), inputs=input_json,
                                                     outputs=output_json)

                            submit_normal_button = gr.Button("修改各个参数提交")

                        if not schema:
                            gr.Markdown("**当调用工具时，此处会显示工具的各种函数。**")
                            return
                        # 显示 name 和 description
                        gr.Markdown(f"## {schema['name']}")
                        gr.Markdown(f"{schema['description']}")

                        inputs = []
                        # 为每个参数生成输入区域
                        for key, prop in schema['input_schema']['properties'].items():
                            title = prop.get('title', key)
                            dtype = prop.get('type', None)
                            agent_val = prop.get('agent_input', "")
                            default_val = prop.get('default', "")
                            # 对于bohr-agent-sdk产生的executor和storage类型：
                            if title == 'Executor':
                                inp = executor_state

                            elif title == 'Storage':
                                inp = storage_state

                            elif title == 'Work Path' and task_submit_mode == "Local":
                                inp = gr.State("/tmp/" + _session_id)

                            else:
                                # 显示参数信息
                                gr.Markdown(
                                    f"**{title}** (type: {dtype}) — 默认: `{default_val}`, Agent 尝试值: `{agent_val}`")
                                # 根据类型选择组件
                                if dtype == 'string':
                                    inp = gr.Textbox(
                                        value=agent_val if agent_val is not None else str(default_val),
                                        label=title, key=key
                                    )
                                elif dtype in ('number', 'integer', 'float'):
                                    # 处理数字类型，默认值转换为 float
                                    num_val = agent_val if agent_val is not None else default_val
                                    try:
                                        num_val = float(num_val)
                                    except:
                                        num_val = 0.0
                                    precision = 0 if dtype == 'integer' else None
                                    inp = gr.Number(
                                        value=num_val, precision=precision,
                                        label=title, key=key
                                    )
                                elif dtype in 'bool':
                                    inp = gr.Dropdown(
                                        choices=["False", "True"],
                                        value=agent_val if agent_val is not None else str(default_val),
                                        label=title,
                                        key=key
                                    )
                                else:
                                    # 其它类型（如对象）使用 JSON 输入
                                    default_json = agent_val if isinstance(agent_val, (dict, list)) else (
                                        default_val if isinstance(default_val, (dict, list)) else {})
                                    inp = gr.JSON(
                                        value=default_json, label=title, key=key
                                    )
                            inputs.append(inp)

                            # 提交按钮收集所有输入并生成输出
                            submit_normal_button.click(lambda _session_id, *vals: collect_inputs(schema, _session_id, *vals),
                                                       inputs=[session_id_state, *inputs],
                                                       outputs=output_json
                            )

        def on_generate_click():
            """当生成按钮被点击时的回调函数"""
            return generate_random_string()

        generate_btn.click(
            fn=on_generate_click,
            outputs=session_id
        )
        # 更新会话信息显示
        def update_session_info(sha, username, project_id, file_path):
            if sha:
                return f"用户: {username} | 项目ID: {project_id} | 文件路径: {file_path}"
            return "未登录"

        # 登录按钮事件
        login_btn.click(
            fn=login,
            inputs=[session_id, mcp_server_url_state, agent_info_state, work_path_state, model_config_state],
            outputs=[login_section, chat_section, status_msg]
        ).then(
            lambda _session_id:
            (_session_id,
             f"会话ID: {_session_id}"),
            inputs=[session_id],
            outputs=[session_id_state, current_info]
        )

        # 清空对话
        clear_btn.click(
            fn=lambda sha: ("对话已清空"),
            inputs=[session_id_state],
            outputs=[chat_status]
        )

        # 登出按钮事件
        logout_btn.click(
            fn=logout,
            outputs=[
                login_section,
                chat_section,
                status_msg
            ]
        ).then(
            lambda: ("", "", "", "", "未登录"),  # 清空状态
            outputs=[session_id_state, current_info]
        )

    return demo
