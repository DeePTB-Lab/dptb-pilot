import os
import json
import time

ENVS = {
    "DPTB_AGENT_WORK_PATH": os.path.join(os.environ.get("WORK_ROOT", "/tmp"), "server_temp"),
    "DPTB_AGENT_SUBMIT_TYPE": "local",  # local, bohrium

    # connection settings
    "DPTB_AGENT_TRANSPORT": "sse",  # sse, streamable-http
    "DPTB_AGENT_HOST": "localhost",
    "DPTB_AGENT_PORT": "50002",
    "DPTB_AGENT_MODEL": "dp",  # fastmcp, dptb, dp

    # bohrium settings
    "BOHRIUM_USERNAME": "",
    "BOHRIUM_PASSWORD": "",
    "BOHRIUM_PROJECT_ID": "",
    "BOHRIUM_DPTB_IMAGE": "null", # THE bohrium image for dptb calculations,
    "BOHRIUM_DPTB_MACHINE": "c32_m64_cpu",  # THE bohrium machine for dptb calculations, c32_m64_cpu
    "BOHRIUM_DPTB_COMMAND": "dptb",
    
    # dptb pp orb settings
    "DPTB_COMMAND": "dptb",  # dptb executable command
    "DPTB_PP_PATH": "",  # dptb pseudopotential library path
    "DPTB_ORB_PATH": "",  # dptb orbital library path
    
    "_comments":{
        "DPTB_WORK_PATH": "The working directory for Dptb_Agent, where all temporary files will be stored.",
        "DPTB_SUBMIT_TYPE": "The type of submission for DPTB, can be local or bohrium.",
        "DPTB_AGENT_TRANSPORT": "The transport protocol for Dptb_Agent, can be 'sse' or 'streamable-http'.",
        "DPTB_AGENT_HOST": "The host address for the Dptb_Agent server.",
        "DPTB_AGENT_PORT": "The port number for the Dptb_Agent server.",
        "DPTB_AGENT_MODEL": "The model to use for Dptb_Agent, can be 'fastmcp', 'test', or 'dp'.",
        "BOHRIUM_USERNAME": "The username for Bohrium.",        
        "BOHRIUM_PASSWORD": "The password for Bohrium.",
        "BOHRIUM_PROJECT_ID": "The project ID for Bohrium.",
        "BOHRIUM_DPTB_IMAGE": "The image for dptb on Bohrium.",
        "BOHRIUM_DPTB_MACHINE": "The machine type for dptb on Bohrium.",
        "BOHRIUM_DPTB_COMMAND": "The command to run dptb on Bohrium",
        "DPTB_COMMAND": "The command to execute dptb on local machine.",
        "DPTB_PP_PATH": "The path to the pseudopotential library for Dptb.",
        "DPTB_ORB_PATH": "The path to the orbital library for Dptb.",
        "_comments": "This dictionary contains the default environment variables for Dptb_Agent."
    }
}

def set_envs(transport_input=None, model_input=None, port_input=None, host_input=None):
    """
    Set environment variables for Dptb_Agent.
    
    Args:
        transport_input (str, optional): The transport protocol to use. Defaults to None.
        model_input (str, optional): The model to use. Defaults to None.
        port_input (int, optional): The port number to run the MCP server on. Defaults to None.
        host_input (str, optional): The host address to run the MCP server on. Defaults to None.
    
    Returns:
        dict: The environment variables that have been set.
    
    Notes:
        - The input parameters has higher priority than the default values in `ENVS`.
        - If the `~/.dptb_agent_tools/env.json` file does not exist, it will be created with default values.
    """
    # read setting in ~/.dptb_agent_tools/env.json
    envjson_file = os.path.expanduser("~/.dptb_agent_tools/env.json")
    if os.path.isfile(envjson_file):
        envjson = json.load(open(envjson_file, "r"))
    else:
        envjson = {}
        
    update_envjson = False    
    for key, value in ENVS.items():
        if key not in envjson:
            envjson[key] = value
            update_envjson = True
    
    if transport_input is not None:
        envjson["DPTB_AGENT_TRANSPORT"] = str(transport_input)
    if port_input is not None:
        envjson["DPTB_AGENT_PORT"] = str(port_input)
    if host_input is not None:
        envjson["DPTB_AGENT_HOST"] = str(host_input)
    if model_input is not None:
        envjson["DPTB_AGENT_MODEL"] = str(model_input)
        
    for key, value in envjson.items():
        os.environ[key] = str(value)
    
    if update_envjson:
        # write envjson to ~/.dptb_agent_tools/env.json
        os.makedirs(os.path.dirname(envjson_file), exist_ok=True)
        json.dump(
            envjson,
            open(envjson_file, "w"),
            indent=4
        )
    return envjson
    
def create_workpath():
    """
    Create the working directory for dptb agent, and change the current working directory to it.
    
    Returns:
        str: The path to the working directory.
    """
    # 优先使用显式设置的 WORK_ROOT
    work_root = os.environ.get("WORK_ROOT")
    
    if work_root:
        base_path = os.path.join(work_root, "server_temp")
    else:
        # 否则回退到可能的缓存配置或默认值
        default_work_path = os.path.join("/tmp", "server_temp")
        base_path = os.environ.get("DPTB_AGENT_WORK_PATH", default_work_path)

    work_path = f"{base_path}/{time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(work_path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work_path)
    print(f"Changed working directory to: {work_path}")
    # write the environment variables to a file
    json.dump({
        k: os.environ.get(k) for k in ENVS.keys()
    }.update({"DPTB_AGENT_START_PATH": cwd}),
        open("env.json", "w"), indent=4)
    
    return work_path    