# -*- coding: utf-8 -*-
"""
测试 Executor 和 Storage 参数生成功能
"""
import sys
import os
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dptb_pilot.server.app import (
    build_execution_config,
    generate_executor_and_storage,
)
from dp.agent.server.calculation_mcp_server import init_executor
from dp.agent.server.storage import BohriumStorage


def test_build_bohrium_execution_config_uses_sdk_contract():
    config = build_execution_config('Remote', {
        'type': 'Bohrium',
        'config': {
            'username': 'user@example.com',
            'password': 'secret',
            'project_id': '123',
            'image_name': 'example/image:latest',
            'scass_type': 'c4_m8_cpu',
        },
    })

    assert config['executor']['type'] == 'dispatcher'
    assert config['executor']['machine']['context_type'] == 'Bohrium'
    assert config['executor']['machine']['remote_profile']['program_id'] == 123
    assert config['storage']['type'] == 'bohrium'
    assert config['storage']['project_id'] == '123'
    assert 'program_id' not in config['storage']

    executor_type, executor = init_executor(config['executor'])
    assert executor_type == 'dispatcher'
    assert executor.machine['context_type'] == 'Bohrium'
    storage_parameters = inspect.signature(BohriumStorage).parameters
    assert 'project_id' in storage_parameters
    assert 'program_id' not in storage_parameters


def test_build_execution_config_rejects_executor_shorthand():
    try:
        build_execution_config('Remote', {
            'type': 'bohrium',
            'config': {'username': 'user', 'password': 'secret'},
        })
    except ValueError as exc:
        assert 'project_id' in str(exc)
    else:
        raise AssertionError('Incomplete Bohrium config should be rejected')


def test_executor_generation_slurm():
    """测试 Slurm 类型机器的 Executor 生成"""
    print("=" * 80)
    print("测试 Slurm Executor 生成")
    print("=" * 80)

    # 模拟输入 schema（小写 key）
    tool_schema = {
        "name": "band_with_baseline_model",
        "description": "Test tool",
        "input_schema": {
            "properties": {
                "basemodel": {
                    "type": "string",
                    "default": "poly4",
                    "agent_input": "poly4",
                    "user_input": "poly4"
                },
                "executor": {  # 小写 key
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"}
                    ],
                    "default": None,
                    "title": "Executor",
                    "agent_input": {},
                    "user_input": {}
                },
                "storage": {  # 小写 key
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"}
                    ],
                    "default": None,
                    "title": "Storage",
                    "agent_input": {},
                    "user_input": {}
                }
            }
        }
    }

    # 模拟远程机器配置
    remote_machine = {
        "id": "test-001",
        "name": "test-slurm",
        "type": "Slurm",
        "config": {
            "remote_root": "/public4/home/test/work",
            "hostname": "test.server.com",
            "username": "testuser",
            "key_filename": "/home/test/.ssh/id_rsa",
            "number_node": "1",
            "gpu_per_node": "0",
            "cpu_per_node": "64",
            "queue_name": "test_queue",
            "custom_flags": "source /home/test/env.sh"
        }
    }

    # 调用函数
    result_schema = generate_executor_and_storage(
        execution_mode='Remote',
        remote_machine=remote_machine,
        tool_schema=tool_schema
    )

    # 检查结果
    print("\n原始 schema 中的 executor:")
    print(f"  key: 'executor'")
    print(f"  user_input (before): {tool_schema['input_schema']['properties']['executor']['user_input']}")

    print("\n返回 schema 中的 executor:")
    executor_key = None
    for key in ['Executor', 'executor']:
        if key in result_schema.get('input_schema', {}).get('properties', {}):
            executor_key = key
            break

    if executor_key:
        executor_prop = result_schema['input_schema']['properties'][executor_key]
        print(f"  key: '{executor_key}'")
        print(f"  user_input (after): {executor_prop.get('user_input')}")

        # 检查是否生成了正确的 Executor 配置
        user_input = executor_prop.get('user_input')
        if user_input and isinstance(user_input, dict):
            if user_input.get('type') == 'dispatcher' and \
               user_input.get('machine', {}).get('batch_type') == 'Slurm':
                print("\n✅ Slurm Executor 配置生成成功！")
                print(f"Executor 配置: {user_input}")
                return True

    print("\n❌ Executor 配置生成失败！")
    return False


def test_executor_generation_bohrium():
    """测试 Bohrium 类型机器的 Executor 和 Storage 生成"""
    print("=" * 80)
    print("测试 Bohrium Executor 和 Storage 生成")
    print("=" * 80)

    # 模拟输入 schema（大写 key）
    tool_schema = {
        "name": "test_tool",
        "description": "Test tool",
        "input_schema": {
            "properties": {
                "Executor": {  # 大写 key
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"}
                    ],
                    "default": None,
                    "title": "Executor",
                    "agent_input": {},
                    "user_input": {}
                },
                "Storage": {  # 大写 key
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"}
                    ],
                    "default": None,
                    "title": "Storage",
                    "agent_input": {},
                    "user_input": {}
                }
            }
        }
    }

    # 模拟远程机器配置
    remote_machine = {
        "id": "test-002",
        "name": "test-bohrium",
        "type": "Bohrium",
        "config": {
            "username": "testuser",
            "password": "testpass",
            "project_id": "12345",
            "image_name": "test-image:latest",
            "scass_type": "1 * NVIDIA V100_32g"
        }
    }

    # 调用函数
    result_schema = generate_executor_and_storage(
        execution_mode='Remote',
        remote_machine=remote_machine,
        tool_schema=tool_schema
    )

    # 检查 Executor
    print("\n返回 schema 中的 Executor:")
    executor_key = None
    for key in ['Executor', 'executor']:
        if key in result_schema.get('input_schema', {}).get('properties', {}):
            executor_key = key
            break

    if executor_key:
        executor_prop = result_schema['input_schema']['properties'][executor_key]
        user_input = executor_prop.get('user_input')
        print(f"  key: '{executor_key}'")
        print(f"  user_input: {user_input}")

    # 检查 Storage
    print("\n返回 schema 中的 Storage:")
    storage_key = None
    for key in ['Storage', 'storage']:
        if key in result_schema.get('input_schema', {}).get('properties', {}):
            storage_key = key
            break

    if storage_key:
        storage_prop = result_schema['input_schema']['properties'][storage_key]
        user_input = storage_prop.get('user_input')
        print(f"  key: '{storage_key}'")
        print(f"  user_input: {user_input}")

        # 检查是否生成了正确的 Storage 配置
        if user_input and isinstance(user_input, dict):
            if user_input.get('type') == 'bohrium':
                print("\n✅ Bohrium Storage 配置生成成功！")
                return True

    print("\n❌ Storage 配置生成失败！")
    return False


def test_local_mode():
    """测试 Local 模式（不生成 Executor 和 Storage）"""
    print("=" * 80)
    print("测试 Local 模式（不应修改 Executor 和 Storage）")
    print("=" * 80)

    tool_schema = {
        "name": "test_tool",
        "description": "Test tool",
        "input_schema": {
            "properties": {
                "Executor": {
                    "type": "object",
                    "user_input": {}
                },
                "Storage": {
                    "type": "object",
                    "user_input": {}
                }
            }
        }
    }

    # 不传 remote_machine
    result_schema = generate_executor_and_storage(
        execution_mode='Local',
        remote_machine=None,
        tool_schema=tool_schema
    )

    # 检查是否未修改
    executor_prop = result_schema['input_schema']['properties'].get('Executor', {})
    storage_prop = result_schema['input_schema']['properties'].get('Storage', {})

    if executor_prop.get('user_input') == {} and storage_prop.get('user_input') == {}:
        print("✅ Local 模式正确保持原有配置")
        return True
    else:
        print("❌ Local 模式错误地修改了配置")
        return False


if __name__ == "__main__":
    print("\n\n开始单元测试...\n")

    # 测试 1: Local 模式
    test1_passed = test_local_mode()
    print()

    # 测试 2: Slurm Executor 生成
    test2_passed = test_executor_generation_slurm()
    print()

    # 测试 3: Bohrium Executor 和 Storage 生成
    test3_passed = test_executor_generation_bohrium()
    print()

    print("=" * 80)
    print("测试结果汇总:")
    print("=" * 80)
    print(f"Local 模式测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"Slurm Executor 生成测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"Bohrium Executor/Storage 生成测试: {'✅ 通过' if test3_passed else '❌ 失败'}")

    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 存在失败'}")

    sys.exit(0 if all_passed else 1)
