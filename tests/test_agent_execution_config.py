from dptb_pilot.core.agent import REMOTE_EXECUTION_TOOLS, mcp_tools


def test_mcp_toolset_overrides_model_executor_values():
    remote_executor = {
        'type': 'dispatcher',
        'machine': {
            'batch_type': 'Bohrium',
            'context_type': 'Bohrium',
            'remote_profile': {},
        },
    }

    toolset = mcp_tools(
        'http://localhost:50001/sse',
        {'executor': remote_executor},
    )

    assert toolset.override is True
    assert toolset.executor == {'type': 'local'}
    assert toolset.storage == {'type': 'local'}
    assert toolset.executor_map['run_abacus'] == remote_executor
    assert toolset.executor_map['run_abacus_scf_nscf'] == remote_executor
    assert 'convert_vasp_poscar_structure' not in REMOTE_EXECUTION_TOOLS
