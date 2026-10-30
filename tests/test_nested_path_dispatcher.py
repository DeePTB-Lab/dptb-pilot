from pathlib import Path

from dptb_pilot.tools.executor import NestedPathDispatcherExecutor


def test_nested_paths_are_added_to_forward_files(monkeypatch, tmp_path):
    pp = tmp_path / 'C.upf'
    orb = tmp_path / 'C.orb'
    pp.write_text('pp')
    orb.write_text('orb')
    captured = {}

    def fake_submit(self, fn, kwargs):
        captured['kwargs'] = kwargs
        captured['packages'] = list(self.python_packages)
        return {'job_id': 'test'}

    monkeypatch.setattr(
        'dp.agent.server.executor.dispatcher_executor.DispatcherExecutor.submit',
        fake_submit,
    )
    executor = NestedPathDispatcherExecutor(
        machine={'context_type': 'SSHContext'},
    )

    result = executor.submit(
        lambda **kwargs: kwargs,
        {'pp_orb_paths': [pp, orb], 'stru_path': tmp_path / 'STRU'},
    )

    assert result == {'job_id': 'test'}
    forwarded_paths = captured['kwargs']['pp_orb_paths']
    assert [path.name for path in forwarded_paths] == ['C.upf', 'C.orb']
    assert captured['kwargs']['stru_path'] == tmp_path / 'STRU'
    assert len(set(path.parts[0] for path in forwarded_paths)) == 1
    assert any('.dptb_nested_inputs_' in package for package in captured['packages'])
    assert all('.dptb_nested_inputs_' not in package for package in executor.python_packages)


def test_nested_paths_preserve_conflicting_basenames(monkeypatch, tmp_path):
    first = tmp_path / 'a' / 'same.upf'
    second = tmp_path / 'b' / 'same.upf'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('first')
    second.write_text('second')
    captured = {}

    def fake_submit(self, fn, kwargs):
        captured['kwargs'] = kwargs
        return {'job_id': 'test'}

    monkeypatch.setattr(
        'dp.agent.server.executor.dispatcher_executor.DispatcherExecutor.submit',
        fake_submit,
    )
    executor = NestedPathDispatcherExecutor(machine={'context_type': 'SSHContext'})

    executor.submit(lambda **kwargs: kwargs, {'paths': [first, second, first]})

    forwarded = captured['kwargs']['paths']
    assert forwarded[0].name == forwarded[1].name == 'same.upf'
    assert forwarded[0].parent != forwarded[1].parent
    assert forwarded[2] == forwarded[0]


def test_missing_remote_result_is_not_reported_as_empty_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'dp.agent.server.executor.dispatcher_executor.DispatcherExecutor.get_results',
        lambda self, job_id: {},
    )
    executor = NestedPathDispatcherExecutor(machine={'context_type': 'SSHContext'})

    try:
        executor.get_results('job-123')
    except RuntimeError as exc:
        assert 'job-123' in str(exc)
        assert 'results.txt' in str(exc)
    else:
        raise AssertionError('Missing remote results should raise an error')
