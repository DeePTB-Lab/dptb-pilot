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
    assert captured['kwargs']['pp_orb_paths'] == [Path('C.upf'), Path('C.orb')]
    assert captured['kwargs']['stru_path'] == tmp_path / 'STRU'
    assert str(pp.resolve()) in captured['packages']
    assert str(orb.resolve()) in captured['packages']


def test_nested_paths_reject_conflicting_basenames(tmp_path):
    first = tmp_path / 'a' / 'same.upf'
    second = tmp_path / 'b' / 'same.upf'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('first')
    second.write_text('second')
    executor = NestedPathDispatcherExecutor(
        machine={'context_type': 'SSHContext'},
    )

    try:
        executor.submit(lambda **kwargs: kwargs, {'paths': [first, second]})
    except ValueError as exc:
        assert 'same.upf' in str(exc)
    else:
        raise AssertionError('Conflicting nested input names must be rejected')
