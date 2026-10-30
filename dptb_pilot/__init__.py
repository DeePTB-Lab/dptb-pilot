"""Public entry points for :mod:`dptb_pilot`.

Keep these imports lazy so tool modules can be used without loading the web
server and Google ADK stack as a side effect of importing the package.
"""


def react_launch(*args, **kwargs):
    from dptb_pilot.main import react_launch as _react_launch

    return _react_launch(*args, **kwargs)


def launch(*args, **kwargs):
    from dptb_pilot.core.legacy_main import launch as _launch

    return _launch(*args, **kwargs)


__all__ = ["react_launch", "launch"]
