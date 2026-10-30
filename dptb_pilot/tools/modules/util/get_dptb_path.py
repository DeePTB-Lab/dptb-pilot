import importlib.util
import os

def get_dptb_path():
    spec = importlib.util.find_spec("dptb")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("dptb package not found")
    return spec.submodule_search_locations[0]