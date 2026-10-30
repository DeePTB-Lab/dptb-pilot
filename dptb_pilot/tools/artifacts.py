import inspect
import os
import types
from collections import abc
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from dp.agent.server import calculation_mcp_server


def _path_annotation(annotation: Any) -> Any | None:
    """Return the Path-containing branch of a supported annotation."""
    if annotation is Path:
        return Path

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _path_annotation(args[0])
    if origin in (Union, types.UnionType):
        path_shapes = []
        for branch in args:
            path_branch = _path_annotation(branch)
            if path_branch is not None:
                path_shapes.append(path_branch)
        if len(path_shapes) == 1:
            return path_shapes[0]
        if path_shapes:
            return (origin, tuple(path_shapes))
        return None
    if origin in (list, set, frozenset, abc.Iterable, abc.Sequence):
        item_annotation = _path_annotation(args[0]) if args else None
        return (origin, item_annotation) if item_annotation is not None else None
    if origin is tuple:
        item_annotations = tuple(
            _path_annotation(arg) if arg is not Ellipsis else Ellipsis
            for arg in args
        )
        if any(item not in (None, Ellipsis) for item in item_annotations):
            return (tuple, item_annotations)
        return None
    if origin in (dict, abc.Mapping) and len(args) == 2:
        key_annotation = _path_annotation(args[0])
        value_annotation = _path_annotation(args[1])
        if key_annotation is not None or value_annotation is not None:
            return (dict, key_annotation, value_annotation)
    return None


def _download_path(uri: str, destination: Path, storage_type: str, storage: Any) -> Path:
    uri = os.fspath(uri)
    if os.path.isabs(uri):
        scheme, key = "local", uri
    else:
        scheme, key = calculation_mcp_server.parse_uri(uri)
    source_storage = (
        storage
        if scheme == storage_type
        else calculation_mcp_server.storage_dict[scheme]()
    )
    path = source_storage.download(key, str(destination))
    calculation_mcp_server.logger.info("Artifact %s downloaded to %s", uri, path)
    return Path(path)


def _convert_paths(
    value: Any,
    annotation: Any,
    destination: Path,
    storage_type: str,
    storage: Any,
) -> Any:
    if annotation is Path:
        if isinstance(value, Path):
            return value
        return _download_path(value, destination, storage_type, storage)

    origin = annotation[0]
    if origin in (Union, types.UnionType):
        shapes = annotation[1]
        for shape in shapes:
            if shape is Path and isinstance(value, (str, os.PathLike)):
                return _convert_paths(value, shape, destination, storage_type, storage)
            if isinstance(shape, tuple) and isinstance(value, (list, tuple, set, frozenset)):
                return _convert_paths(value, shape, destination, storage_type, storage)
        # Preserve optional/non-path values in a mixed union.
        return value
    if origin in (list, set, frozenset, abc.Iterable, abc.Sequence):
        converted = [
            _convert_paths(item, annotation[1], destination / str(index), storage_type, storage)
            for index, item in enumerate(value)
        ]
        if origin in (abc.Iterable, abc.Sequence):
            return converted
        return origin(converted)
    if origin is tuple:
        item_annotations = annotation[1]
        repeated = len(item_annotations) == 2 and item_annotations[1] is Ellipsis
        converted = []
        for index, item in enumerate(value):
            item_annotation = item_annotations[0] if repeated else item_annotations[index]
            converted.append(
                _convert_paths(item, item_annotation, destination / str(index), storage_type, storage)
                if item_annotation is not None
                else item
            )
        return tuple(converted)
    if origin in (dict, abc.Mapping):
        key_annotation, value_annotation = annotation[1:]
        converted = {}
        for index, (key, item) in enumerate(value.items()):
            converted_key = (
                _convert_paths(key, key_annotation, destination / f"key-{index}", storage_type, storage)
                if key_annotation is not None
                else key
            )
            converted_value = (
                _convert_paths(
                    item,
                    value_annotation,
                    destination / f"value-{index}",
                    storage_type,
                    storage,
                )
                if value_annotation is not None
                else item
            )
            converted[converted_key] = converted_value
        return converted
    raise TypeError(f"Unsupported path annotation: {annotation!r}")


def handle_nested_input_artifacts(fn, kwargs, storage_config):
    """Download Path values recursively according to the tool annotation."""
    storage_type, storage = calculation_mcp_server.init_storage(storage_config)
    try:
        annotations = get_type_hints(fn, include_extras=True)
    except (NameError, TypeError):
        annotations = {
            name: parameter.annotation
            for name, parameter in inspect.signature(fn).parameters.items()
        }

    input_artifacts = {}
    for name, value in list(kwargs.items()):
        if value is None:
            continue
        path_shape = _path_annotation(annotations.get(name, inspect.Parameter.empty))
        if path_shape is None:
            continue
        kwargs[name] = _convert_paths(
            value,
            path_shape,
            Path("inputs") / name,
            storage_type,
            storage,
        )
        input_artifacts[name] = {
            "storage_type": storage_type,
            "uri": value,
        }
    return kwargs, input_artifacts


def register_nested_path_artifacts() -> None:
    """Install the recursive artifact adapter used by CalculationMCPServer."""
    calculation_mcp_server.handle_input_artifacts = handle_nested_input_artifacts
