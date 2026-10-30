import ast
import re
from pathlib import Path


TOOLS_ROOT = Path(__file__).parents[1] / "dptb_pilot" / "tools" / "modules"
PARAMETER_ENTRY = re.compile(
    r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:"
)


def _is_mcp_tool(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "tool"
        for decorator in node.decorator_list
    )


def _signature_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    return [argument.arg for argument in arguments if argument.arg not in {"self", "cls"}]


def _documented_parameters(docstring: str) -> set[str]:
    documented = set()
    in_parameters = False
    for line in docstring.splitlines():
        heading = line.strip().rstrip(":")
        if heading in {"Parameters", "Args", "参数"}:
            in_parameters = True
            continue
        if in_parameters and heading in {
            "Returns",
            "返回",
            "Raises",
            "抛出",
            "Notes",
            "Example",
            "Examples",
        }:
            break
        if not in_parameters:
            continue
        match = PARAMETER_ENTRY.match(line)
        if match:
            names = [name.strip() for name in match.group(1).split(",")]
            documented.update(name for name in names if name[0].islower())
    return documented


def test_mcp_tool_docstrings_document_every_signature_parameter():
    failures = []
    for path in sorted(TOOLS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_mcp_tool(node):
                continue

            documented = _documented_parameters(ast.get_docstring(node) or "")
            parameters = _signature_parameters(node)
            missing = [
                name for name in parameters if name not in documented
            ]
            extra = sorted(documented - set(parameters))
            if missing or extra:
                failures.append(
                    f"{path}:{node.lineno} {node.name}: "
                    f"missing={missing}, extra={extra}"
                )

    assert not failures, "Mismatched MCP tool docstring parameters:\n" + "\n".join(
        failures
    )
