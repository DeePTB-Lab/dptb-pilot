import ast
from pathlib import Path


TOOLS_ROOT = Path(__file__).parents[1] / "dptb_pilot" / "tools" / "modules"


def _is_mcp_tool(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "tool"
        for decorator in node.decorator_list
    )


def test_mcp_tool_signatures_use_fermi_level_terminology():
    legacy_names = []
    for path in sorted(TOOLS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_mcp_tool(node):
                continue

            parameter_names = [
                argument.arg
                for argument in [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
            ]
            legacy_names.extend(
                f"{path}:{node.lineno} {node.name}({parameter_name})"
                for parameter_name in parameter_names
                if "efermi" in parameter_name.lower()
            )
            if "efermi" in node.name.lower():
                legacy_names.append(f"{path}:{node.lineno} {node.name}")

    assert not legacy_names, "Legacy MCP tool names:\n" + "\n".join(legacy_names)
