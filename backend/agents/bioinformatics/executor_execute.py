"""Pure response assembly for the execute-code tool family."""
from __future__ import annotations


def build_execution_tool_result(code_execution, result) -> str:
    content = (
        f"Execution {'successful' if not result.error else 'failed'}.\n"
        f"Evidence execution ID: {code_execution.id}\n"
    )
    if result.output:
        content += f"Output:\n{result.output}\n"
    if result.error:
        content += f"Error: {result.error}\n"
        if "ModuleNotFoundError" in result.error:
            content += (
                "\nHint: A required package is not installed. "
                "Report the missing dependency, explain which analysis step is limited, "
                "and use an already-installed alternative only when the substitution is "
                "scientifically valid.\n"
            )
    if result.figures:
        content += f"Generated {len(result.figures)} figure(s).\n"
    return content
