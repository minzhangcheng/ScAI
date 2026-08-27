"""Stable cross-cutting contract for autonomous tool dispatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping


class DispatchStatus(Enum):
    LEGACY_HANDLER = "legacy_handler"
    HANDLED = "handled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True)
class ToolDispatchResult:
    status: DispatchStatus
    tool_name: str
    content: str = ""
    value: Any = None


@dataclass
class ToolDispatchContext:
    messages: list
    known_tool_names: frozenset[str]
    logger: Any
    handlers: Mapping[str, Callable[[Any], Awaitable[Any]]] = field(
        default_factory=dict
    )
    propagate_exceptions: tuple[type[BaseException], ...] = ()


def parse_error_message(tool_call) -> str:
    return (
        f"ERROR: failed to parse arguments JSON for tool '{tool_call.name}': "
        f"{tool_call.parse_error}. Raw arguments: {tool_call.raw_arguments!r}. "
        "Please retry the call with a valid JSON arguments object."
    )


def unknown_tool_message(tool_name: str, known_tool_names: frozenset[str]) -> str:
    known = ", ".join(sorted(known_tool_names))
    return (
        f"ERROR: Unknown tool '{tool_name}'. This tool does not exist. "
        f"Available tools: {known}. Please retry using one of the available tools."
    )


def append_tool_message(context: ToolDispatchContext, tool_call, content: str) -> None:
    context.messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": content}
    )


async def dispatch_tool(context: ToolDispatchContext, tool_call) -> ToolDispatchResult:
    """Apply invariant dispatch behavior, then call an installed family handler.

    A known tool without an installed handler is deliberately returned to the
    historical branch chain. This sentinel lets the turn-loop owner migrate
    handlers one family at a time without changing transcript behavior.
    """
    tool_name = tool_call.name
    if tool_call.parse_error:
        content = parse_error_message(tool_call)
        context.logger.warning("[Bioinformatics] %s", content)
        append_tool_message(context, tool_call, content)
        return ToolDispatchResult(DispatchStatus.REJECTED, tool_name, content)

    handler = context.handlers.get(tool_name)
    if handler is not None:
        try:
            value = await handler(tool_call)
        except context.propagate_exceptions:
            raise
        except Exception as exc:
            content = tool_error_message(tool_name, str(exc))
            context.logger.warning(
                "[Bioinformatics] Tool '%s' failed: %s; continuing",
                tool_name,
                exc,
            )
            append_tool_message(context, tool_call, content)
            return ToolDispatchResult(DispatchStatus.ERROR, tool_name, content)
        return ToolDispatchResult(DispatchStatus.HANDLED, tool_name, value=value)

    if tool_name in context.known_tool_names:
        return ToolDispatchResult(DispatchStatus.LEGACY_HANDLER, tool_name)

    content = unknown_tool_message(tool_name, context.known_tool_names)
    context.logger.warning(
        "[Bioinformatics] Unknown tool '%s' called (tool_call_id=%s); "
        "responding with error to keep conversation well-formed",
        tool_name,
        tool_call.id,
    )
    append_tool_message(context, tool_call, content)
    return ToolDispatchResult(DispatchStatus.REJECTED, tool_name, content)


def tool_error_message(tool_name: str, compact_error: str) -> str:
    return f"Error processing tool '{tool_name}': {compact_error}"
