"""WebSocket delivery payloads owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'serialize_realtime_message', 'strip_realtime_media_blob_references', 'strip_nested_binary_payload', '_is_media_value_key', '_is_encoded_media', 'json_safe_payload', '_copy_payload', '_is_base64_key', '_normalized_payload_key', '_cycle_placeholder', 'append_durable_realtime_message', '_json_key'})

def bind_facade(facade) -> None:
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


def _normalized_payload_key(key: Any) -> str:
    return _NON_ALNUM_RE.sub("", str(key).casefold())

def _is_media_value_key(key: Any) -> bool:
    normalized = _normalized_payload_key(key)
    return normalized in {
        "image",
        "imagepng",
        "imagejpeg",
        "imagejpg",
        "imagewebp",
        "png",
        "jpeg",
        "jpg",
        "webp",
        "thumbnail",
        "preview",
        "figuredata",
        "plotdata",
        "graphicdata",
    }

def _is_base64_key(key: Any) -> bool:
    return _normalized_payload_key(key) in _EXACT_BASE64_KEYS

def _is_encoded_media(value: str, *, key: Any = None, media_value: bool = False) -> bool:
    lowered = value.casefold()
    if lowered.startswith("data:image/") or lowered.startswith("data:application/pdf"):
        return True
    if key is not None and _is_base64_key(key):
        return bool(value)
    if media_value and value.startswith(_KNOWN_ENCODED_PREFIXES):
        return True
    return value.startswith(_KNOWN_ENCODED_PREFIXES)

def _json_key(key: Any) -> Any:
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)

def _cycle_placeholder(json_mode: bool) -> Any:
    return "[cycle]"

def _copy_payload(
    value: Any,
    *,
    seen: set[int],
    depth: int,
    strip_binary: bool,
    json_mode: bool,
    key: Any = None,
    media_value: bool = False,
) -> Any:
    """Copy a payload without allowing binary output to cross a boundary.

    Durable copies preserve ordinary Python container types so LangGraph and
    session models can continue to deserialize them. The live JSON mode is
    deliberately stricter: every value is converted to a JSON-compatible
    representation and the original object graph is never mutated.
    """
    if depth > _PAYLOAD_MAX_DEPTH:
        return _cycle_placeholder(json_mode)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return STRIPPED_FIGURE_PLACEHOLDER
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and strip_binary and _is_encoded_media(
            value, key=key, media_value=media_value
        ):
            return STRIPPED_FIGURE_PLACEHOLDER
        return value
    if isinstance(value, float):
        if json_mode and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        enum_value = value.value
        if json_mode:
            return _copy_payload(
                enum_value,
                seen=seen,
                depth=depth + 1,
                strip_binary=strip_binary,
                json_mode=True,
                key=key,
                media_value=media_value,
            )
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat() if json_mode else value
    if isinstance(value, Path):
        return str(value) if json_mode else value

    object_id = id(value)
    if object_id in seen:
        return _cycle_placeholder(json_mode)
    seen.add(object_id)
    try:
        if isinstance(value, Mapping):
            items = []
            for raw_key, raw_value in value.items():
                copied_value = _copy_payload(
                    raw_value,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=raw_key,
                    media_value=_is_media_value_key(raw_key),
                )
                items.append((_json_key(raw_key) if json_mode else raw_key, copied_value))
            if json_mode:
                return dict(items)
            try:
                return type(value)(items)
            except (TypeError, ValueError):
                return dict(items)

        if isinstance(value, list):
            return [
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            copied = tuple(
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            )
            return copied
        if isinstance(value, (set, frozenset)):
            copied_items = [
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            ]
            if json_mode:
                return sorted(copied_items, key=repr)
            return type(value)(copied_items)

        if hasattr(value, "model_copy") and hasattr(value, "model_fields"):
            updates = {
                name: _copy_payload(
                    getattr(value, name),
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=name,
                    media_value=_is_media_value_key(name),
                )
                for name in value.model_fields
                if hasattr(value, name)
            }
            if json_mode:
                return {
                    _json_key(name): copied
                    for name, copied in updates.items()
                }
            return value.model_copy(update=updates, deep=False)

        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field.name: _copy_payload(
                    getattr(value, field.name),
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=field.name,
                    media_value=_is_media_value_key(field.name),
                )
                for field in dataclass_fields(value)
            }
            if json_mode:
                return updates
            return replace(value, **updates)

        if json_mode:
            return str(value)
        return value
    finally:
        seen.discard(object_id)

def strip_nested_binary_payload(value: Any) -> Any:
    """Return a copy suitable for durable state or reconnect history."""
    return _copy_payload(
        value,
        seen=set(),
        depth=0,
        strip_binary=True,
        json_mode=False,
    )

def strip_realtime_media_blob_references(message: Any) -> Any:
    """Replace unresolved blob handles only in semantic media positions."""
    if not isinstance(message, Mapping):
        return message

    message_type = str(message.get("type") or "")

    def walk(
        value: Any,
        *,
        key: Any = None,
        direct_content: bool = False,
        in_figures: bool = False,
    ) -> Any:
        if isinstance(value, str):
            if not value.startswith(BLOB_REF_PREFIX):
                return value
            normalized_key = _normalized_payload_key(key) if key is not None else ""
            is_media = (
                (direct_content and message_type == "figure")
                or (key is not None and (_is_base64_key(key) or _is_media_value_key(key)))
                or (normalized_key == "data" and (message_type == "figure" or in_figures))
            )
            return STRIPPED_FIGURE_PLACEHOLDER if is_media else value
        if isinstance(value, Mapping):
            return {
                raw_key: walk(
                    child,
                    key=raw_key,
                    in_figures=(
                        in_figures
                        or _normalized_payload_key(raw_key) == "figures"
                    ),
                )
                for raw_key, child in value.items()
            }
        if isinstance(value, list):
            return [
                walk(item, in_figures=in_figures)
                for item in value
            ]
        return value

    return {
        key: walk(
            value,
            key=key,
            direct_content=(key == "content"),
        )
        for key, value in message.items()
    }

def append_durable_realtime_message(target: Any, message: Any) -> None:
    """Append a detached binary-free message under the shared history cap."""
    durable_content = strip_nested_binary_payload(message.content)
    if hasattr(message, "model_copy"):
        durable_message = message.model_copy(
            deep=False,
            update={"content": durable_content},
        )
    else:
        durable_message = copy.copy(message)
        durable_message.content = durable_content
    target.realtime_messages.append(durable_message)
    if len(target.realtime_messages) > _DURABLE_REALTIME_MESSAGE_LIMIT:
        target.realtime_messages = target.realtime_messages[
            -_DURABLE_REALTIME_MESSAGE_LIMIT:
        ]

def json_safe_payload(value: Any, *, strip_binary: bool = True) -> Any:
    """Return a detached JSON-safe copy for the final WebSocket boundary."""
    return _copy_payload(
        value,
        seen=set(),
        depth=0,
        strip_binary=strip_binary,
        json_mode=True,
    )

def serialize_realtime_message(
    message: Any, *, strip_binary: bool = True
) -> dict[str, Any]:
    """Serialize one realtime message without mutating its Pydantic model."""
    if hasattr(message, "model_copy") and hasattr(message, "model_fields"):
        safe = _copy_payload(
            message,
            seen=set(),
            depth=0,
            strip_binary=strip_binary,
            json_mode=True,
        )
    elif hasattr(message, "model_dump"):
        safe = json_safe_payload(
            message.model_dump(mode="python"),
            strip_binary=strip_binary,
        )
    elif isinstance(message, Mapping):
        safe = json_safe_payload(message, strip_binary=strip_binary)
    else:
        safe = json_safe_payload({
            "type": getattr(message, "type", None),
            "agent": getattr(message, "agent", None),
            "content": getattr(message, "content", None),
            "timestamp": getattr(message, "timestamp", None),
            "id": getattr(message, "id", None),
        }, strip_binary=strip_binary)
    if isinstance(safe, dict):
        return safe
    return {"content": safe}
