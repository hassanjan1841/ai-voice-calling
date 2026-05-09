from typing import Optional

_store: dict[str, list[dict]] = {}


def get_history(conversation_id: str) -> list[dict]:
    if conversation_id not in _store:
        _store[conversation_id] = []
    return _store[conversation_id]


def append_message(conversation_id: str, role: str, content: str):
    if conversation_id not in _store:
        _store[conversation_id] = []
    _store[conversation_id].append({"role": role, "content": content})
