"""The one interface every LLM backend implements:

    complete(system, prompt, schema) -> validated pydantic object

The graph never knows which model is behind it. Switching variant (2) <-> (3) is
one constructor argument.
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMBackend(Protocol):
    name: str
    n_calls: int  # incremented once per model call (including repair calls)

    def raw_json(self, system: str, prompt: str, json_schema: dict) -> str:
        """One model call that should return a JSON string matching json_schema."""
        ...


def complete(backend: LLMBackend, system: str, prompt: str, schema: type[T], max_repairs: int = 1) -> T:
    """Ask for JSON, validate it with pydantic; on failure, show the error and ask once more."""
    js = schema.model_json_schema()
    text = backend.raw_json(system, prompt, js)
    for attempt in range(max_repairs + 1):
        try:
            return schema.model_validate_json(_strip_fences(text))
        except ValidationError as e:
            if attempt == max_repairs:
                raise
            text = backend.raw_json(system, prompt + "\n\nYour previous answer was invalid:\n"
                                    f"{text[:1500]}\n\nValidation errors:\n{e}\n\nReturn corrected JSON only.", js)
    raise RuntimeError("unreachable")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    # keep only the outermost JSON object if the model added prose
    if not t.lstrip().startswith("{") and "{" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    return t


def schema_hint(js: dict) -> str:
    return json.dumps(js, separators=(",", ":"))
