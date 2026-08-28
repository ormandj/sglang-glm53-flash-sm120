#!/usr/bin/env python3
"""Build-time contract for the GLM47 part of upstream SGLang #36626."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.utils import get_schema_properties


EXPECTED = {"kind": "acme", "payload": {"value": "hello"}}
TEXT = (
    "<tool_call>acme"
    "<arg_key>kind</arg_key><arg_value>acme</arg_value>"
    "<arg_key>payload</arg_key>"
    '<arg_value>{"value": "hello"}</arg_value>'
    "</tool_call>"
)


def tool(parameters):
    return Tool(
        type="function",
        function=Function(name="acme", description="Send", parameters=parameters),
    )


def parse_streaming(parameters):
    detector = Glm47MoeDetector()
    name = None
    arguments = ""
    for start in range(0, len(TEXT), 8):
        result = detector.parse_streaming_increment(
            TEXT[start : start + 8], [tool(parameters)]
        )
        for call in result.calls:
            name = call.name or name
            arguments += call.parameters
    assert name == "acme", name
    assert json.loads(arguments) == EXPECTED, arguments


flat = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "payload": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
    "required": ["kind", "payload"],
}
composite = {
    "type": "object",
    "oneOf": [
        flat,
        {
            "type": "object",
            "properties": {"kind": {"const": "other"}},
            "required": ["kind"],
        },
    ],
}

assert get_schema_properties(None) == {}
assert set(get_schema_properties(flat)) == {"kind", "payload"}
assert set(get_schema_properties(composite)) == {"kind", "payload"}
parse_streaming(flat)
parse_streaming(composite)
print("GLM47 nested and composite tool-schema streaming contract valid")
