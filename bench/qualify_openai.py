#!/usr/bin/env python3
"""Small, dependency-free OpenAI streaming qualification client.

The API key is read only from SGLANG_API_KEY and is never written to evidence.
Each request records TTFT, total latency, reported token usage, finish reason,
and the complete model output. Vision inputs are represented in evidence by
their path, byte count, and SHA-256 rather than by the embedded data URL.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CALIBRATION_PREFIX = (
    "Review the following deterministic engineering ledger, identify the most "
    "important runtime invariants, and propose a concrete validation plan. "
    "Do not quote the ledger verbatim.\n\n"
)
CALIBRATION_PADDING = (
    "The engineering ledger records deterministic identifiers, timestamps, "
    "integer counters, cache-lifecycle transitions, and short implementation "
    "notes for later verification. "
)
DISTINCT_WORKSTREAMS = (
    "Analyze radix-cache ownership and page-aligned request lifetimes.",
    "Analyze CUDA graph and workspace memory without changing model values.",
    "Analyze packed FP8 KV layout invariants across target and draft paths.",
    "Analyze recurrent-state admission, rollback, and eviction invariants.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm53-flash")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument(
        "--target-prompt-tokens",
        type=int,
        help="Generate deterministic text calibrated with the server tokenizer",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        help="GLM reasoning budget; omitted to use the model default",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--distinct-suffixes",
        action="store_true",
        help="Append a deterministic per-request workstream after the shared prefix",
    )
    parser.add_argument(
        "--suffix-salt",
        help="Append a recorded wave discriminator to every distinct workstream",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.repetitions < 1:
        parser.error("concurrency and repetitions must be positive")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.target_prompt_tokens is not None and args.target_prompt_tokens < 1:
        parser.error("target-prompt-tokens must be positive")
    args.prompt_evidence = None
    if args.prompt_file is not None:
        raw = args.prompt_file.read_bytes()
        args.prompt = raw.decode("utf-8")
        args.prompt_evidence = {
            "path": str(args.prompt_file.resolve()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return args


def post_json(*, url: str, api_key: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def make_calibrated_prompt(
    *, base_url: str, model: str, api_key: str, target_tokens: int, timeout: float
) -> tuple[str, dict[str, Any]]:
    target_chars = max(1, target_tokens * 6)
    prompt = ""
    token_count = 0
    for _ in range(5):
        repeats = target_chars // len(CALIBRATION_PADDING) + 1
        ledger = (CALIBRATION_PADDING * repeats)[:target_chars]
        prompt = CALIBRATION_PREFIX + ledger
        tokenized = post_json(
            url=base_url.rstrip("/") + "/tokenize",
            api_key=api_key,
            payload={"model": model, "prompt": prompt},
            timeout=timeout,
        )
        token_count = int(tokenized["count"])
        if abs(token_count - target_tokens) <= 16:
            break
        target_chars = max(1, round(target_chars * target_tokens / token_count))

    raw = prompt.encode("utf-8")
    return prompt, {
        "generator": "deterministic-engineering-ledger-v1",
        "target_tokens": target_tokens,
        "calibrated_tokens": token_count,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def image_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    content = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }
    evidence = {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "mime_type": mime,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return content, evidence


def make_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    image_evidence = None
    if args.image is None:
        content: Any = args.prompt
    else:
        image, image_evidence = image_payload(args.image)
        content = [image, {"type": "text", "text": args.prompt}]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_spec_tokens_details": True,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    return payload, image_evidence


def payload_for_ordinal(
    payload: dict[str, Any],
    *,
    ordinal: int,
    distinct_suffixes: bool,
    suffix_salt: str | None,
) -> dict[str, Any]:
    if not distinct_suffixes:
        return payload

    request_payload = json.loads(json.dumps(payload))
    suffix = DISTINCT_WORKSTREAMS[ordinal % len(DISTINCT_WORKSTREAMS)]
    if suffix_salt is not None:
        suffix = f"{suffix} Wave discriminator: {suffix_salt}."
    content = request_payload["messages"][0]["content"]
    if isinstance(content, str):
        request_payload["messages"][0]["content"] = f"{content}\n\n{suffix}"
    else:
        text_part = next(part for part in reversed(content) if part.get("type") == "text")
        text_part["text"] = f"{text_part['text']}\n\n{suffix}"
    return request_payload


def one_request(
    *,
    ordinal: int,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started_wall = time.time()
    started = time.perf_counter()
    first_delta_at = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    finish_reason = None
    response_id = None
    model = None
    sgl_ext = None
    chunk_count = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                chunk_count += 1
                response_id = response_id or chunk.get("id")
                model = model or chunk.get("model")
                if chunk.get("sgl_ext") is not None:
                    sgl_ext = chunk["sgl_ext"]
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                text = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if text:
                    content_parts.append(text)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if first_delta_at is None and (text or reasoning):
                    first_delta_at = time.perf_counter()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body[:2000]}") from exc

    ended = time.perf_counter()
    completion_tokens = None if usage is None else usage.get("completion_tokens")
    decode_seconds = None if first_delta_at is None else ended - first_delta_at
    output_tps = None
    if completion_tokens is not None and decode_seconds and decode_seconds > 0:
        output_tps = completion_tokens / decode_seconds
    return {
        "ordinal": ordinal,
        "started_unix": started_wall,
        "http_status": status,
        "response_id": response_id,
        "model": model,
        "ttft_seconds": None if first_delta_at is None else first_delta_at - started,
        "latency_seconds": ended - started,
        "decode_seconds_from_first_delta": decode_seconds,
        "reported_output_tokens_per_second": output_tps,
        "chunk_count": chunk_count,
        "usage": usage,
        "sgl_ext": sgl_ext,
        "finish_reason": finish_reason,
        "reasoning_content": "".join(reasoning_parts),
        "content": "".join(content_parts),
    }


def finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ttfts = finite([result["ttft_seconds"] for result in results])
    latencies = finite([result["latency_seconds"] for result in results])
    rates = finite([result["reported_output_tokens_per_second"] for result in results])
    completion_tokens = finite(
        [
            None if result["usage"] is None else result["usage"].get("completion_tokens")
            for result in results
        ]
    )
    return {
        "requests": len(results),
        "mean_ttft_seconds": statistics.fmean(ttfts) if ttfts else None,
        "max_ttft_seconds": max(ttfts) if ttfts else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "mean_reported_output_tokens_per_second": statistics.fmean(rates) if rates else None,
        "aggregate_completion_tokens": int(sum(completion_tokens)),
        "wall_seconds": None,
        "aggregate_output_tokens_per_second": None,
    }


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("SGLANG_API_KEY")
    if not api_key:
        print("SGLANG_API_KEY is required", file=sys.stderr)
        return 2
    if args.target_prompt_tokens is not None:
        args.prompt, args.prompt_evidence = make_calibrated_prompt(
            base_url=args.url,
            model=args.model,
            api_key=api_key,
            target_tokens=args.target_prompt_tokens,
            timeout=args.timeout,
        )
    payload, image_evidence = make_payload(args)
    total = args.concurrency * args.repetitions
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                one_request,
                ordinal=ordinal,
                url=args.url,
                api_key=api_key,
                payload=payload_for_ordinal(
                    payload,
                    ordinal=ordinal,
                    distinct_suffixes=args.distinct_suffixes,
                    suffix_salt=args.suffix_salt,
                ),
                timeout=args.timeout,
            )
            for ordinal in range(total)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - wall_start
    results.sort(key=lambda result: result["ordinal"])
    totals = summary(results)
    totals["wall_seconds"] = wall_seconds
    if wall_seconds > 0:
        totals["aggregate_output_tokens_per_second"] = (
            totals["aggregate_completion_tokens"] / wall_seconds
        )

    evidence = {
        "schema": 1,
        "client": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "request": {
            "url": args.url,
            "model": args.model,
            "prompt": args.prompt if args.prompt_evidence is None else None,
            "prompt_file": args.prompt_evidence,
            "image": image_evidence,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "thinking_enabled": not args.disable_thinking,
            "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency,
            "repetitions": args.repetitions,
            "distinct_suffixes": args.distinct_suffixes,
            "suffix_salt": args.suffix_salt,
            "stream": True,
        },
        "summary": totals,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
