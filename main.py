#!/usr/bin/env python3
"""
Ask a question with OpenAI or Anthropic, then evaluate the answer for
correctness, hallucination risk, and confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Any

load_dotenv()

DEBUG_LOG_PATH = "/Users/belizsoyak/llm-eval-demo/.cursor/debug-d7a532.log"
DEBUG_SESSION = "d7a532"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
    # endregion


EVAL_SYSTEM = """You are a strict evaluator. Given a user question and a candidate answer from another model, assess:
- correctness: how well the answer matches verifiable facts (not opinion). Use low/medium/high.
- hallucination_risk: likelihood the answer asserts specifics that may be invented or unverifiable. low/medium/high.
- confidence: your confidence in this evaluation itself, as low/medium/high.

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{"correctness":"low|medium|high","hallucination_risk":"low|medium|high","confidence":"low|medium|high","rationale":"one or two short sentences"}
"""

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# Matches Anthropic Messages API defaults used with Sonnet 4.x-style calls.
DEFAULT_ANTHROPIC_MAX_TOKENS = 20000
DEFAULT_ANTHROPIC_TEMPERATURE = 1.0
# Disable extended thinking so responses are plain text blocks (see Anthropic docs).
ANTHROPIC_THINKING_DISABLED: dict[str, str] = {"type": "disabled"}

# API returns 404 for some "-latest" strings; map to Anthropic snapshot model IDs.
ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
}


def chat_openai(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    r = client.chat.completions.create(model=model, messages=messages)
    return (r.choices[0].message.content or "").strip()


def chat_anthropic(messages: list[dict[str, str]], model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    rest = [m for m in messages if m["role"] != "system"]
    max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", str(DEFAULT_ANTHROPIC_MAX_TOKENS)))
    temperature = float(os.environ.get("ANTHROPIC_TEMPERATURE", str(DEFAULT_ANTHROPIC_TEMPERATURE)))
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": rest,
        "thinking": ANTHROPIC_THINKING_DISABLED,
    }
    if system:
        kwargs["system"] = system
    # region agent log
    _agent_log(
        "H3",
        "chat_anthropic:entry",
        "before messages.create",
        {
            "model": model,
            "has_system": bool(system),
            "rest_count": len(rest),
            "rest_roles": [m.get("role") for m in rest],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": "disabled",
        },
    )
    # endregion
    try:
        r = client.messages.create(**kwargs)
    except Exception as e:
        # region agent log
        info: dict[str, Any] = {"exc_type": type(e).__name__, "exc_msg": str(e)[:800]}
        for attr in ("status_code", "type"):
            v = getattr(e, attr, None)
            if v is not None:
                info[attr] = str(v)
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                info["response_text_snippet"] = resp.text[:400] if hasattr(resp, "text") else str(resp)[:400]
            except Exception:
                info["response_text_snippet"] = "(unreadable)"
        _agent_log("H1", "chat_anthropic:create_failed", "messages.create raised", info)
        # endregion
        raise
    block_types = [str(getattr(b, "type", type(b).__name__)) for b in r.content]
    # region agent log
    _agent_log(
        "H3",
        "chat_anthropic:success",
        "after messages.create",
        {"block_types": block_types, "n_blocks": len(r.content)},
    )
    # endregion
    parts = []
    for block in r.content:
        if block.type == "text":
            parts.append(block.text)
    out = "".join(parts).strip()
    # region agent log
    _agent_log("H3", "chat_anthropic:text_extracted", "assembled assistant text", {"out_len": len(out)})
    # endregion
    return out


def parse_eval_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def save_log(question: str, answer: str, evaluation: dict[str, Any]) -> None:
    # region run persistence
    try:
        log_path = os.path.join(os.path.dirname(__file__), "logs.jsonl")
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never break the main program output if file logging fails.
        pass
    # endregion


def main() -> None:
    parser = argparse.ArgumentParser(description="QA with auto-evaluation (OpenAI or Anthropic).")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default=os.environ.get("LLM_PROVIDER", "openai"),
    )
    parser.add_argument("--question", "-q", help="Question to ask (omit to type interactively)")
    parser.add_argument("--model", "-m", help="Override model name")
    args = parser.parse_args()

    question = (args.question or "").strip()
    if not question:
        question = input("Your question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    # region agent log
    _agent_log(
        "H2",
        "main:resolved_inputs",
        "args and key presence",
        {
            "provider": args.provider,
            "model_cli": args.model,
            "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "question_len": len(question),
        },
    )
    # endregion

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY.", file=sys.stderr)
            sys.exit(1)
        model = args.model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        chat = lambda msgs: chat_openai(msgs, model)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            # region agent log
            _agent_log("H2", "main:anthropic_exit", "ANTHROPIC_API_KEY missing", {})
            # endregion
            print("Set ANTHROPIC_API_KEY.", file=sys.stderr)
            sys.exit(1)
        requested = args.model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        model = ANTHROPIC_MODEL_ALIASES.get(requested, requested)
        # region agent log
        _agent_log(
            "H5",
            "main:anthropic_model_resolved",
            "model used for Anthropic",
            {
                "requested_model": requested,
                "api_model": model,
                "alias_remapped": requested != model,
                "from_cli": bool(args.model),
            },
        )
        # endregion
        chat = lambda msgs: chat_anthropic(msgs, model)

    answer_messages = [
        {"role": "system", "content": "Answer clearly and concisely. If unsure, say so."},
        {"role": "user", "content": question},
    ]
    answer = chat(answer_messages)

    eval_user = (
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return only the JSON object described in your instructions."
    )
    eval_messages = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": eval_user},
    ]
    eval_raw = chat(eval_messages)

    # region agent log
    _agent_log(
        "H4",
        "main:both_calls_ok",
        "answer and eval completed",
        {"answer_len": len(answer), "eval_raw_len": len(eval_raw)},
    )
    # endregion

    try:
        ev = parse_eval_json(eval_raw)
    except json.JSONDecodeError:
        ev = {
            "correctness": "?",
            "hallucination_risk": "?",
            "confidence": "?",
            "rationale": eval_raw,
        }

    save_log(question, answer, ev)

    line = "=" * 60
    print(line)
    print("QUESTION")
    print(line)
    print(question)
    print()
    print(line)
    print("ANSWER")
    print(line)
    print(answer)
    print()
    print(line)
    print("EVALUATION")
    print(line)
    print(f"  Correctness:         {ev.get('correctness', '?')}")
    print(f"  Hallucination risk:  {ev.get('hallucination_risk', '?')}")
    print(f"  Confidence:          {ev.get('confidence', '?')}")
    r = ev.get("rationale", "")
    if r:
        print(f"  Rationale:           {r}")



if __name__ == "__main__":
    # region agent log
    try:
        main()
    except Exception as e:
        _agent_log(
            "H4",
            "main:unhandled",
            "uncaught at top level",
            {"exc_type": type(e).__name__, "exc_msg": str(e)[:800]},
        )
        raise
    # endregion
