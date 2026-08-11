import argparse
import json
import re

import torch
from tokenizers import Tokenizer

from generate import generate, load_model
from sft_train import ASSISTANT_TAG, SQUAD_INSTRUCTION, SYSTEM_TAG, USER_TAG
from web_search import search as web_search_live

# Same system-message shape smoltalk's apigen-80k trains on (tool schema as
# JSON, <tool_call>[...]</tool_call> as the required output format) -- a
# from-scratch phrasing here would be off-distribution for what the SFT
# checkpoint actually learned. Note apigen-80k's assistant turns are *only*
# ever a tool call, never a direct answer using a result -- there's no
# "here's the function result, now answer" turn in that data. That's why
# this is a two-phase loop rather than one continuous generation: phase 2
# reuses squad_v2's grounded-answer-or-hedge template (SQUAD_INSTRUCTION),
# which *is* exactly "here's retrieved text, answer or say you can't" --
# the two datasets were chosen to compose this way.
TOOL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>[{"name": "web_search", "description": "Searches the web and returns a list of results with title, url, and snippet.", "parameters": {"query": {"description": "The search query.", "type": "str"}}}]</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>"""

MOCK_RESULTS = [
    {
        "title": "[MOCK] Example Result 1",
        "url": "https://example.com/1",
        "snippet": "This is a canned mock search result used to test the agent loop "
        "without a live SearXNG instance deployed.",
    },
    {
        "title": "[MOCK] Example Result 2",
        "url": "https://example.com/2",
        "snippet": "A second canned result, so the loop's multi-result formatting path gets exercised too.",
    },
]


def parse_tool_call(text):
    """Returns a list of {"name", "arguments"} dicts, or None if the model
    didn't produce a well-formed tool call (missing tags, invalid JSON, or
    an explicit empty list) -- any of which should fall through to treating
    the raw text as a direct answer instead of erroring out. A 110M model's
    tool-call formatting will not always be well-formed; this is the
    expected failure mode, not an edge case to special-case away.

    Returns (calls, attempted): `calls` is a list of parsed tool calls or
    None; `attempted` is True whenever a <tool_call> tag was present at all,
    even if what's inside it didn't parse -- callers need this distinction,
    because "no tag" (a genuine direct answer) and "tag present but broken"
    (garbage that should not be shown to the user as if it were an answer)
    look identical if you only check whether `calls` came back empty."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None, False
    try:
        calls = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, True
    if not calls or not isinstance(calls, list):
        return None, True
    return calls, True


def run(model, tokenizer, question, gen_kwargs, search_base_url, mock):
    phase1_text = SYSTEM_TAG + TOOL_SYSTEM_PROMPT + "\n\n" + USER_TAG + question + "\n\n" + ASSISTANT_TAG
    phase1_ids = tokenizer.encode(phase1_text).ids
    phase1_out = generate(model, tokenizer, phase1_ids, **gen_kwargs)
    phase1_text_out = tokenizer.decode(phase1_out)

    calls, attempted = parse_tool_call(phase1_text_out)
    if not calls:
        if attempted:
            return {
                "phase": "tool_call_malformed",
                "raw_output": phase1_text_out,
                "answer": "(the model tried to call a tool but didn't produce a well-formed call -- no answer was generated)",
            }
        return {"phase": "direct", "raw_output": phase1_text_out, "answer": phase1_text_out}

    call = calls[0]
    query = call.get("arguments", {}).get("query", question) if isinstance(call, dict) else question

    if mock:
        results = MOCK_RESULTS
    else:
        try:
            results = web_search_live(query, base_url=search_base_url)
        except Exception as e:
            return {
                "phase": "tool_error",
                "tool_call": call,
                "error": str(e),
                "answer": f"(couldn't reach the search tool: {e})",
            }

    passage = "\n".join(f"{r['title']}: {r['snippet']}" for r in results) if results else "(no results found)"
    phase2_text = (
        USER_TAG + SQUAD_INSTRUCTION.format(context=passage, question=question) + "\n\n" + ASSISTANT_TAG
    )
    phase2_ids = tokenizer.encode(phase2_text).ids
    phase2_out = generate(model, tokenizer, phase2_ids, **gen_kwargs)
    answer = tokenizer.decode(phase2_out)

    return {
        "phase": "tool_used",
        "tool_call": call,
        "query": query,
        "search_results": results,
        "answer": answer,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=3072)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search-base-url", type=str, default="http://localhost:8888")
    p.add_argument("--mock", action="store_true", help="use canned search results instead of a live SearXNG instance")
    p.add_argument("question", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>")
    model = load_model(args.checkpoint, args, device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_id=eos_id,
        seq_length=args.seq_length,
        device=device,
    )

    result = run(model, tokenizer, args.question, gen_kwargs, args.search_base_url, args.mock)

    print(f"[phase] {result['phase']}")
    if "tool_call" in result:
        print(f"[tool_call] {result['tool_call']}")
    if "search_results" in result:
        for r in result["search_results"]:
            print(f"  - {r['title']}: {r['snippet']}")
    print(f"[answer]\n{result['answer']}")


if __name__ == "__main__":
    main()
