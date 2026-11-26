#!/usr/bin/env python3
"""
Workflow vs Agent — reproducible same-ticket comparison (local, free).

Three designs answer the SAME frozen support-ticket questions using a LOCAL
Ollama model. The numbers in results.json are produced by YOUR machine, not by
me — you run this yourself and the transcripts are saved so you can read every
model response verbatim.

Designs:
  W  workflow : a router picks a known intent; code runs that intent's fixed
                tool sequence; the model writes the reply. "other" -> handoff.
  A  agent    : the model gets all tools and chooses what to call, in what order.
  H  hybrid   : same router; known intents take W's fixed path, "other" goes to
                the agent loop (read-only tools).

Usage:
  python run_experiment.py --model qwen2.5:3b --seeds 3
  python run_experiment.py --quick          # 1 seed, fast sanity check
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def ollama_chat(model, messages, tools=None, fmt=None, seed=1):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"seed": seed, "temperature": 0.0, "num_predict": 512},
    }
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    req = urllib.request.Request(
        OLLAMA + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit(
                f"\nOllama is running, but the model '{model}' is NOT pulled yet.\n"
                f"Fix:  ollama pull {model}\n"
                f"(Installed models: ollama list)\n"
            )
        raise SystemExit(f"\nOllama returned HTTP {e.code} {e.reason}.\n")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"\nCould not reach Ollama at {OLLAMA}. Is it running? (run: ollama serve)\n"
            f"Details: {e}\n"
        )
    return out, (time.time() - t0)


DATA = {}


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Look up an order by id. Returns item, status, and delivery age or shipping ETA.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_refund_policy",
                "description": "Return the store's refund policy.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cancellation_policy",
                "description": "Return the store's subscription cancellation policy.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_payments",
                "description": "Return a customer's payment/charge history with statuses.",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                },
            },
        },
    ]


def call_tool(name, args):
    if name == "get_order":
        oid = (args.get("order_id") or "").upper()
        rec = DATA["orders"].get(oid)
        return json.dumps(rec) if rec else json.dumps({"error": f"no order {oid}"})
    if name == "get_refund_policy":
        return DATA["refund_policy"]
    if name == "get_cancellation_policy":
        return DATA["cancellation_policy"]
    if name == "get_payments":
        cid = (args.get("customer_id") or "").upper()
        rec = DATA["payments"].get(cid)
        return json.dumps(rec) if rec else json.dumps({"error": f"no customer {cid}"})
    return json.dumps({"error": f"unknown tool {name}"})


def parse_tool_calls_from_text(content):
    """qwen2.5-coder returns tool calls as JSON text in `content` instead of the
    `tool_calls` field. Detect that and convert it to the standard shape."""
    c = (content or "").strip()
    if not (c.startswith("{") and c.endswith("}")):
        return None
    try:
        obj = json.loads(c)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        return [{"function": {"name": obj["name"], "arguments": obj["arguments"]}}]
    return None


def extract_order_id(question):
    m = re.search(r"(INV|ORD)-\d+", question.upper())
    return m.group(0) if m else "UNKNOWN"


ROUTER_PROMPT = (
    "Classify this customer support message into exactly one intent.\n"
    "Allowed intents: refund_status, order_status, cancellation, refund_policy, other.\n"
    "Respond ONLY with JSON like {\"intent\": \"refund_status\"}."
)


def route(model, question, seed):
    msgs = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": question},
    ]
    out, lat = ollama_chat(model, msgs, fmt="json", seed=seed)
    text = (out.get("message", {}).get("content") or "").strip()
    intent = "other"
    try:
        intent = json.loads(text).get("intent", "other")
    except json.JSONDecodeError:
        low = text.lower()
        for cand in ("refund_status", "order_status", "cancellation", "refund_policy", "other"):
            if cand in low:
                intent = cand
                break
    tokens = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
    return intent, lat, tokens, text


FIXED_PATH = {
    "refund_status": ["get_order", "get_refund_policy"],
    "order_status": ["get_order"],
    "cancellation": ["get_cancellation_policy"],
    "refund_policy": ["get_refund_policy"],
}


def run_fixed(model, question, intent, seed):
    tool_results = []
    for tool_name in FIXED_PATH.get(intent, []):
        args = {}
        if tool_name == "get_order":
            args = {"order_id": extract_order_id(question)}
        result = call_tool(tool_name, args)
        tool_results.append({"tool": tool_name, "args": args, "result": result})

    context = "\n".join(f"[{r['tool']} -> {r['result']}]" for r in tool_results)
    msgs = [
        {"role": "system", "content": "You are a support assistant. Answer the customer using ONLY the provided tool results. If a fact is unknown, say so."},
        {"role": "user", "content": question},
        {"role": "user", "content": f"Tool results:\n{context}"},
    ]
    out, lat = ollama_chat(model, msgs, seed=seed)
    answer = out.get("message", {}).get("content") or ""
    return {
        "answer": answer,
        "model_calls": 1,
        "input_tokens": out.get("prompt_eval_count", 0),
        "output_tokens": out.get("eval_count", 0),
        "latency_sec": lat,
        "tool_results": tool_results,
        "transcript": msgs + [{"role": "assistant", "content": answer}],
    }


def run_agent(model, question, seed, max_steps=5):
    msgs = [
        {"role": "system", "content": "You are a support assistant. You can call tools to look up orders, the refund policy, and payments. Investigate and answer the customer. Never invent data — use tool results only."},
        {"role": "user", "content": question},
    ]
    calls = 0
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    tool_calls_made = []
    for _ in range(max_steps):
        out, lat = ollama_chat(model, msgs, tools=make_tools(), seed=seed)
        calls += 1
        input_tokens += out.get("prompt_eval_count", 0)
        output_tokens += out.get("eval_count", 0)
        latency += lat
        msg = out.get("message", {})
        tcs = msg.get("tool_calls") or []
        if not tcs:
            parsed = parse_tool_calls_from_text(msg.get("content") or "")
            if parsed:
                tcs = parsed
                msg["content"] = ""
            else:
                answer = msg.get("content", "") or ""
                return {
                    "answer": answer,
                    "model_calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_sec": latency,
                    "tool_calls": tool_calls_made,
                    "transcript": msgs + [{"role": "assistant", "content": answer}],
                }
        msgs.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tcs})
        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = call_tool(name, args)
            tool_calls_made.append({"name": name, "args": args, "result": result})
            msgs.append({"role": "tool", "content": result})
    return {
        "answer": "(no final answer — hit step limit)",
        "model_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_sec": latency,
        "tool_calls": tool_calls_made,
        "transcript": msgs,
    }


def run_design(design, model, question, seed):
    if design == "workflow":
        intent, lat_r, toks_r, raw = route(model, question, seed)
        if intent == "other":
            return {
                "outcome": "handoff", "intent": intent, "answer": "",
                "model_calls": 1, "input_tokens": toks_r, "output_tokens": 0,
                "latency_sec": lat_r, "tool_calls": [], "transcript": [{"raw_router": raw}],
            }
        r = run_fixed(model, question, intent, seed)
        r["intent"] = intent
        r["model_calls"] += 1
        r["input_tokens"] += toks_r
        r["latency_sec"] += lat_r
        return r
    if design == "agent":
        r = run_agent(model, question, seed)
        r["intent"] = None
        return r
    if design == "hybrid":
        intent, lat_r, toks_r, raw = route(model, question, seed)
        if intent == "other":
            r = run_agent(model, question, seed)
        else:
            r = run_fixed(model, question, intent, seed)
            r["model_calls"] += 1
            r["input_tokens"] += toks_r
            r["latency_sec"] += lat_r
        r["intent"] = intent
        return r
    raise ValueError(design)


YES_WORDS = ("yes", "eligible", "can get", "you can", "still within", "within the 30")
NO_WORDS = ("no", "cannot", "not eligible", "no longer", "outside the", "past the 30")


def grade(answer, question):
    low = answer.lower()
    missing = [f for f in question["required"] if f.lower() not in low]
    facts_ok = not missing
    verdict = question.get("verdict")
    verdict_ok = True
    if verdict == "yes":
        verdict_ok = any(k in low for k in YES_WORDS)
    elif verdict == "no":
        verdict_ok = any(k in low for k in NO_WORDS)
    if not facts_ok:
        return "wrong", missing
    if not verdict_ok:
        return "facts_only", []
    return "success", []


def classify(run, question):
    if run.get("outcome") == "handoff":
        return "handoff"
    if question["kind"] == "known" and run.get("intent") is not None and run["intent"] != question["intent"]:
        return "routing_error"
    ok, _ = grade(run.get("answer", ""), question)
    return ok  # "success" or "wrong"


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed, fast sanity check")
    ap.add_argument("--dataset", default=os.path.join(HERE, "questions.json"))
    args = ap.parse_args()
    seeds = 1 if args.quick else args.seeds

    ds = load_dataset(args.dataset)
    global DATA
    DATA = ds["data"]
    questions = ds["questions"]
    dh = dataset_hash(args.dataset)

    runs_dir = os.path.join(HERE, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print("=" * 72)
    print("Workflow vs Agent — same-ticket comparison")
    print(f"model   : {args.model}")
    print(f"seeds   : {seeds}")
    print(f"dataset : {args.dataset}")
    print(f"dataset SHA256 (first 16) : {dh}   <- verify this never changes")
    print(f"questions: {len(questions)}  ({sum(1 for q in questions if q['kind']=='known')} known / {sum(1 for q in questions if q['kind']=='open')} open)")
    print("=" * 72)

    # quick reachability check
    try:
        _, _ = ollama_chat(args.model, [{"role": "user", "content": "ping"}], seed=1)
    except SystemExit as e:
        print(e)
        return
    print(f"\nOllama reachable. Running {len(questions) * 3 * seeds} conversations...\n")

    results = {}
    per_run = []
    for seed in range(1, seeds + 1):
        for q in questions:
            for design in ("workflow", "agent", "hybrid"):
                t0 = time.time()
                run = run_design(design, args.model, q["question"], seed)
                run["seed"] = seed
                run["question_id"] = q["id"]
                run["design"] = design
                run["question"] = q["question"]
                run["kind"] = q["kind"]
                run["expected_intent"] = q.get("intent")
                run["outcome"] = classify(run, q)
                if run["outcome"] == "wrong":
                    _, missing = grade(run.get("answer", ""), q)
                    run["missing_facts"] = missing
                run["wall_sec"] = round(time.time() - t0, 2)
                per_run.append(run)
                # save transcript
                fn = os.path.join(runs_dir, f"{design}_{q['id']}_seed{seed}.json")
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(run, f, indent=2, ensure_ascii=False)
                tag = {"success": "OK ", "facts_only": "F/O", "wrong": "BAD", "handoff": "H/O", "routing_error": "RTE"}[run["outcome"]]
                print(f"  [{tag}] seed{seed} {design:8s} {q['id']}: {q['question'][:45]}")

    # aggregate per design
    summary = {}
    for design in ("workflow", "agent", "hybrid"):
        rows = [r for r in per_run if r["design"] == design]
        outcomes = {"success": 0, "facts_only": 0, "wrong": 0, "handoff": 0, "routing_error": 0}
        for r in rows:
            outcomes[r["outcome"]] += 1
        total = len(rows)
        calls = [r["model_calls"] for r in rows]
        itoks = [r["input_tokens"] for r in rows]
        otoks = [r["output_tokens"] for r in rows]
        lats = [r["latency_sec"] for r in rows]
        summary[design] = {
            "runs": total,
            "success": outcomes["success"],
            "facts_only": outcomes["facts_only"],
            "wrong": outcomes["wrong"],
            "handoff": outcomes["handoff"],
            "routing_error": outcomes["routing_error"],
            "correct_answer_rate_pct": pct(outcomes["success"], total),
            "facts_retrieved_rate_pct": pct(outcomes["success"] + outcomes["facts_only"], total),
            "avg_model_calls": round(statistics.mean(calls), 2) if calls else 0,
            "avg_input_tokens": round(statistics.mean(itoks), 1) if itoks else 0,
            "avg_output_tokens": round(statistics.mean(otoks), 1) if otoks else 0,
            "median_latency_sec": round(statistics.median(lats), 2) if lats else 0,
        }

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "seeds": seeds, "dataset_sha256_16": dh,
                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "summary": summary}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("RESULTS (real numbers from your local Ollama)")
    print("success = correct facts AND correct verdict; facts_only = right facts, wrong conclusion")
    print("=" * 76)
    hdr = f"{'design':9s} {'runs':>4s} {'success':>7s} {'facts_only':>10s} {'wrong':>5s} {'handoff':>7s} {'route_err':>9s} {'correct%':>8s} {'facts%':>6s} {'calls':>5s} {'med_lat':>7s}"
    print(hdr)
    for design in ("workflow", "agent", "hybrid"):
        s = summary[design]
        print(f"{design:9s} {s['runs']:>4d} {s['success']:>7d} {s['facts_only']:>10d} {s['wrong']:>5d} {s['handoff']:>7d} {s['routing_error']:>9d} {s['correct_answer_rate_pct']:>8.1f} {s['facts_retrieved_rate_pct']:>6.1f} {s['avg_model_calls']:>5.2f} {s['median_latency_sec']:>7.2f}")
    print("\nSaved: results.json + runs/ (one transcript per conversation)")
    print("Open runs/ to read every model response verbatim — that is your proof.")


if __name__ == "__main__":
    main()
