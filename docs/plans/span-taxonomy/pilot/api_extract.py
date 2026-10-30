"""Stage 2 extraction through the Claude API, for the M3 cost trial (plan section 7.33).

Usage: python3 api_extract.py D ARM SCHEMA [--effort low|medium|high] [--model M] [--only U1,U2] [--batch]

Sends each candidates/<uuid>.md in run folder D to the model with the schema as a cached
system prompt, writes out/ARM/<uuid>.json, and appends one usage row per agreement to
out/ARM/_usage.tsv. --batch submits the same requests through the Message Batches API and
waits for the results. Needs the anthropic package and ANTHROPIC_API_KEY.
"""
import json, os, re, sys, time
import anthropic

CONCEPTS = ["ma.termination.fee.target", "ma.termination.fee.reverse", "ma.termination.expense_reimbursement",
            "ma.termination.outside_date", "ma.mae.definition", "ma.financing.structure", "ma.indemn.cap",
            "ma.indemn.basket", "ma.indemn.survival", "tax.election.338h10"]

RULES = """You extract provisions from one merger or acquisition agreement under the schema below.

The input is the agreement's stage 1 candidate sections, each opened by a "@@ S<n> |" marker; nothing else of the agreement is available.
Extract all ten concepts: """ + ", ".join(CONCEPTS) + """.
Use the schema's instance shape for each concept, including spans (role, section "S<n>", verbatim quote), notes and scope_reviewed.
A concept with no such provision is {"concept": ..., "finding": "absent", "scope_reviewed": [...], "notes": "..."}.

Hard rules:
- Every quote must be copied verbatim (character for character) from one line of the input, never paraphrased or stitched across lines. Section ids come from the "@@ S<n> |" markers.
- Amounts: v2.6 rule 1 - a percentage-of-price amount gets amount "not_stated" unless the text states the dollar figure itself. Do not compute figures.
- Later "clarifications" sections of the schema override earlier text.
- Write American English in notes.

Reply with one JSON object and nothing else: {"agreement": "<uuid>", "instances": [<exactly ten instances, one per concept>]}.

# Schema

"""


def system(schema_path, ttl="5m"):
    return [{"type": "text", "text": RULES + open(schema_path).read(), "cache_control": {"type": "ephemeral", "ttl": ttl}}]


def params(D, u, sys_blocks, model, effort):
    text = open(f"{D}/candidates/{u}.md").read()
    return {"model": model, "max_tokens": 32000, "system": sys_blocks, "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": f"Agreement {u}. Candidate sections:\n\n{text}"}]}


def parse(msg):
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def save(D, arm, u, msg, secs):
    out = f"{D}/out/{arm}"
    try:
        obj = parse(msg)
    except json.JSONDecodeError:
        obj = None
    if obj is not None:
        obj["agreement"] = u
        json.dump(obj, open(f"{out}/{u}.json", "w"), indent=1)
    lines = open(f"{D}/candidates/{u}.md").read().split("\n")
    quotes = [s.get(k) for i in (obj or {}).get("instances", []) for s in i.get("spans") or [] for k in ("quote", "quote_end") if s.get(k)]
    bad = sum(not any(q in ln for ln in lines) for q in quotes)
    us = msg.usage
    with open(f"{out}/_usage.tsv", "a") as f:
        f.write("\t".join(map(str, [u, us.input_tokens, us.cache_creation_input_tokens or 0, us.cache_read_input_tokens or 0,
                                    us.output_tokens, msg.stop_reason, obj is not None, len(quotes), bad, round(secs, 1)])) + "\n")
    print(u, us.input_tokens, us.cache_read_input_tokens, us.output_tokens, msg.stop_reason, f"bad_quotes={bad}/{len(quotes)}", flush=True)


def main(D, arm, schema, effort="high", model="claude-sonnet-5-5", only=None, batch=False):
    client = anthropic.Anthropic()
    os.makedirs(f"{D}/out/{arm}", exist_ok=True)
    uuids = only or [m["uuid"] for m in json.load(open(f"{D}/manifest.json"))]
    sys_blocks = system(schema, "1h" if batch else "5m")
    if not batch:
        for u in uuids:
            t = time.time()
            with client.messages.stream(**params(D, u, sys_blocks, model, effort)) as s:
                msg = s.get_final_message()
            save(D, arm, u, msg, time.time() - t)
        return
    # Batch: caching is best effort; the one-hour lifetime lets one write cover the batch.
    reqs = [{"custom_id": u, "params": params(D, u, sys_blocks, model, effort)} for u in uuids]
    b = client.messages.batches.create(requests=reqs)
    print("batch", b.id, flush=True)
    t = time.time()
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(30)
    for r in client.messages.batches.results(b.id):
        if r.result.type == "succeeded":
            save(D, arm, r.custom_id, r.result.message, time.time() - t)
        else:
            print(r.custom_id, r.result.type, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    flags = {k: v for k, v in zip(a, a[1:]) if k.startswith("--")}
    main(a[0], a[1], a[2], flags.get("--effort", "high"), flags.get("--model", "claude-sonnet-5-5"),
         flags["--only"].split(",") if "--only" in flags else None, "--batch" in a)
