"""Stage 2 extraction through the Claude API, for the M3 cost trial (plan section 7.33).

Usage: python3 api_extract.py D ARM SCHEMA [--effort low|medium|high] [--model M] [--only U1,U2] [--batch] [--budget USD]

Sends each candidates/<uuid>.md in run folder D to the model with the schema as a cached
system prompt, writes out/ARM/<uuid>.json, and appends one usage row per agreement to
out/ARM/_usage.tsv. --batch submits the same requests through the Message Batches API and
waits for the results. --budget (default 14) is a ceiling in dollars on everything logged under
D/out: a run stops before the agreement that would start past it, and a batch is refused if its
pessimistic estimate would pass it. Needs the anthropic package and ANTHROPIC_API_KEY.
"""
import glob, json, os, re, sys, time
import anthropic

CONCEPTS = ["ma.termination.fee.target", "ma.termination.fee.reverse", "ma.termination.expense_reimbursement",
            "ma.termination.outside_date", "ma.mae.definition", "ma.financing.structure", "ma.indemn.cap",
            "ma.indemn.basket", "ma.indemn.survival", "tax.election.338h10"]

# dollars per million tokens: input, output; a Haiku prompt over 100K tokens costs five times as much
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}

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
    return {"model": model, "max_tokens": 64000, "system": sys_blocks, "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": f"Agreement {u}. Candidate sections:\n\n{text}"}]}


def cost(model, inp, write, read, out, ttl, batch):
    base, output = PRICES[model]
    if model == "claude-haiku-5-5" and inp + write + read > 100_000:
        base, output = 0.50, 2.50
    usd = (inp * base + write * base * (2 if ttl == "1h" else 1.25) + read * base * 0.1 + out * output) / 1e6
    return usd / 2 if batch else usd


def spent(D):
    """Dollars logged by every arm in D; the cost is the last column of _usage.tsv."""
    return sum(float(ln.rstrip("\n").split("\t")[-1]) for f in glob.glob(f"{D}/out/*/_usage.tsv") for ln in open(f))


def estimate(D, uuids, model, schema):
    """Pessimistic dollars for a batch: 3 characters per token, 20K tokens of output, no cache reads."""
    base, output = PRICES[model]
    chars = sum(os.path.getsize(f"{D}/candidates/{u}.md") for u in uuids) + len(uuids) * os.path.getsize(schema)
    return (chars / 3 * base + len(uuids) * 20_000 * output) / 1e6 / 2


def parse(msg):
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def save(D, arm, u, msg, secs, model, ttl, batch):
    out = f"{D}/out/{arm}"
    try:
        obj = parse(msg)
    except json.JSONDecodeError:
        obj = None
    if obj is not None:
        obj["agreement"] = u
        json.dump(obj, open(f"{out}/{u}.json", "w"), indent=1)
    else:  # keep what came back: a reply that does not parse is still billed
        open(f"{out}/_failed_{u}.txt", "w").write("".join(b.text for b in msg.content if b.type == "text"))
    lines = open(f"{D}/candidates/{u}.md").read().split("\n")
    quotes = [s.get(k) for i in (obj or {}).get("instances", []) for s in i.get("spans") or [] for k in ("quote", "quote_end") if s.get(k)]
    bad = sum(not any(q in ln for ln in lines) for q in quotes)
    us = msg.usage
    usd = cost(model, us.input_tokens, us.cache_creation_input_tokens or 0, us.cache_read_input_tokens or 0, us.output_tokens, ttl, batch)
    with open(f"{out}/_usage.tsv", "a") as f:
        f.write("\t".join(map(str, [u, us.input_tokens, us.cache_creation_input_tokens or 0, us.cache_read_input_tokens or 0,
                                    us.output_tokens, msg.stop_reason, obj is not None, len(quotes), bad, round(secs, 1),
                                    model, batch, round(usd, 5)])) + "\n")
    print(u, us.input_tokens, us.cache_read_input_tokens, us.output_tokens, msg.stop_reason,
          f"bad_quotes={bad}/{len(quotes)} ${usd:.3f} total ${spent(D):.2f}", flush=True)


def main(D, arm, schema, effort="high", model="claude-sonnet-5-5", only=None, batch=False, budget=14.0):
    client = anthropic.Anthropic()
    os.makedirs(f"{D}/out/{arm}", exist_ok=True)
    uuids = only or [m["uuid"] for m in json.load(open(f"{D}/manifest.json"))]
    ttl = "1h" if batch else "5m"
    sys_blocks = system(schema, ttl)
    if not batch:
        for u in uuids:
            if spent(D) >= budget:
                sys.exit(f"budget ${budget} reached at ${spent(D):.2f}; stopped before {u}")
            t = time.time()
            with client.messages.stream(**params(D, u, sys_blocks, model, effort)) as s:
                msg = s.get_final_message()
            save(D, arm, u, msg, time.time() - t, model, ttl, batch)
        return
    est = estimate(D, uuids, model, schema)
    if spent(D) + est > budget:
        sys.exit(f"batch estimate ${est:.2f} on top of ${spent(D):.2f} passes the ${budget} budget")
    # Batch: caching is best effort; the one-hour lifetime lets one write cover the batch.
    reqs = [{"custom_id": u, "params": params(D, u, sys_blocks, model, effort)} for u in uuids]
    b = client.messages.batches.create(requests=reqs)
    print("batch", b.id, flush=True)
    t = time.time()
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(30)
    for r in client.messages.batches.results(b.id):
        if r.result.type == "succeeded":
            save(D, arm, r.custom_id, r.result.message, time.time() - t, model, ttl, batch)
        else:
            print(r.custom_id, r.result.type, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    flags = {k: v for k, v in zip(a, a[1:]) if k.startswith("--")}
    main(a[0], a[1], a[2], flags.get("--effort", "high"), flags.get("--model", "claude-sonnet-5-5"),
         flags["--only"].split(",") if "--only" in flags else None, "--batch" in a, float(flags.get("--budget", 14)))
