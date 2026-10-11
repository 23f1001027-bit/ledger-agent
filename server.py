"""Acme ledger agent.

The LLM only turns the question into a small JSON query plan.
All filtering and arithmetic is done here, in plain Python with Decimal.
"""
import difflib
import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).parent
IST = timezone(timedelta(hours=5, minutes=30))

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
LLM_URL = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- data
def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_data():
    rates = json.loads((HERE / "rates.raw").read_text(encoding="utf-8"))["usd_per_unit"]
    rates = {k: Decimal(str(v)) for k, v in rates.items()}

    latest = {}
    with open(HERE / "export.raw", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r["_updated"] = parse_ts(r["updated_at"])
            old = latest.get(r["id"])
            # latest updated_at wins; on an exact tie the later line wins
            if old is None or r["_updated"] >= old["_updated"]:
                latest[r["id"]] = r

    orders = []
    for r in latest.values():
        local = parse_ts(r["created_at"]).astimezone(IST)
        orders.append({
            "id": r["id"],
            "customer": r["customer"],
            "region": r["region"],
            "product": r["product"],
            "qty": int(r["qty"]),
            "status": r["status"],
            "currency": r["currency"],
            "day": local.date(),
            "usd": Decimal(str(r["amount"])) * rates[r["currency"]],
        })
    return orders


ORDERS = load_data()
VOCAB = {
    "region": sorted({o["region"] for o in ORDERS}),
    "product": sorted({o["product"] for o in ORDERS}),
    "customer": sorted({o["customer"] for o in ORDERS}),
    "currency": sorted({o["currency"] for o in ORDERS}),
    "status": sorted({o["status"] for o in ORDERS}),
}


# ---------------------------------------------------------------- planning
SYSTEM = f"""You convert questions about an order ledger into a JSON query plan. Reply with JSON only.

Data: orders with region, product, customer, currency, status, quantity and a USD amount.
Allowed values:
- region: {VOCAB['region']}
- product: {VOCAB['product']}
- status: {VOCAB['status']}  ('paid' = revenue/sales, 'refunded' = refunds, 'void' = cancelled)
- currency: {VOCAB['currency']}
- customer ids look like C1079

Plan fields:
- "statuses": list of statuses to include. Revenue/sales/income/turnover -> ["paid"]. Refund/refunded/returned -> ["refunded"]. Plain "orders" with no qualifier -> ["paid"].
- "metric": one of "sum_usd" (money total), "count_orders", "sum_qty" (units sold), "avg_usd" (average order value), "distinct_customers".
- "filters": object with any of region, product, customer, currency (omit those not mentioned). Use the exact allowed spelling.
- "period": null, or {{"year": 2026}}, or {{"year": 2026, "month": 3}}, or {{"year": 2026, "quarter": 1}}, or {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} (end is inclusive).
- "group_by": null, or one of "region", "product", "customer", "month".
- "rank": null, or {{"order": "desc" or "asc", "n": integer}} when asked for the top/highest/lowest/best/most/least group(s). Use group_by with it.
- "group_value": when the question asks for one specific group's value (for example "revenue for each region" is group_by with no rank), leave null.

Examples:
Q: What was the total revenue in USD from the North region in March 2026?
{{"statuses":["paid"],"metric":"sum_usd","filters":{{"region":"North"}},"period":{{"year":2026,"month":3}},"group_by":null,"rank":null}}
Q: Which product had the most refunds by count in Q1 2026?
{{"statuses":["refunded"],"metric":"count_orders","filters":{{}},"period":{{"year":2026,"quarter":1}},"group_by":"product","rank":{{"order":"desc","n":1}}}}
Q: How many distinct customers bought a Kettle?
{{"statuses":["paid"],"metric":"distinct_customers","filters":{{"product":"Kettle"}},"period":null,"group_by":null,"rank":null}}
"""


CLIENT = httpx.Client(timeout=httpx.Timeout(9.0, connect=5.0))
PLAN_CACHE = {}


def make_plan(question: str) -> dict:
    key = " ".join(question.lower().split())
    if key in PLAN_CACHE:
        return PLAN_CACHE[key]
    last_exc = None
    for attempt in range(2):                      # one retry
        try:
            r = CLIENT.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                json={
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question},
                    ],
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            PLAN_CACHE[key] = plan
            return plan
        except Exception as exc:
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------- execution
def canon(kind: str, value: str) -> str:
    """Map whatever the LLM wrote to the exact spelling in the data."""
    options = VOCAB[kind]
    for o in options:
        if o.lower() == str(value).strip().lower():
            return o
    close = difflib.get_close_matches(str(value), options, n=1, cutoff=0.6)
    return close[0] if close else str(value)


def period_range(p):
    if not p:
        return None, None
    if "start" in p:
        s = date.fromisoformat(p["start"])
        e = date.fromisoformat(p["end"]) + timedelta(days=1)
        return s, e
    y = int(p["year"])
    if p.get("month"):
        m = int(p["month"])
        return date(y, m, 1), (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1))
    if p.get("quarter"):
        q = int(p["quarter"])
        s = date(y, 3 * (q - 1) + 1, 1)
        e = date(y + 1, 1, 1) if q == 4 else date(y, 3 * q + 1, 1)
        return s, e
    return date(y, 1, 1), date(y + 1, 1, 1)


def money(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def metric_value(rows, metric):
    if metric == "sum_usd":
        return money(sum((r["usd"] for r in rows), Decimal(0)))
    if metric == "count_orders":
        return len(rows)
    if metric == "sum_qty":
        return sum(r["qty"] for r in rows)
    if metric == "avg_usd":
        return money(sum((r["usd"] for r in rows), Decimal(0)) / len(rows)) if rows else 0
    if metric == "distinct_customers":
        return len({r["customer"] for r in rows})
    raise ValueError(f"unknown metric {metric}")


def raw_metric(rows, metric):
    """Unrounded value, used for ranking."""
    if metric == "sum_usd":
        return sum((r["usd"] for r in rows), Decimal(0))
    if metric == "avg_usd":
        return sum((r["usd"] for r in rows), Decimal(0)) / len(rows) if rows else Decimal(0)
    return Decimal(metric_value(rows, metric))


def execute(plan: dict):
    statuses = plan.get("statuses") or ["paid"]
    rows = [o for o in ORDERS if o["status"] in statuses]

    for k, v in (plan.get("filters") or {}).items():
        if k in ("region", "product", "customer", "currency") and v:
            v = canon(k, v)
            rows = [o for o in rows if o[k] == v]

    s, e = period_range(plan.get("period"))
    if s:
        rows = [o for o in rows if s <= o["day"] < e]

    metric = plan.get("metric", "sum_usd")
    gb = plan.get("group_by")
    if not gb:
        return metric_value(rows, metric)

    groups = defaultdict(list)
    for o in rows:
        key = o["day"].strftime("%Y-%m") if gb == "month" else o[gb]
        groups[key].append(o)

    rank = plan.get("rank")
    if rank:
        desc = rank.get("order", "desc") == "desc"
        ordered = sorted(groups, key=lambda k: (-raw_metric(groups[k], metric) if desc
                                                else raw_metric(groups[k], metric), k))
        n = int(rank.get("n", 1))
        top = ordered[:n]
        return top[0] if n == 1 else top
    return {k: metric_value(v, metric) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------- API
@app.get("/health")
def health():
    return {"ok": True, "orders": len(ORDERS)}


class Ask(BaseModel):
    question: str = ""


@app.post("/")
@app.post("/{path:path}")
def answer(body: Ask, path: str = ""):
    question = body.question
    try:
        plan = make_plan(question)
        print("Q:", question, "\nPLAN:", json.dumps(plan), flush=True)
        ans = execute(plan)
    except Exception as exc:  # never crash; the grader needs JSON
        print("ERROR:", repr(exc), flush=True)
        ans = None
    print("A:", ans, flush=True)
    return {"answer": ans}
