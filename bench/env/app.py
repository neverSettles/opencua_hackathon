from __future__ import annotations

import argparse
import html
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bench.env.seller_simulator import generate_quote
from bench.harness.io import (
    effective_unit_price,
    load_sellers,
    load_task,
    load_world,
    write_json,
)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #172026; background: #f6f7f8; }
header { background: #0f1720; color: white; padding: 14px 22px; display: flex; align-items: center; justify-content: space-between; }
nav a { color: white; margin-left: 14px; text-decoration: none; font-weight: 600; }
main { max-width: 1100px; margin: 24px auto; padding: 0 18px; }
.panel { background: white; border: 1px solid #dde2e7; border-radius: 8px; padding: 18px; margin-bottom: 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border-bottom: 1px solid #e4e8ec; padding: 9px; text-align: left; vertical-align: top; }
th { background: #f1f4f7; }
button, input[type=submit] { border: 1px solid #26313d; background: #26313d; color: white; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
input, select, textarea { border: 1px solid #cbd3dc; border-radius: 6px; padding: 7px; font: inherit; }
textarea { width: 100%; min-height: 150px; }
.muted { color: #5d6875; }
.ok { color: #116b3a; font-weight: 700; }
.warn { color: #985800; font-weight: 700; }
.small { font-size: 13px; }
"""


class BenchmarkState:
    def __init__(self, world_id: str, task_id: str, state_dir: Path):
        self.world_id = world_id
        self.task_id = task_id
        self.state_dir = state_dir
        self.world = load_world(world_id)
        self.task = load_task(world_id, task_id)
        self.sellers = load_sellers(world_id)
        self.state_path = state_dir / "final_state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = {
            "world_id": world_id,
            "task_id": task_id,
            "started_at": time.time(),
            "carts": {},
            "emails": [],
            "quotes": [],
            "accepted_quotes": [],
            "final_response": None,
            "events": [],
        }
        self.persist()

    def persist(self) -> None:
        write_json(self.state_path, self.state)

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.state["events"].append({"ts": time.time(), "kind": kind, "payload": payload})
        self.persist()


class Handler(BaseHTTPRequestHandler):
    bench: BenchmarkState

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, body: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html") -> None:
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(data, indent=2, sort_keys=True), status, "application/json")

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode()
        parsed = parse_qs(raw)
        return {key: values[-1] for key, values in parsed.items()}

    def _layout(self, title: str, body: str) -> str:
        world = self.bench.world
        task = self.bench.task
        seller_links = "".join(
            f'<a href="/store/{seller_id}">{html.escape(self.bench.sellers[seller_id]["name"])}</a>'
            for seller_id in task["config"]["allowed_sellers"]
        )
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <div><strong>Buy-Side Optimization Bench</strong> <span class="small">{html.escape(world["id"])} / {html.escape(task["id"])}</span></div>
  <nav><a href="/">Task</a>{seller_links}<a href="/email">Email</a><a href="/cart">Cart</a><a href="/final">Final Answer</a></nav>
</header>
<main>{body}</main>
</body>
</html>"""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            return self._json({"ok": True, "world_id": self.bench.world_id, "task_id": self.bench.task_id})
        if path == "/api/state":
            return self._json(self.bench.state)
        if path == "/api/task":
            return self._json({"world": self.bench.world, "task": self.bench.task})
        if path == "/":
            return self._send(self._layout("Task", self.render_home()))
        if path.startswith("/store/"):
            seller_id = path.removeprefix("/store/")
            if seller_id not in self.bench.sellers:
                return self._send("Unknown seller", HTTPStatus.NOT_FOUND)
            return self._send(self._layout(f"Store {seller_id}", self.render_store(seller_id)))
        if path == "/cart":
            return self._send(self._layout("Cart", self.render_cart()))
        if path == "/email":
            return self._send(self._layout("Email", self.render_email()))
        if path == "/final":
            return self._send(self._layout("Final Answer", self.render_final()))
        return self._send("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        form = self._form()
        if path == "/api/cart/add":
            return self.add_cart(form)
        if path == "/api/email/send":
            return self.send_email(form)
        if path == "/api/quote/accept":
            return self.accept_quote(form)
        if path == "/api/final":
            return self.submit_final(form)
        return self._send("Not found", HTTPStatus.NOT_FOUND)

    def render_home(self) -> str:
        task = self.bench.task
        intent_path = Path(__file__).resolve().parents[1] / "worlds" / self.bench.world_id / "tasks" / self.bench.task_id / "intent.md"
        intent = intent_path.read_text()
        if task["task_type"] == "pc_build":
            requirements = task["config"]["required_components"]
            req_html = "".join(f"<li>{html.escape(row['category'])}: {html.escape(json.dumps(row))}</li>" for row in requirements)
        else:
            req_html = "".join(
                f"<li>{html.escape(row['name'])}: qty {int(row['qty'])}</li>"
                for row in task["config"]["items"]
            )
        sellers_html = "".join(
            f"<li><a href='/store/{sid}'>{html.escape(self.bench.sellers[sid]['name'])}</a></li>"
            for sid in task["config"]["allowed_sellers"]
        )
        return f"""
<section class="panel">
  <h1>{html.escape(task["title"])}</h1>
  <p class="muted">{html.escape(intent)}</p>
</section>
<section class="grid">
  <div class="panel"><h2>Requested Work</h2><ul>{req_html}</ul></div>
  <div class="panel"><h2>Allowed Sellers</h2><ul>{sellers_html}</ul></div>
  <div class="panel"><h2>Finish Condition</h2><p>Add valid items to cart or accept quotes, then submit structured JSON in Final Answer.</p></div>
</section>
"""

    def render_store(self, seller_id: str) -> str:
        seller = self.bench.sellers[seller_id]
        rows = []
        for item in seller.get("inventory", []):
            discounts = ", ".join(
                f"{int(rule['min_qty'])}+ => {int(rule['discount_pct'] * 100)}% off"
                for rule in item.get("volume_discounts", [])
            )
            rows.append(
                f"""<tr>
<td>{html.escape(item["name"])}<br><span class="muted small">{html.escape(item["sku"])}</span></td>
<td>${float(item["list_price"]):.2f}</td>
<td>{int(item.get("stock", 0))}</td>
<td>{html.escape(discounts or "none")}</td>
<td>
  <form method="post" action="/api/cart/add">
    <input type="hidden" name="seller_id" value="{html.escape(seller_id)}">
    <input type="hidden" name="sku" value="{html.escape(item["sku"])}">
    <input type="number" name="qty" min="1" value="1" style="width: 72px">
    <input type="submit" value="Add">
  </form>
</td>
</tr>"""
            )
        quote = ""
        if seller.get("channels", {}).get("quote_form"):
            quote = f"""
<section class="panel">
  <h2>Request Quote From {html.escape(seller["name"])}</h2>
  <form method="post" action="/api/email/send">
    <input type="hidden" name="seller_id" value="{html.escape(seller_id)}">
    <textarea name="body">Please send your best bulk quote for the requested benchmark task.</textarea>
    <p><input type="submit" value="Send RFQ"></p>
  </form>
</section>"""
        return f"""
<section class="panel">
  <h1>{html.escape(seller["name"])}</h1>
  <p class="muted">Channel flags: {html.escape(json.dumps(seller.get("channels", {})))}</p>
</section>
<section class="panel">
  <table><thead><tr><th>Product</th><th>List Price</th><th>Stock</th><th>Discounts</th><th>Cart</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</section>
{quote}
"""

    def render_cart(self) -> str:
        rows = []
        total = 0.0
        for seller_id, lines in self.bench.state["carts"].items():
            for line in lines:
                total += float(line["line_total_usd"])
                rows.append(
                    f"<tr><td>{html.escape(seller_id)}</td><td>{html.escape(line['name'])}</td><td>{line['qty']}</td><td>${line['unit_price_usd']:.2f}</td><td>${line['line_total_usd']:.2f}</td></tr>"
                )
        for quote in self.bench.state["accepted_quotes"]:
            for line in quote["lines"]:
                total += float(line["line_total_usd"])
                rows.append(
                    f"<tr><td>{html.escape(quote['seller_id'])} quote</td><td>{html.escape(line['name'])}</td><td>{line['qty']}</td><td>${line['unit_price_usd']:.2f}</td><td>${line['line_total_usd']:.2f}</td></tr>"
                )
        if not rows:
            rows.append("<tr><td colspan='5' class='muted'>No cart lines yet.</td></tr>")
        return f"""
<section class="panel">
  <h1>Cart And Accepted Quotes</h1>
  <table><thead><tr><th>Seller</th><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>Total: ${total:.2f}</h2>
</section>
"""

    def render_email(self) -> str:
        seller_options = "".join(
            f"<option value='{html.escape(sid)}'>{html.escape(self.bench.sellers[sid]['name'])}</option>"
            for sid in self.bench.task["config"]["allowed_sellers"]
            if self.bench.sellers[sid].get("channels", {}).get("email")
        )
        email_rows = []
        for email_msg in self.bench.state["emails"]:
            email_rows.append(
                f"<tr><td>{html.escape(email_msg['direction'])}</td><td>{html.escape(email_msg['seller_id'])}</td><td>{html.escape(email_msg['body'])}</td></tr>"
            )
        quote_blocks = []
        for quote in self.bench.state["quotes"]:
            line_rows = "".join(
                f"<tr><td>{html.escape(line['name'])}</td><td>{line['qty']}</td><td>${line['unit_price_usd']:.2f}</td><td>${line['line_total_usd']:.2f}</td></tr>"
                for line in quote["lines"]
            )
            action = ""
            if quote["status"] == "open":
                action = f"""
<form method="post" action="/api/quote/accept">
  <input type="hidden" name="quote_id" value="{html.escape(quote['quote_id'])}">
  <input type="submit" value="Accept Quote">
</form>"""
            else:
                action = "<span class='ok'>Accepted</span>"
            quote_blocks.append(
                f"""<div class="panel">
<h3>{html.escape(quote['seller_name'])} Quote ${quote['total_usd']:.2f}</h3>
<p>{html.escape(quote['body'])}</p>
<table><thead><tr><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead><tbody>{line_rows}</tbody></table>
{action}
</div>"""
            )
        return f"""
<section class="panel">
  <h1>Email RFQs</h1>
  <form method="post" action="/api/email/send">
    <label>Seller <select name="seller_id">{seller_options}</select></label>
    <textarea name="body">Please send your best bulk quote for the requested benchmark task.</textarea>
    <p><input type="submit" value="Send RFQ"></p>
  </form>
</section>
<section class="panel">
  <h2>Message Log</h2>
  <table><thead><tr><th>Direction</th><th>Seller</th><th>Body</th></tr></thead><tbody>{''.join(email_rows) or "<tr><td colspan='3' class='muted'>No emails yet.</td></tr>"}</tbody></table>
</section>
{''.join(quote_blocks)}
"""

    def render_final(self) -> str:
        example = {
            "status": "SUCCESS",
            "summary": "Selected the cheapest valid cart/quotes.",
            "chosen_sellers": [],
            "estimated_total_usd": 0,
        }
        stored = self.bench.state.get("final_response")
        value = json.dumps(stored or example, indent=2)
        return f"""
<section class="panel">
  <h1>Final Answer</h1>
  <p class="muted">Submit structured JSON. The evaluator verifies real environment state, not only this response.</p>
  <form method="post" action="/api/final">
    <textarea name="response_json">{html.escape(value)}</textarea>
    <p><input type="submit" value="Submit Final Answer"></p>
  </form>
</section>
"""

    def add_cart(self, form: dict[str, str]) -> None:
        seller_id = form["seller_id"]
        sku = form["sku"]
        qty = max(1, int(float(form.get("qty", "1"))))
        seller = self.bench.sellers[seller_id]
        item = next(row for row in seller["inventory"] if row["sku"] == sku)
        unit_price = effective_unit_price(item, qty, "list")
        line = {
            "seller_id": seller_id,
            "sku": sku,
            "item_id": item.get("item_id"),
            "category": item.get("category"),
            "kind": item.get("kind"),
            "name": item["name"],
            "qty": qty,
            "unit_price_usd": unit_price,
            "line_total_usd": round(unit_price * qty, 2),
            "source": "cart",
        }
        self.bench.state["carts"].setdefault(seller_id, []).append(line)
        self.bench.event("cart_add", line)
        self._redirect("/cart")

    def send_email(self, form: dict[str, str]) -> None:
        seller_id = form["seller_id"]
        body = form.get("body", "")
        outgoing = {"direction": "outgoing", "seller_id": seller_id, "body": body, "ts": time.time()}
        self.bench.state["emails"].append(outgoing)
        quote = generate_quote(self.bench.sellers[seller_id], self.bench.task, body, len(self.bench.state["emails"]))
        incoming = {"direction": "incoming", "seller_id": seller_id, "body": quote["body"], "quote_id": quote["quote_id"], "ts": time.time()}
        self.bench.state["emails"].append(incoming)
        self.bench.state["quotes"].append(quote)
        self.bench.event("email_quote", {"outgoing": outgoing, "quote": quote})
        self._redirect("/email")

    def accept_quote(self, form: dict[str, str]) -> None:
        quote_id = form["quote_id"]
        quote = next(row for row in self.bench.state["quotes"] if row["quote_id"] == quote_id)
        quote["status"] = "accepted"
        accepted = dict(quote)
        self.bench.state["accepted_quotes"].append(accepted)
        self.bench.event("quote_accept", accepted)
        self._redirect("/cart")

    def submit_final(self, form: dict[str, str]) -> None:
        raw = form.get("response_json", "")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"status": "UNPARSEABLE", "raw": raw}
        self.bench.state["final_response"] = parsed
        self.bench.event("final_response", parsed)
        self._redirect("/final")


def run_server(world_id: str, task_id: str, state_dir: Path, host: str, port: int) -> None:
    state = BenchmarkState(world_id, task_id, state_dir)
    handler = type("BoundHandler", (Handler,), {"bench": state})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {world_id}/{task_id} at http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.world, args.task, Path(args.state_dir), args.host, args.port)


if __name__ == "__main__":
    main()
