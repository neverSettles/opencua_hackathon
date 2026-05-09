"""T4 used-books-basket optimizer.

Enumerates all valid sourcing assignments across 4 sellers for 6 books and
finds the minimum-cost one given each seller's shipping rule. Also computes a
naive baseline (each book at its individually-cheapest seller, shipping
applied as if buying one book) so we can verify the optimization matters.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

SELLERS = ("AbeBooks", "ThriftBooks", "BetterWorldBooks", "Powells")

# Conditions we treat as eligible per the spec ("Good or better").
ELIGIBLE_CONDITIONS = {
    "Good", "Very Good", "Like New", "New", "As new", "Fine",
    "Used Good", "Used Very Good", "Used Like New",
}


def cheapest_eligible(book):
    """Return {seller: best_listing_or_None} where 'best' minimizes price (+ shipping for AbeBooks)."""
    out = {}
    for seller in SELLERS:
        listings = book["listings"].get(seller, [])
        eligible = [l for l in listings if l.get("condition") in ELIGIBLE_CONDITIONS]
        if not eligible:
            out[seller] = None
            continue
        if seller == "AbeBooks":
            best = min(eligible, key=lambda l: l["price"] + l.get("shipping", 0.0))
        else:
            best = min(eligible, key=lambda l: l["price"])
        out[seller] = best
    return out


def order_total(seller, listings):
    """Compute the total an order at a given seller costs given its listings."""
    if not listings:
        return 0.0
    subtotal = sum(l["price"] for l in listings)
    if seller == "AbeBooks":
        return subtotal + sum(l.get("shipping", 0.0) for l in listings)
    if seller == "ThriftBooks":
        return subtotal + (0.0 if subtotal >= 15.0 else 4.99)
    if seller == "BetterWorldBooks":
        return subtotal
    if seller == "Powells":
        return subtotal + 4.99
    raise ValueError(f"Unknown seller: {seller}")


def compute_total(assignment, book_options):
    by_seller = {s: [] for s in SELLERS}
    for book_idx, seller in enumerate(assignment):
        listing = book_options[book_idx].get(seller)
        if listing is None:
            return None
        by_seller[seller].append(listing)
    return round(sum(order_total(s, by_seller[s]) for s in SELLERS), 2)


def shipping_attribution(assignment, book_options):
    """Per-book shipping attribution for the agent-output schema (split equal shares)."""
    by_seller_idx = {s: [] for s in SELLERS}
    for i, seller in enumerate(assignment):
        by_seller_idx[seller].append(i)
    attribution = [0.0] * len(assignment)
    for seller, idxs in by_seller_idx.items():
        if not idxs:
            continue
        listings = [book_options[i][seller] for i in idxs]
        if seller == "AbeBooks":
            for i, l in zip(idxs, listings):
                attribution[i] = round(l.get("shipping", 0.0), 2)
        else:
            seller_ship = order_total(seller, listings) - sum(l["price"] for l in listings)
            per_book = round(seller_ship / len(idxs), 2) if idxs else 0.0
            for i in idxs:
                attribution[i] = per_book
    return attribution


def naive_assignment(book_options):
    """Naive: each book at the seller minimizing its single-book cost."""
    def naive_cost(seller, listing):
        if seller == "AbeBooks":
            return listing["price"] + listing.get("shipping", 0.0)
        if seller == "ThriftBooks":
            return listing["price"] + (0.0 if listing["price"] >= 15.0 else 4.99)
        if seller == "BetterWorldBooks":
            return listing["price"]
        if seller == "Powells":
            return listing["price"] + 4.99
    out = []
    for opts in book_options:
        avail = {s: l for s, l in opts.items() if l is not None}
        best = min(avail.keys(), key=lambda s: naive_cost(s, avail[s]))
        out.append(best)
    return tuple(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ground_truth/T4_ground_truth.json")
    parser.add_argument("--inplace", action="store_true",
                        help="Write optimal_sourcing/optimal_total_cost/naive_total_cost back into the input JSON.")
    args = parser.parse_args()

    path = Path(args.input)
    gt = json.loads(path.read_text())
    books = gt["books"]
    book_options = [cheapest_eligible(b) for b in books]

    # Sanity: print availability matrix.
    print("Availability (cheapest eligible listing per book per seller):")
    for i, b in enumerate(books):
        print(f"  {b['isbn']} {b['title'][:40]:<40}", end="  ")
        for s in SELLERS:
            l = book_options[i][s]
            if l is None:
                print(f"{s[:6]}: ----   ", end=" ")
            else:
                ship = l.get("shipping", 0.0) if s == "AbeBooks" else 0.0
                print(f"{s[:6]}: ${l['price']:5.2f}{('+'+str(ship)) if ship else '   '}", end=" ")
        print()

    # Enumerate all 4^N assignments.
    best_total, best_asg = None, None
    for asg in itertools.product(SELLERS, repeat=len(books)):
        total = compute_total(asg, book_options)
        if total is None:
            continue
        if best_total is None or total < best_total:
            best_total, best_asg = total, asg

    naive = naive_assignment(book_options)
    naive_total = compute_total(naive, book_options)
    gap_pct = (naive_total - best_total) / naive_total * 100 if naive_total else 0.0

    print("\nOPTIMAL")
    for i, seller in enumerate(best_asg):
        l = book_options[i][seller]
        ship_extra = f" + ${l.get('shipping', 0.0):.2f} ship" if seller == "AbeBooks" else ""
        print(f"  {books[i]['title'][:42]:<42} -> {seller:<17} ${l['price']:5.2f} ({l.get('condition','?')}){ship_extra}")
    print(f"  Total: ${best_total:.2f}")

    print("\nNAIVE (each book at its single-book-cheapest seller)")
    for i, seller in enumerate(naive):
        l = book_options[i][seller]
        print(f"  {books[i]['title'][:42]:<42} -> {seller}")
    print(f"  Total: ${naive_total:.2f}")

    print(f"\nGap (naive-optimal)/naive = {gap_pct:.1f}%   "
          f"({'>=10% ✓' if gap_pct >= 10 else 'BELOW 10% — task may not exercise the optimization'})")

    if args.inplace:
        attribution = shipping_attribution(best_asg, book_options)
        sourcing = {}
        for i, seller in enumerate(best_asg):
            l = book_options[i][seller]
            sourcing[books[i]["isbn"]] = {
                "seller": seller,
                "condition": l.get("condition"),
                "price": l["price"],
                "shipping_attributed": attribution[i],
                "listing_url": l["listing_url"],
            }
        gt["optimal_sourcing"] = sourcing
        gt["optimal_total_cost"] = best_total
        gt["naive_total_cost"] = naive_total
        gt["naive_minus_optimal_pct"] = round(gap_pct, 2)
        path.write_text(json.dumps(gt, indent=2) + "\n")
        print(f"\nWrote optimal_sourcing + totals back into {path}")


if __name__ == "__main__":
    main()
