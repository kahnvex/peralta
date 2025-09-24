#!/usr/bin/env python3
"""
Expand a math NL2Code dataset by ~10x with small "hardening" edits.

- Input CSV columns: python_expression,natural_language
- Output CSV columns: python_expression,natural_language
- For each input row, we write the original row (unless --no-keep-originals)
  and then produce --expansions-per-row new rows by adding exactly one extra
  operation on TOP of the original expression, with parentheses to preserve
  order of operations and a matching NL instruction.

Usage:
  python expand_dataset.py --in data.csv --out expanded.csv \
      --expansions-per-row 9 --seed 42
"""

import argparse
import csv
import random
from typing import Callable, Tuple

# ---- helpers ----------------------------------------------------------------


def _append_instruction(nl: str, clause: str) -> str:
    """Append 'then <clause>.' to an existing natural-language instruction."""
    base = nl.strip()
    # Trim a trailing period to avoid `..`
    if base.endswith("."):
        base = base[:-1]
    # Normalize spacing/punctuation
    return f"{base}, then {clause}."


def _two_ints(rng: random.Random, lo: int, hi: int) -> Tuple[int, int]:
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _three_ints(rng: random.Random, lo: int, hi: int) -> Tuple[int, int, int]:
    return rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)


# ---- expander functions -----------------------------------------------------
# Each expander receives (expr, nl, rng) and returns (new_expr, new_nl)


def add_constant(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    c = rng.randint(2, 50)
    new_expr = f"({expr}) + {c}"
    new_nl = _append_instruction(nl, f"add {c}")
    return new_expr, new_nl


def subtract_constant(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    c = rng.randint(2, 50)
    new_expr = f"({expr}) - {c}"
    new_nl = _append_instruction(nl, f"subtract {c}")
    return new_expr, new_nl


def multiply_constant(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    k = rng.randint(2, 12)
    new_expr = f"({expr}) * {k}"
    new_nl = _append_instruction(nl, f"multiply that by {k}")
    return new_expr, new_nl


def add_sum(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 50)
    new_expr = f"({expr}) + ({a} + {b})"
    new_nl = _append_instruction(nl, f"add the sum of {a} and {b}")
    return new_expr, new_nl


def subtract_sum(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 50)
    new_expr = f"({expr}) - ({a} + {b})"
    new_nl = _append_instruction(nl, f"subtract the sum of {a} and {b}")
    return new_expr, new_nl


def add_product(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 12)
    new_expr = f"({expr}) + ({a} * {b})"
    new_nl = _append_instruction(nl, f"add the product of {a} and {b}")
    return new_expr, new_nl


def subtract_product(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 12)
    new_expr = f"({expr}) - ({a} * {b})"
    new_nl = _append_instruction(nl, f"subtract the product of {a} and {b}")
    return new_expr, new_nl


def multiply_by_sum(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 20)
    new_expr = f"({expr}) * ({a} + {b})"
    new_nl = _append_instruction(nl, f"multiply that by the sum of {a} and {b}")
    return new_expr, new_nl


def multiply_by_product(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 10)
    new_expr = f"({expr}) * ({a} * {b})"
    new_nl = _append_instruction(nl, f"multiply that by the product of {a} and {b}")
    return new_expr, new_nl


def add_difference(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 50)
    new_expr = f"({expr}) + ({a} - {b})"
    new_nl = _append_instruction(nl, f"add the difference of {a} and {b})")
    return new_expr, new_nl


def subtract_difference(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b = _two_ints(rng, 2, 50)
    new_expr = f"({expr}) - ({a} - {b})"
    new_nl = _append_instruction(nl, f"subtract the difference of {a} and {b}")
    return new_expr, new_nl


def add_triple_sum(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b, c = _three_ints(rng, 2, 30)
    new_expr = f"({expr}) + ({a} + {b} + {c})"
    new_nl = _append_instruction(nl, f"add the sum of {a}, {b}, and {c}")
    return new_expr, new_nl


def subtract_triple_sum(expr: str, nl: str, rng: random.Random) -> Tuple[str, str]:
    a, b, c = _three_ints(rng, 2, 30)
    new_expr = f"({expr}) - ({a} + {b} + {c})"
    new_nl = _append_instruction(nl, f"subtract the sum of {a}, {b}, and {c}")
    return new_expr, new_nl


EXPANDERS: Tuple[Callable[[str, str, random.Random], Tuple[str, str]], ...] = (
    add_constant,
    subtract_constant,
    multiply_constant,
    add_sum,
    subtract_sum,
    add_product,
    subtract_product,
    multiply_by_sum,
    multiply_by_product,
    add_difference,
    subtract_difference,
    add_triple_sum,
    subtract_triple_sum,
)

# ---- main -------------------------------------------------------------------


def expand_dataset(
    in_path: str,
    out_path: str,
    expansions_per_row: int = 9,
    keep_originals: bool = True,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)

    with (
        open(in_path, newline="", encoding="utf-8") as fin,
        open(out_path, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        fieldnames = ["python_expression", "natural_language"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            expr = row["python_expression"].strip()
            nl = row["natural_language"].strip()

            if keep_originals:
                writer.writerow({"python_expression": expr, "natural_language": nl})

            for _ in range(expansions_per_row):
                expander = rng.choice(EXPANDERS)
                new_expr, new_nl = expander(expr, nl, rng)
                writer.writerow(
                    {"python_expression": new_expr, "natural_language": new_nl}
                )


def main():
    ap = argparse.ArgumentParser(
        description="10x a NL↔python_expression dataset by adding one extra operation per example."
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input CSV with columns: python_expression,natural_language",
    )
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument(
        "--expansions-per-row",
        type=int,
        default=9,
        help="How many new rows per original (default: 9, gives ~10x with originals kept)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument(
        "--no-keep-originals",
        action="store_true",
        help="If set, only write expansions (not the original rows)",
    )
    args = ap.parse_args()

    expand_dataset(
        in_path=args.in_path,
        out_path=args.out_path,
        expansions_per_row=args.expansions_per_row,
        keep_originals=not args.no_keep_originals,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
