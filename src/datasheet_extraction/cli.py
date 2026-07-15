"""Command line entry point: extract, evaluate, ablate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import prompts
from .corpus import load_corpus
from .cost import Usage, estimate_cost
from .evaluate import evaluate
from .extract import DEFAULT_MODEL, Extraction, extract_corpus
from .schema import SensorSpec

DEFAULT_CORPUS = Path("corpus/demo")


def _client():
    """Build a client, resolving credentials the way the SDK does.

    Deliberately zero-arg: the SDK reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    or an `ant auth login` profile, so an unset API key does not mean unauthenticated.
    """
    import anthropic

    return anthropic.Anthropic()


def _render_table(rows: list[dict], columns: list[str]) -> str:
    """Render rows as a fixed-width table."""
    widths = {
        col: max(len(col), *(len(f"{row[col]}") for row in rows)) for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    rule = "  ".join("-" * widths[col] for col in columns)
    body = [
        "  ".join(f"{row[col]}".ljust(widths[col]) for col in columns) for row in rows
    ]
    return "\n".join([header, rule, *body])


def _run_extraction(args, documents) -> list[Extraction]:
    client = _client()
    total = len(documents)

    def progress(result: Extraction) -> None:
        marker = "ok " if result.ok else "FAIL"
        note = "" if result.ok else f"  {result.error}"
        print(
            f"  [{marker}] {result.doc_id:<16} {result.latency_s:5.1f}s"
            f"  ${result.cost_usd:.4f}{note}",
            file=sys.stderr,
        )

    print(f"extracting {total} documents with {args.model} / {args.variant}", file=sys.stderr)
    return extract_corpus(
        client,
        documents,
        model=args.model,
        variant=args.variant,
        cache_system=args.cache_system,
        on_result=progress,
    )


def _summarise(results: list[Extraction], model: str) -> dict:
    usage = sum((r.usage for r in results), Usage())
    return {
        "documents": len(results),
        "failed": sum(1 for r in results if not r.ok),
        "prompt_tokens": usage.total_prompt_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "cost_usd": round(estimate_cost(model, usage), 4),
        "mean_latency_s": (
            round(sum(r.latency_s for r in results) / len(results), 2) if results else 0.0
        ),
    }


def cmd_extract(args) -> int:
    documents, _ = load_corpus(args.corpus)
    results = _run_extraction(args, documents)

    payload = {
        r.doc_id: (r.spec.model_dump() if r.spec else None) for r in results
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", file=sys.stderr)
    print(json.dumps(_summarise(results, args.model), indent=2))
    return 0


def cmd_evaluate(args) -> int:
    documents, gold = load_corpus(args.corpus)

    if args.predictions:
        raw = json.loads(args.predictions.read_text(encoding="utf-8"))
        predicted = {
            doc_id: SensorSpec.model_validate(fields)
            for doc_id, fields in raw.items()
            if fields is not None
        }
        summary = None
    else:
        results = _run_extraction(args, documents)
        predicted = {r.doc_id: r.spec for r in results if r.spec is not None}
        summary = _summarise(results, args.model)

    report = evaluate(gold, predicted)
    print()
    print(
        _render_table(
            report.as_rows(),
            ["field", "support", "precision", "recall", "f1", "hallucination_rate"],
        )
    )
    if summary:
        print()
        print(json.dumps(summary, indent=2))
    return 0


def cmd_ablate(args) -> int:
    documents, gold = load_corpus(args.corpus)
    rows = []

    for model in args.models:
        for variant in args.variants:
            run_args = argparse.Namespace(
                model=model, variant=variant, cache_system=args.cache_system
            )
            results = _run_extraction(run_args, documents)
            predicted = {r.doc_id: r.spec for r in results if r.spec is not None}
            report = evaluate(gold, predicted)
            total = report.total
            summary = _summarise(results, model)

            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "f1": round(total.f1, 4),
                    "precision": round(total.precision, 4),
                    "recall": round(total.recall, 4),
                    "halluc_rate": round(total.hallucination_rate, 4),
                    "cost_usd": summary["cost_usd"],
                    "latency_s": summary["mean_latency_s"],
                    "failed": summary["failed"],
                }
            )

    print()
    print(
        _render_table(
            rows,
            [
                "model",
                "variant",
                "f1",
                "precision",
                "recall",
                "halluc_rate",
                "cost_usd",
                "latency_s",
                "failed",
            ],
        )
    )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasheet-extraction",
        description="Extract sensor specs from datasheets and measure how well it worked.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument(
            "--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus directory"
        )
        p.add_argument("--model", default=DEFAULT_MODEL, help="model id")
        p.add_argument(
            "--variant",
            default=prompts.DEFAULT_VARIANT,
            choices=sorted(prompts.VARIANTS),
            help="prompt variant",
        )
        p.add_argument(
            "--cache-system",
            action="store_true",
            help="request prompt caching on the system prefix (a no-op below the "
            "model's minimum cacheable size — check cache_read_tokens)",
        )

    p_extract = sub.add_parser("extract", help="extract and write predictions to JSON")
    add_common(p_extract)
    p_extract.add_argument(
        "--output", type=Path, default=Path("predictions.json"), help="where to write"
    )
    p_extract.set_defaults(func=cmd_extract)

    p_eval = sub.add_parser("evaluate", help="score predictions against gold labels")
    add_common(p_eval)
    p_eval.add_argument(
        "--predictions",
        type=Path,
        help="score an existing predictions file instead of calling the API",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_ablate = sub.add_parser("ablate", help="sweep models x prompt variants")
    p_ablate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p_ablate.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    p_ablate.add_argument(
        "--variants", nargs="+", default=sorted(prompts.VARIANTS), choices=sorted(prompts.VARIANTS)
    )
    p_ablate.add_argument("--cache-system", action="store_true")
    p_ablate.add_argument("--output", type=Path, help="write the results table as JSON")
    p_ablate.set_defaults(func=cmd_ablate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
