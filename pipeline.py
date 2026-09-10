"""Full benchmark pipeline: iterate all languages, evaluate each."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="nsanku-TTS benchmark pipeline")
    parser.add_argument("--langs", nargs="*", help="Subsets to evaluate (default: all)")
    parser.add_argument("--model", help="Filter to models matching this substring")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--force", action="store_true", help="Re-run completed models")
    parser.add_argument("--dry-run", action="store_true", help="List work without GPU")
    args = parser.parse_args()

    from benchmark.dataset import available_subsets
    from benchmark.evaluate import evaluate_language

    subsets = args.langs or available_subsets()
    print(f"nsanku-TTS benchmark — {len(subsets)} language(s)")
    print(f"Device: {args.device}")

    if args.dry_run:
        for s in subsets:
            print(f"  [dry-run] would evaluate: {s}")
        return

    for subset in subsets:
        try:
            evaluate_language(
                subset,
                model_filter=args.model,
                device=args.device,
                force=args.force,
            )
        except Exception as e:
            print(f"  ERROR on {subset}: {e}")
            continue


if __name__ == "__main__":
    main()
