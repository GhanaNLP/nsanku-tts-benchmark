"""CLI for targeted TTS benchmark runs."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="nsanku-TTS: benchmark a specific model on specific languages")
    parser.add_argument("--langs", nargs="+", required=True, help="Subsets to evaluate")
    parser.add_argument("--model", required=True, help="Model ID to evaluate")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--force", action="store_true", help="Re-run even if already scored")
    parser.add_argument("--samples", type=int, help="Override number of text samples")
    args = parser.parse_args()

    if args.samples:
        import os
        os.environ["NSANKU_TTS_NUM_SAMPLES"] = str(args.samples)

    from benchmark.evaluate import evaluate_language

    for subset in args.langs:
        evaluate_language(
            subset,
            model_filter=args.model,
            device=args.device,
            force=args.force,
        )


if __name__ == "__main__":
    main()
