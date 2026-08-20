import argparse
import json

from steering_recovery.comparison import write_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a table and barplot from denoiser model folders."
    )
    parser.add_argument(
        "root", help="Run directory containing model summary.json files."
    )
    parser.add_argument(
        "--output-dir",
        default="comparison",
        help="Destination for CSV, Markdown and PNG outputs.",
    )
    arguments = parser.parse_args()
    result = write_comparison(arguments.root, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
