"""
Filter evaluation results to extract only correctly predicted examples.

This creates datasets of correct predictions for interpretability analysis,
following the paper's approach of analyzing 400 correct L3 samples.

Usage:
    # Filter full fine-tuned model results
    uv run python scripts/filter_correct_predictions.py \
        --eval-results results/full_2025-11-17-12-42-41-3911.json \
        --output data/blocksworld_L3_correct_full.jsonl \
        --max-samples 400

    # Filter LoRA model results
    uv run python scripts/filter_correct_predictions.py \
        --eval-results results/lora_2025-11-17-09-14-09-9bf6.json \
        --output data/blocksworld_L3_correct_lora.jsonl \
        --max-samples 400
"""

import json
import argparse


def filter_correct_predictions(
    eval_results_path: str,
    original_data_path: str,
    output_path: str,
    max_samples: int = 400,
):
    """
    Filter evaluation results to extract only correct predictions.

    Args:
        eval_results_path: Path to evaluation results JSON (e.g., full_2025-11-17-12-42-41-3911.json)
        original_data_path: Path to original test data (e.g., blocksworld_L3_test.jsonl)
        output_path: Where to save filtered correct examples
        max_samples: Maximum number of correct samples to keep (default: 400, matching paper)
    """
    print(f"Loading evaluation results from {eval_results_path}...")
    with open(eval_results_path, "r") as f:
        eval_data = json.load(f)

    print(f"Loading original test data from {original_data_path}...")
    with open(original_data_path, "r") as f:
        original_examples = [json.loads(line) for line in f]

    model_type = eval_data["model_type"]
    total_examples = eval_data["metrics"]["total_examples"]
    exact_match_rate = eval_data["metrics"]["exact_match"]

    print(f"\n{'='*60}")
    print(f"Model: {model_type}")
    print(f"Total examples evaluated: {total_examples}")
    print(f"Exact match rate: {exact_match_rate:.2%}")
    print(f"{'='*60}\n")

    # Filter to correct predictions
    correct_examples = []
    for result in eval_data["results"]:
        expected = result["expected"].strip()
        predicted = result["predicted"].strip()

        if expected == predicted:
            example_id = result["example_id"]
            # Get the full original example
            original_example = original_examples[example_id]
            correct_examples.append(original_example)

            if len(correct_examples) >= max_samples:
                break

    num_correct = len(correct_examples)
    print(f"Found {num_correct} correct predictions")

    if num_correct < max_samples:
        print(
            f"⚠️  Warning: Only {num_correct} correct examples available, requested {max_samples}"
        )
    else:
        print(f"✓ Using first {max_samples} correct examples (as specified)")

    # Save to output file
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        for example in correct_examples:
            f.write(json.dumps(example) + "\n")

    print(f"✓ Saved {len(correct_examples)} correct examples to {output_path}")

    # Print some statistics
    print(f"\n{'='*60}")
    print("Sample of first correct example:")
    print(f"{'='*60}")
    if correct_examples:
        first = correct_examples[0]
        print(f"Input (first 200 chars):\n{first['input'][:200]}...")
        print(f"\nOutput:\n{first['output']}")
    print(f"{'='*60}\n")

    return len(correct_examples)


def main():
    parser = argparse.ArgumentParser(
        description="Filter evaluation results to extract correct predictions"
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--original-data",
        type=str,
        default="data/blocksworld_L3_test.jsonl",
        help="Path to original test data (default: data/blocksworld_L3_test.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered correct examples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=400,
        help="Maximum number of correct samples to keep (default: 400)",
    )

    args = parser.parse_args()

    filter_correct_predictions(
        args.eval_results,
        args.original_data,
        args.output,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
