import argparse
import os
from review_delight_extractor import ReviewDelightExtractor


def main():
    """
    Main function to parse command-line arguments and run the Review Delight Extractor.
    """
    parser = argparse.ArgumentParser(
        description="Review Delight Point Extractor CLI Tool"
    )
    parser.add_argument(
        "--reviews_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing customer reviews.",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="output_reviews.json",
        help="Path to the output JSON file for processed reviews.",
    )
    parser.add_argument(
        "--output_csv_file",
        type=str,
        default="ranked_attributes.csv",
        help="Path to the output CSV file for ranked attributes.",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        help="Path to the evaluation CSV file. If provided, the tool will run an evaluation.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="Your OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.",
    )

    args = parser.parse_args()

    # --- Path Validation and Directory Creation ---
    # Convert relative paths to absolute paths for robust file handling
    # The default paths are relative to where main.py is run from.

    # Base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve paths relative to the base directory
    reviews_file_path = os.path.join(base_dir, "data", args.reviews_file)
    output_json_file_path = os.path.join(base_dir, "output", args.output_json_file)
    output_csv_file_path = os.path.join(base_dir, "output", args.output_csv_file)

    # If an evaluation file is provided, resolve its path
    evaluate_file_path = None
    if args.evaluate_file:
        evaluate_file_path = os.path.join(base_dir, args.evaluate_file)

    # Validate that the reviews file exists.
    if not os.path.isfile(reviews_file_path):
        print(f"Error: Reviews file not found at {reviews_file_path}")
        return

    # Ensure the output directories exist.
    # If the output directories do not exist, create them.
    output_dir = os.path.dirname(output_csv_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_json_dir = os.path.dirname(output_json_file_path)
    if output_json_dir and not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir, exist_ok=True)

    # Initialize the extractor with the provided API key.
    extractor = ReviewDelightExtractor(openai_api_key=args.openai_api_key)

    # Process the reviews.
    print("\n--- Starting Review Processing ---")
    extractor.process_reviews(
        reviews_file_path, output_json_file_path, output_csv_file_path
    )
    print("--- Review Processing Completed ---")

    # Run evaluation if an evaluation file is provided.
    if args.evaluate_file:
        print("\n--- Starting Evaluation ---")
        # Validate evaluation file existence before passing to extractor
        if not os.path.isfile(evaluate_file_path):
            print(f"Error: Evaluation file not found at {evaluate_file_path}. Skipping evaluation.")
        else:
            extractor.evaluate(args.evaluate_file, output_json_file_path)
        print("--- Evaluation Completed ---")


if __name__ == "__main__":
    main()