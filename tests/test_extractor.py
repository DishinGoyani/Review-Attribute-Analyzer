"""
Unit tests for the Review Delight Point Extractor CLI tool.

This module contains a series of unit tests to verify the functionality of the
`ReviewDelightExtractor` class, including attribute extraction, processing,
and evaluation. Mocking is used for the OpenAI API calls to ensure tests are
fast, reliable, and do not incur actual API costs.
"""

import shutil
import unittest
import json
import os
import pandas as pd
from unittest.mock import MagicMock, patch

# Import the main class from the application.
from review_delight_extractor import ReviewDelightExtractor


class TestReviewDelightExtractor(unittest.TestCase):
    """
    Test suite for the ReviewDelightExtractor class.
    """

    def setUp(self):
        """
        Set up test environment before each test method.
        This includes creating dummy input files (reviews.json, evaluation.csv)
        and defining paths for output files.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.test_reviews_file = os.path.join(self.base_dir, ".temp\\test_reviews.json")
        self.test_output_json_file = os.path.join(self.base_dir, ".temp\\output\\test_output_reviews.json")
        self.test_output_csv_file = os.path.join(self.base_dir, ".temp\\output\\test_ranked_attributes.csv")
        self.test_evaluation_file = os.path.join(self.base_dir, ".temp\\test_evaluation.csv")
        # Use a dummy key for testing purposes; actual API calls are mocked.
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test-key")

        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(self.test_output_json_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_output_csv_file), exist_ok=True)

        # Create a dummy reviews.json file for testing the processing logic.
        dummy_reviews = [
            {
                "review_id": "1",
                "author": "Test User 1",
                "body": "This product has an amazing fragrance and feels very fresh.",
            },
            {
                "review_id": "2",
                "author": "Test User 2",
                "body": "The scent is delightful, but the packaging could be better.",
            },
            {
                "review_id": "3",
                "author": "Test User 3",
                "body": "I love the fresh feeling it gives. The smell is also great.",
            },
        ]

        with open(self.test_reviews_file, "w", encoding="utf-8") as f:
            json.dump(dummy_reviews, f, indent=4)

        # Create a dummy delight-evaluation.csv file for testing the evaluation logic.
        dummy_evaluation = {
            "review_id": ["1", "2", "3"],
            "expected_attributes": [
                "Fragrance, Freshness",
                "Scent",
                "Freshness, Smell",
            ],
        }
        pd.DataFrame(dummy_evaluation).to_csv(self.test_evaluation_file, index=False)

    def tearDown(self):
        """
        Clean up test environment after each test method.
        This removes all temporary files created during the test run.
        """
        # Remove the temporary files created during the tests.
        # Remove .temp directory if it exists.
        temp_dir = os.path.join(self.base_dir, ".temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            

    @patch("review_delight_extractor.OpenAI")
    def test_process_reviews(self, mock_openai):
        """
        Tests the `process_reviews` method to ensure it correctly extracts attributes,
        generates JSON and CSV output files, and that the output content is valid.
        OpenAI API calls are mocked to control responses and avoid external dependencies.
        """
        # Configure the mock OpenAI client to return predefined responses for chat completions.
        # Each call to create() will return a different MagicMock object from this list.
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Fragrance, Freshness"))]
            ),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Scent"))]),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Freshness, Smell"))]
            ),
        ]

        # Initialize the extractor and run the process_reviews method.
        extractor = ReviewDelightExtractor(openai_api_key=self.openai_api_key)
        extractor.process_reviews(
            self.test_reviews_file,
            self.test_output_json_file,
            self.test_output_csv_file,
        )

        # Assert that the output files were created.
        self.assertTrue(os.path.exists(self.test_output_json_file))
        self.assertTrue(os.path.exists(self.test_output_csv_file))

        # Verify the content of the generated JSON output file.
        with open(self.test_output_json_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)
            self.assertIn("reviews", output_data)
            self.assertEqual(len(output_data["reviews"]), 3)
            for review in output_data["reviews"]:
                self.assertIn("delight_attributes", review)
                self.assertIsInstance(review["delight_attributes"], list)

        # Verify the content of the generated CSV output file.
        output_df = pd.read_csv(self.test_output_csv_file)
        self.assertIn("Delight Attribute", output_df.columns)
        self.assertIn("Frequency", output_df.columns)
        self.assertGreater(len(output_df), 0)

    @patch("review_delight_extractor.OpenAI")
    def test_evaluate(self, mock_openai):
        """
        Tests the `evaluate` method to ensure it correctly calculates and reports
        performance metrics based on extracted and expected attributes.
        OpenAI API calls are mocked for consistent test results.
        """
        # Configure the mock OpenAI client for this test as well.
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Fragrance, Freshness"))]
            ),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Scent"))]),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Freshness, Smell"))]
            ),
        ]

        # First, process reviews to create the necessary processed JSON file
        # that the evaluation method will read.
        extractor = ReviewDelightExtractor(openai_api_key=self.openai_api_key)
        extractor.process_reviews(
            self.test_reviews_file,
            self.test_output_json_file,
            self.test_output_csv_file,
        )

        # Capture stdout to verify the printed evaluation results.
        import sys
        from io import StringIO

        captured_output = StringIO()
        sys.stdout = captured_output  # Redirect stdout to the StringIO object.

        # Run the evaluation method.
        extractor.evaluate(self.test_evaluation_file, self.test_output_json_file)

        sys.stdout = sys.__stdout__  # Reset stdout to its original destination.
        output = captured_output.getvalue()

        # Assert that the expected evaluation summary is present in the output.
        self.assertIn("--- Evaluation Results ---", output)
        self.assertIn("Number of correct attribute extractions:", output)
        self.assertIn("Number of incorrect attribute extractions:", output)
        self.assertIn("Overall accuracy percentage:", output)


if __name__ == "__main__":
    unittest.main()
