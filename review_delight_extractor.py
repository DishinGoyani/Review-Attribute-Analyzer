"""
Review Delight Point Extractor CLI Tool

This tool analyzes customer reviews from a JSON file to extract key positive product attributes.
It then groups semantically similar attributes using TF-IDF and K-Means clustering, and ranks them by frequency.
An evaluation component is included to assess the tool's accuracy against a ground truth dataset.

Usage:
    python review_delight_extractor.py --reviews_file <path_to_reviews.json> \
        [--output_json_file <output_reviews.json>] \
        [--output_csv_file <ranked_attributes.csv>] \
        [--evaluate_file <path_to_evaluation.csv>] \
        [--openai_api_key <your_openai_api_key>]
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI, OpenAIError
import concurrent.futures
from tqdm import tqdm


class ReviewDelightExtractor:
    """
    A class to extract, cluster, and rank delight attributes from customer reviews.
    It leverages OpenAI's GPT-3.5-turbo for attribute extraction and scikit-learn
    for clustering and frequency analysis.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initializes the ReviewDelightExtractor with an OpenAI API key.

        Args:
            openai_api_key (str, optional): Your OpenAI API key. If not provided,
                                            it will attempt to read from the OPENAI_API_KEY
                                            environment variable. Defaults to None.
        """
        # Initialize the OpenAI client. The API key can be passed directly or
        # will be picked up from the OPENAI_API_KEY environment variable.
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided as an argument or set in the 'OPENAI_API_KEY' environment variable."
            )
        self.client = OpenAI(api_key=api_key)

    def extract_attributes(self, review_body: str) -> list[str]:
        """
        Extracts positive product attributes from a given review body using the OpenAI API.

        This method sends the review text to the GPT-3.5-turbo model with a specific prompt
        to identify and list positive attributes.

        Args:
            review_body (str): The text content of the customer review.

        Returns:
            list[str]: A list of extracted positive attributes. Returns an empty list
                       if no attributes are found or an error occurs during API call.
        """
        try:
            # Define the system and user messages for the OpenAI API call.
            # The system message sets the persona and expected output format.
            # The user message provides the review content for extraction.
            messages = [
                {
                    "role": "system",
                    "content": """
                                You are a helpful assistant that extracts positive product attributes from customer reviews. 
                                Extract positive product attributes from reviews. Use specific standardized terms:
                                - Fragrance (smell/scent), Quality (build/material), Longevity (long-lasting), Effectiveness (works well)
                                - Packaging, Moisturizing, Non-allergenic, Climate Suitability, Compatibility, Texture
                                - Overall Satisfaction, Customer Service, Overall Quality

                                Return comma-separated list. Examples:
                                "smell is amazing" → Fragrance
                                "good quality, nice packaging, lasts all day" → Quality, Packaging, Longevity
                                "satisfied with product" → Overall Satisfaction
                                If no positive attributes are found, respond with 'None'.
                                """,
                },
                {
                    "role": "user",
                    "content": f"Extract positive product attributes from this review: '{review_body}'",
                },
            ]

            # Make the API call to OpenAI's chat completions endpoint.
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages
            )

            # Extract the content from the API response.
            attributes_raw = response.choices[0].message.content.strip()

            # Process the raw attributes string.
            if attributes_raw.lower() == "none":
                return []  # Return empty list if the model explicitly states 'None'
            else:
                # Split the string by comma and strip whitespace from each attribute.
                return [
                    attr.strip() for attr in attributes_raw.split(",") if attr.strip()
                ]

        except OpenAIError as e:
            # Catch specific OpenAI API errors for better error handling.
            print(f"OpenAI API Error during attribute extraction: {e}")
            return []
        except Exception as e:
            # Catch any other unexpected errors.
            print(f"An unexpected error occurred during attribute extraction: {e}")
            return []

    def determine_optimal_clusters(self, X) -> int:
        """
        Determines the optimal number of clusters using the Elbow Method.
        This method calculates the Within-Cluster Sum of Squares (WCSS) for a range of cluster counts
        and identifies the point where the rate of decrease in WCSS slows down significantly (the "elbow").
        This is a heuristic method to find a suitable number of clusters for K-Means clustering.

        Args:
            X: The vectorized attributes (TF-IDF matrix).

        Returns:
            int: The optimal number of clusters.
        """
        n_samples = X.shape[0]

        # Handle edge cases
        if n_samples <= 1:
            return 1
        if n_samples <= 3:
            return 2

        # Calculate reasonable range
        max_clusters = min(int(np.sqrt(n_samples)) + 3, 15, n_samples - 1)

        # Calculate WCSS for each k
        wcss = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Find elbow using simple percentage drop method
        for i in range(1, len(wcss) - 1):
            # Calculate percentage improvement from previous k
            improvement = (wcss[i - 1] - wcss[i]) / wcss[i - 1] * 100

            # If improvement drops below 15%, we found our elbow
            if improvement < 15:
                return max(i, 2)  # Return k value (i+1-1 because range starts at 1)

        # Fallback: return middle value
        return max(2, len(wcss) // 2 + 1)

    def cluster_attributes(self, attributes: list[str]) -> list[tuple[str, int]]:
        """
        Clusters a list of attributes and returns a list of representative attributes.

        This method uses TF-IDF for text vectorization and K-Means for clustering
        to group semantically similar attributes. The most frequent attribute from
        each cluster is chosen as its representative.

        Args:
            attributes (list[str]): A list of attributes to be clustered.

        Returns:
            list[tuple[str, int]]: A list of representative attributes after clustering and deduplication.
        """
        if not attributes:
            return []

        # Initialize TF-IDF Vectorizer to convert text attributes into numerical vectors.
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(attributes)

        # Determine the optimal number of clusters using the Elbow Method.
        num_clusters = self.determine_optimal_clusters(X)

        # Handle case where num_clusters might be 0 or 1 if data is too sparse or identical.
        if num_clusters <= 1:
            # If only one cluster, return unique attributes as representatives.
            return list(set(attributes))

        # Perform K-Means clustering.
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(X)

        # Group attributes by their assigned cluster labels.
        clustered_attribute_groups = {i: [] for i in range(num_clusters)}
        for i, label in enumerate(kmeans.labels_):
            clustered_attribute_groups[label].append(attributes[i])

        # Select a representative attribute for each cluster.
        # The most frequent attribute within each cluster is chosen as the representative.
        representative_attributes = []
        for cluster_id in sorted(clustered_attribute_groups.keys()):
            attrs_in_cluster = clustered_attribute_groups[cluster_id]
            if attrs_in_cluster:
                # Use pandas Series.value_counts() to find the most frequent item.
                most_frequent_attr = pd.Series(attrs_in_cluster).value_counts().index[0]
                cluster_len = len(attrs_in_cluster)
                representative_attributes.append((most_frequent_attr, cluster_len))

        return representative_attributes

    def process_reviews(
        self,
        reviews_file_path: str,
        output_json_file_path: str,
        output_csv_file_path: str,
    ):
        """
        Processes a JSON file of customer reviews, extracts attributes, clusters them,
        and saves the results to JSON and CSV files.

        Args:
            reviews_file_path (str): Path to the input JSON file containing customer reviews.
            output_json_file_path (str): Path to save the processed reviews (with extracted attributes).
            output_csv_file_path (str): Path to save the ranked attributes and their frequencies.
        """
        try:
            with open(reviews_file_path, "r", encoding="utf-8") as f:
                reviews_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Reviews file not found at {reviews_file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {reviews_file_path}")
            return

        processed_reviews = []
        all_extracted_attributes = []

        # Use ThreadPoolExecutor for parallel processing of attribute extraction.
        # The number of workers can be adjusted based on system resources and API rate limits.
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit tasks for each review's body to the executor.
            # Store futures along with their original review objects.
            future_to_review = {
                executor.submit(self.extract_attributes, review.get("body")): review
                for review in reviews_data
                if review.get("body")
            }

            # As tasks complete, collect results and update review objects.
            for future in tqdm(
                concurrent.futures.as_completed(future_to_review),
                total=len(future_to_review),
                desc="Extracting attributes",
            ):
                original_review = future_to_review[future]
                try:
                    extracted_attributes = future.result()
                    original_review["delight_attributes"] = extracted_attributes
                    processed_reviews.append(original_review)
                    all_extracted_attributes.extend(extracted_attributes)
                except Exception as exc:
                    print(
                        f"Review {original_review.get('review_id', 'N/A')} generated an exception: {exc}"
                    )

        # Handle reviews that might not have a 'body' field.
        for review in reviews_data:
            if not review.get("body"):
                print(
                    f"Warning: Review with ID {review.get('review_id', 'N/A')} has no 'body' field. Skipping."
                )

        # Save the processed reviews to a JSON file.
        try:
            with open(output_json_file_path, "w", encoding="utf-8") as f:
                json.dump({"reviews": processed_reviews}, f, indent=4)
            print(f"Processed reviews saved to {output_json_file_path}")
        except IOError as e:
            print(f"Error saving processed JSON file to {output_json_file_path}: {e}")

        # Cluster and rank the extracted attributes.
        clustered_and_ranked_attributes = self.cluster_attributes(
            all_extracted_attributes
        )

        # This creates a DataFrame with 'Delight Attribute' and 'Frequency' columns.
        attribute_frequencies = pd.DataFrame(
            clustered_and_ranked_attributes, columns=["Delight Attribute", "Frequency"]
        )
        # Sort the DataFrame by frequency in descending order.
        attribute_frequencies = attribute_frequencies.sort_values(
            by="Frequency", ascending=False
        )

        # Save the ranked attributes to a CSV file.
        try:
            attribute_frequencies.to_csv(
                output_csv_file_path, index=False, encoding="utf-8"
            )
            print(f"Ranked attributes saved to {output_csv_file_path}")
        except IOError as e:
            print(f"Error saving ranked CSV file to {output_csv_file_path}: {e}")

    def evaluate(self, evaluation_file_path: str, processed_json_file_path: str):
        """
        Evaluates the accuracy of attribute extraction against a ground truth dataset.

        Args:
            evaluation_file_path (str): Path to the CSV file containing ground truth for evaluation.
            processed_json_file_path (str): Path to the JSON file containing reviews with extracted attributes.
        """
        try:
            eval_df = pd.read_csv(evaluation_file_path)
        except FileNotFoundError:
            print(f"Error: Evaluation file not found at {evaluation_file_path}")
            return

        try:
            with open(processed_json_file_path, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Processed JSON file not found at {processed_json_file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {processed_json_file_path}")
            return

        correct_extractions = 0
        incorrect_extractions = 0

        # Create a dictionary for efficient lookup of extracted attributes by review_id.
        processed_review_map = {
            review.get("review_id"): review.get("delight_attributes", [])
            for review in processed_data.get("reviews", [])
        }

        # Iterate through the evaluation dataset to compare expected vs. extracted attributes.
        for _, row in eval_df.iterrows():
            review_id = row.get("review_id")
            # Ensure expected_attributes is a list, handling potential NaN from CSV.
            expected_attributes_raw = str(row.get("delight_attribute", "")).strip()
            expected_attributes = [
                attr.strip()
                for attr in expected_attributes_raw.split(",")
                if attr.strip()
            ]

            extracted_attributes = processed_review_map.get(review_id, [])

            # Simple semantic equivalence check: if any expected attribute is a substring
            # of an extracted attribute, or vice-versa, it's considered a match.
            # NOTE: This is a simplified check. A more robust solution would involve
            # advanced NLP techniques for semantic similarity (e.g., word embeddings).
            match_found = False
            for expected in expected_attributes:
                for extracted in extracted_attributes:
                    if (
                        expected.lower() in extracted.lower()
                        or extracted.lower() in expected.lower()
                    ):
                        match_found = True
                        break
                if match_found:
                    break

            if match_found:
                correct_extractions += 1
            else:
                incorrect_extractions += 1

        total_extractions = correct_extractions + incorrect_extractions
        # Calculate accuracy, handling division by zero if no extractions were made.
        accuracy = (
            (correct_extractions / total_extractions) * 100
            if total_extractions > 0
            else 0.0
        )

        # Print evaluation results to the console.
        print(f"\n--- Evaluation Results ---")
        print(f"Number of correct attribute extractions: {correct_extractions}")
        print(f"Number of incorrect attribute extractions: {incorrect_extractions}")
        print(f"Overall accuracy percentage: {accuracy:.2f}%")
