# Review Delight Point Extractor

## Introduction

This CLI tool analyzes customer reviews from a JSON file to extract key positive product attributes. It then groups semantically similar attributes and ranks them by frequency, providing valuable insights for marketing and product development. The tool also includes an evaluation mechanism to assess its accuracy against a ground truth dataset.

## Features

- **Attribute Extraction**: Identifies positive product attributes from review text using OpenAI's GPT-3.5-turbo model.
- **Attribute Clustering**: Groups semantically similar attributes using TF-IDF vectorization and K-Means clustering.
- **Frequency Ranking**: Ranks extracted attributes based on their frequency of mention.
- **Input/Output Handling**: Processes JSON input files and generates JSON and CSV output files.
- **Evaluation**: Assesses the tool's performance against a provided evaluation dataset, reporting accuracy metrics.
