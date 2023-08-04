import sys

sys.path.append("..")

import os
import spacy
import pandas as pd
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nlp.data_cleaning import run_batch_text_normalization_pipeline
from nlp.sentiments import compute_sentiments_in_batch
from typing import List

load_dotenv()
PANDAS_SAMPLE_LIMIT = int(os.getenv("PANDAS_SAMPLE_LIMIT", 95_000))
AMAZON_BIGDATA_INPUT_DIRECTORY = os.getenv(
    "AMAZON_BIGDATA_INPUT_DIRECTORY", "../input-amazon"
)
AMAZON_BIGDATA_OUTPUT_DIRECTORY = os.getenv(
    "AMAZON_BIGDATA_OUTPUT_DIRECTORY", "../output-amazon"
)


COLUMNS_TO_DROP: List[str] = [
    "marketplace",
    "customer_id",
    "review_id",
    "product_id",
    "product_parent",
    "verified_purchase",
    "review_date",
    "review_headline",
    "vine",
    "total_votes",
]


def dataset_is_processed_already(filename: str) -> bool:
    output_files = os.listdir(AMAZON_BIGDATA_OUTPUT_DIRECTORY)
    return any(filename in output_file for output_file in output_files)


def process_amazon_reviews():
    if not os.path.exists(AMAZON_BIGDATA_OUTPUT_DIRECTORY):
        os.mkdir(AMAZON_BIGDATA_OUTPUT_DIRECTORY)
        print(f"Created an output directory at {AMAZON_BIGDATA_OUTPUT_DIRECTORY}")

    nlp_model = spacy.load("en_core_web_md", disable=["parser", "ner", "tok2vec"])
    sentiment_analyzer = SentimentIntensityAnalyzer()

    input_files = os.listdir(AMAZON_BIGDATA_INPUT_DIRECTORY)

    for input_file in input_files:
        relative_path = os.path.join(AMAZON_BIGDATA_INPUT_DIRECTORY, input_file)

        if dataset_is_processed_already(input_file):
            print(
                f"Skipping the dataset at {relative_path}, because it has already been processed in past runs! \n"
            )
            continue

        print(f"Starting processing on this dataset: {relative_path}")

        amazon_reviews = (
            pd.read_csv(relative_path, sep="\t", header="infer", on_bad_lines="skip")
            .dropna(subset=["review_body", "star_rating"])
            .drop(labels=COLUMNS_TO_DROP, axis=1)
        )

        rows = amazon_reviews.shape[0]
        amazon_reviews_samples = amazon_reviews.sample(
            n=PANDAS_SAMPLE_LIMIT if rows > PANDAS_SAMPLE_LIMIT else rows
        )

        amazon_reviews_samples[
            "normalized_review_body"
        ] = run_batch_text_normalization_pipeline(
            nlp_model,
            amazon_reviews_samples["review_body"],
            show_batch_progress=True,
            texts_type="list",
        )

        print(f"Running sentiment analysis calculations on {relative_path}")
        review_compound_scores, review_sentiment_labels = compute_sentiments_in_batch(
            sentiment_analyzer,
            amazon_reviews_samples["review_body"],
            show_batch_progress=True,
        )
        amazon_reviews_samples["review_body_compound_score"] = review_compound_scores
        amazon_reviews_samples["review_body_sentiment_label"] = review_sentiment_labels

        output_file_path = os.path.join(AMAZON_BIGDATA_OUTPUT_DIRECTORY, input_file)
        print(f"Saving dataframe as a CSV on {output_file_path}")
        amazon_reviews_samples.to_csv(output_file_path, index=False)

        del [amazon_reviews_samples, amazon_reviews]

        print(f"Finished processing on this dataset: {relative_path} \n")


if __name__ == "__main__":
    process_amazon_reviews()
