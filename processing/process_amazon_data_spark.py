import sys

sys.path.append("..")

import pandas as pd
import os
import spacy
from spacy.language import Language
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from pyspark.sql import SparkSession, types
from pyspark.sql.functions import pandas_udf
from nlp.data_cleaning import run_batch_text_normalization_pipeline
from nlp.sentiments import get_sentiment_compound_score, get_sentiment_label
from typing import List

load_dotenv()
SPARK_SAMPLE_LIMIT = int(os.getenv("SPARK_SAMPLE_LIMIT", 100_000))
SPARK_SAMPLE_FRACTION = float(os.getenv("SPARK_SAMPLE_FRACTION", 0.7))
AMAZON_BIGDATA_INPUT_DIRECTORY = os.getenv(
    "AMAZON_BIGDATA_INPUT_DIRECTORY", "../input-amazon"
)
AMAZON_BIGDATA_OUTPUT_DIRECTORY = os.getenv(
    "AMAZON_BIGDATA_OUTPUT_DIRECTORY", "../output-amazon"
)

spark: SparkSession = SparkSession.builder.appName(
    "amazon reviews sentiments job"
).getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.memory.OffHeap.size", "4g")
spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.addPyFile("../package-zips/nlp.zip")

GLOBAL_NLP: Language | None = None
GLOBAL_VADER_ANALYZER: SentimentIntensityAnalyzer | None = None


def get_global_vader_analyzer() -> SentimentIntensityAnalyzer:
    global GLOBAL_VADER_ANALYZER

    if GLOBAL_VADER_ANALYZER is None:
        GLOBAL_VADER_ANALYZER = SentimentIntensityAnalyzer()

    return GLOBAL_VADER_ANALYZER


def get_global_nlp_model() -> Language:
    global GLOBAL_NLP

    if GLOBAL_NLP is None:
        GLOBAL_NLP = spacy.load("en_core_web_md", disable=["parser", "ner", "tok2vec"])

    return GLOBAL_NLP


@pandas_udf(returnType=types.StringType())
def normalize_reviews(review_bodies: pd.Series) -> pd.Series:
    nlp_model = get_global_nlp_model()
    normalized_reviews = run_batch_text_normalization_pipeline(nlp_model, review_bodies)

    return normalized_reviews


@pandas_udf(returnType=types.FloatType())
def get_compound_sentiment_scores(review_bodies: pd.Series) -> pd.Series:
    sentiment_analyzer = get_global_vader_analyzer()
    compound_scores = review_bodies.apply(
        lambda text: get_sentiment_compound_score(sentiment_analyzer, text)
    )
    return compound_scores


@pandas_udf(returnType=types.StringType())
def get_sentiment_labels(compound_scores: pd.Series) -> pd.Series:
    sentiment_labels = compound_scores.apply(get_sentiment_label)
    return sentiment_labels


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


def process_amazon_reviews():
    amazon_reviews = spark.read.csv(
        AMAZON_BIGDATA_INPUT_DIRECTORY, header=True, inferSchema=True
    ).dropna()
    amazon_reviews_dropped = amazon_reviews.drop(*COLUMNS_TO_DROP)

    normalized_amazon_reviews = amazon_reviews_dropped.withColumn(
        "normalized_review_body",
        normalize_reviews(amazon_reviews_dropped["review_body"]),
    )

    normalized_amazon_reviews_with_compound_score = (
        normalized_amazon_reviews.withColumn(
            "review_compound_score",
            get_compound_sentiment_scores(normalized_amazon_reviews["review_body"]),
        )
    )

    final_amazon_reviews = normalized_amazon_reviews_with_compound_score.withColumn(
        "review_sentiment_label",
        get_sentiment_labels(
            normalized_amazon_reviews_with_compound_score["review_compound_score"]
        ),
    )

    final_amazon_reviews.write.csv(AMAZON_BIGDATA_OUTPUT_DIRECTORY, mode="overwrite")


if __name__ == "__main__":
    process_amazon_reviews()
