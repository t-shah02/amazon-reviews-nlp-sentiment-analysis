from re import A
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from read_data import (
    get_output_amazon_data,
    get_validation_data_from_split,
    get_all_model_validation_data,
)
from read_ml_models import load_all_models
from typing import Dict, List, Tuple, TypedDict
import pandas as pd


class ModelLabels(TypedDict):
    X_labels: str
    y_label: str


class ModelMetadata(TypedDict):
    model_name: str
    model: Pipeline
    labels: ModelLabels
    original_validation_data: Tuple[np.ndarray, np.ndarray]
    existing_validation_data: Tuple[pd.Series, pd.Series]


MODEL_LABELS: Dict[str, ModelLabels] = {
    "star_rating_linregress": {
        "X_labels": "normalized_star_rating",
        "y_label": "review_body_compound_score",
    },
    "sentiment_label_bayes": {
        "X_labels": "review_body",
        "y_label": "review_body_sentiment_label",
    },
    "product_category_bayes": {
        "X_labels": "review_body",
        "y_label": "product_category",
    },
}


def join_models_and_testing_data(
    models: Dict[str, Pipeline],
    all_validation_data: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
) -> List[ModelMetadata]:
    joined_data: List[ModelMetadata] = []

    for model_name in models:
        model = models[model_name]
        labels = MODEL_LABELS[model_name]
        fake_validation_df = all_validation_data[model_name]

        X_existing = df[labels["X_labels"]]
        y_existing = df[labels["y_label"]]
        X_valid_fake = fake_validation_df[labels["X_labels"]]
        y_valid_fake = (
            fake_validation_df[labels["y_label"]] if "bayes" in model_name else None
        )

        X_valid_existing, y_valid_existing = get_validation_data_from_split(
            X_existing, y_existing
        )

        if "bayes" in model_name:
            X_valid_existing = X_valid_existing.to_list()
            X_valid_fake = X_valid_fake.to_list()
        else:
            X_valid_existing = X_valid_existing.to_numpy().reshape(-1, 1)

        joined_data.append(
            {
                "model_name": model_name,
                "model": model,
                "labels": labels,
                "original_validation_data": (X_valid_fake, y_valid_fake),
                "existing_validation_data": (X_valid_existing, y_valid_existing),
            }
        )

    return joined_data


def showcase_ml_models():
    print("Loading the Amazon reviews dataset from disk...")
    amazon_data = get_output_amazon_data()
    amazon_data["normalized_star_rating"] = (
        amazon_data["star_rating"] * amazon_data["review_body_compound_score"]
    )
    print("Finished loading the Amazon reviews dataset\n")

    print("Loading machine learning models from disk...")
    models = load_all_models()
    print("Finished loading machine learning models from disk\n")

    print("Loading fake validation datasets from disk...")
    all_validation_data = get_all_model_validation_data()
    print("Finished loading fake validation datasets from disk\n")

    model_metadatas = join_models_and_testing_data(
        models, all_validation_data, amazon_data
    )

    for model_metadata in model_metadatas:
        model_name = model_metadata["model_name"]
        model = model_metadata["model"]
        X_valid, y_valid = model_metadata["existing_validation_data"]
        X_fake_valid, y_fake_valid = model_metadata["original_validation_data"]

        raw_score = model.score(X_valid, y_valid)
        print(f"Raw score for {model_name}: {raw_score}")

        if "bayes" in model_name:
            y_predictions = model.predict(X_fake_valid)
            report = classification_report(y_fake_valid, y_predictions, zero_division=1)
            print(f"{report}\n")


if __name__ == "__main__":
    showcase_ml_models()
