import os
import sys

sys.path.append("../..")

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


load_dotenv()
AMAZON_BIGDATA_OUTPUT_DIRECTORY = os.getenv(
    "AMAZON_BIGDATA_OUTPUT_DIRECTORY", "../output-amazon"
)
ML_MODEL_TESTING_FOLDER = os.getenv("ML_MODEL_TESTING_FOLDER", "/testing_data/")

OUTPUT_DATAFRAME_DTYPES = {
    "product_title": str,
    "product_category": str,
    "star_rating": np.float64,
    "helpful_votes": np.float64,
    "review_body": str,
    "normalized_review_body": str,
    "review_body_compound_score": np.float64,
    "review_body_sentiment_label": str,
}


def get_output_amazon_data() -> pd.DataFrame:
    output_dfs: List[pd.DataFrame] = []
    output_datafiles = os.listdir(AMAZON_BIGDATA_OUTPUT_DIRECTORY)

    for output_datafile in output_datafiles:
        relative_path = os.path.join(AMAZON_BIGDATA_OUTPUT_DIRECTORY, output_datafile)
        output_df = pd.read_csv(
            relative_path, dtype=OUTPUT_DATAFRAME_DTYPES, compression="gzip"
        )

        output_dfs.append(output_df)

    return pd.concat(output_dfs, axis=0).dropna()


def get_validation_data_from_split(
    X: pd.DataFrame | pd.Series, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_valid, y_valid


def get_all_model_validation_data() -> Dict[str, pd.DataFrame]:
    validation_data: Dict[str, pd.DataFrame] = {}

    validation_data_filenames = os.listdir(ML_MODEL_TESTING_FOLDER)

    for validation_date_filename in validation_data_filenames:
        relative_testing_data_path = os.path.join(
            ML_MODEL_TESTING_FOLDER, validation_date_filename
        )
        validation_df = pd.read_csv(relative_testing_data_path)
        validation_data[validation_date_filename.replace(".csv", "")] = validation_df

    return validation_data
