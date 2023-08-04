import os
from typing import Dict
from dotenv import load_dotenv
from joblib import dump, load
from sklearn.pipeline import Pipeline

load_dotenv()

MODEL_FOLDER = os.environ["ML_MODEL_FOLDER"]


def save_model(model: Pipeline, model_filename: str):
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    dump(model, f"{MODEL_FOLDER}{model_filename}")


def load_model(model_filename: str) -> Pipeline:
    model: Pipeline = load(f"{MODEL_FOLDER}{model_filename}")
    return model


def load_all_models() -> Dict[str, Pipeline]:
    model_filenames = os.listdir(MODEL_FOLDER)
    all_models: Dict[str, Pipeline] = {}

    for model_filename in model_filenames:
        loaded_model = load_model(model_filename)
        all_models[model_filename.replace(".joblib.gz", "")] = loaded_model

    return all_models
