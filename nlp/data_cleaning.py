import re
import pandas as pd
import html
from spacy.language import Language
from spacy.tokens import Doc
from typing import List, TypeVar, Callable
from tqdm import tqdm


def lower_text(text: str) -> str:
    return text.lower()


def strip_text(text: str) -> str:
    return text.strip()


def unescape_html(text: str) -> str:
    return html.unescape(text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def remove_special_characters(text: str) -> str:
    return re.sub("[^a-zA-Z0-9\s]", "", text)


def remove_html_tags(text: str) -> str:
    return re.sub("<.*?>", "", text)


PRE_NLP_STRING_PREPROCESS_FUNCTION = TypeVar(
    "PRE_NLP_STRING_PREPROCESS_FUNCTION", bound="Callable[[str], str]"
)

DEFAULT_PRE_NLP_STRING_PREPROCESS_FUNCTION: List[PRE_NLP_STRING_PREPROCESS_FUNCTION] = [
    lower_text,
    strip_text,
    remove_special_characters,
    remove_punctuation,
    unescape_html,
    remove_html_tags,
]


def remove_stopwords(nlp: Language, text: str) -> str:
    doc = nlp(text)
    tokens = []

    for token in doc:
        if not token.is_stop:
            tokens.append(token.text)

    return " ".join(tokens)


def lemmatize(nlp: Language, text: str) -> str:
    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token.lemma_)

    return " ".join(tokens)


def lemmatize_and_remove_stopwords(doc: Doc) -> str:
    tokens = []

    for token in doc:
        if not token.is_stop:
            tokens.append(token.lemma_)

    return " ".join(tokens)


def run_batch_text_normalization_pipeline(
    nlp: Language,
    texts: "pd.Series[str]",
    pre_nlp_string_functions: List[
        PRE_NLP_STRING_PREPROCESS_FUNCTION
    ] = DEFAULT_PRE_NLP_STRING_PREPROCESS_FUNCTION,
    show_batch_progress: bool = False,
    texts_type: str = "series",
) -> "pd.Series[str]":
    def apply_prenlp_processing_steps(text: str) -> str:
        for pre_nlp_string_function in pre_nlp_string_functions:
            text = pre_nlp_string_function(text)

        return text

    prenlp_texts = texts.apply(apply_prenlp_processing_steps)
    docs = nlp.pipe(texts=prenlp_texts, n_process=-1)
    final_docs = (
        docs if not show_batch_progress else tqdm(docs, total=prenlp_texts.shape[0])
    )
    texts: List[str] = []

    for doc in final_docs:
        texts.append(lemmatize_and_remove_stopwords(doc))

    if texts_type == "series":
        return pd.Series(texts)

    return texts
