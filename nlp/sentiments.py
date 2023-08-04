# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 00:12:03 2023

@author: anmol
"""
#   Github link for vader sentiment
#   https://github.com/cjhutto/vaderSentiment
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Tuple
import pandas as pd


def get_sentiment_compound_score(
    analyzer: SentimentIntensityAnalyzer, text: str
) -> float:
    senti_scorer = analyzer.polarity_scores(
        text
    )  # get polarity scores based on the lexicon of the vader sentimental analysis
    return senti_scorer["compound"]  # returns compound score of the analysis


def get_sentiment_label(compound_score: float) -> str:
    # return one of the three labels based on the if statements

    #   Range of emotions are
    #   Positive >= 0.05
    #   -0.05 <= Neutral <= 0.05
    #   Negative <= -0.05

    if compound_score >= 0.05:
        return "positive"

    if compound_score <= -0.05:
        return "negative"

    return "neutral"


def sscorer(text: str) -> Tuple[float, str]:
    compound_score = get_sentiment_compound_score(text)
    sentiment_label = get_sentiment_label(compound_score)

    return compound_score, sentiment_label


def compute_sentiments_in_batch(
    analyzer: SentimentIntensityAnalyzer,
    texts: pd.Series,
    show_batch_progress: bool = False,
) -> Tuple[List[float], List[str]]:
    compound_scores: List[float] = []
    sentiment_labels: List[str] = []

    for text in tqdm(texts, total=texts.shape[0]) if show_batch_progress else texts:
        compound_score = get_sentiment_compound_score(analyzer, text)
        sentiment_label = get_sentiment_label(compound_score)

        compound_scores.append(compound_score)
        sentiment_labels.append(sentiment_label)

    return compound_scores, sentiment_labels
