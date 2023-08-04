#!/bin/bash

# Create a directory for the output datasets
echo "[WORKING] Creating the directory (./output-amazon) for storing all the data..."
mkdir -p output-amazon

# Download output datasets and place them inside of output-amazon folder
urls=(
    "https://archive.org/download/amazon_reviews_dump/apparel.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/automative.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/electronics.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/furniture.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/grocery.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/personal_care_applications.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/shoes.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/software.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/sports.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/tools.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/video_games.csv.gz"
    "https://archive.org/download/amazon_reviews_dump/watches.csv.gz"
)

echo "[WORKING] Downloading output datasets from archive.org..."
for url in "${urls[@]}"
do
    curl -L -o "output-amazon/$(basename "${url}")" "${url}"
done

# Create a directory for the models
echo "[WORKING] Creating the directory (./models) for storing all the mdoels..."
mkdir -p models

# Download output datasets and place them inside of output-amazon folder
urls=(
    "https://archive.org/download/amazon_reviews_dump/product_category_bayes.joblib.gz"
    "https://archive.org/download/amazon_reviews_dump/sentiment_label_bayes.joblib.gz"
    "https://archive.org/download/amazon_reviews_dump/star_rating_linregress.joblib.gz"
)

echo "[WORKING] Downloading models from archive.org..."
for url in "${urls[@]}"
do
    curl -L -o "models/$(basename "${url}")" "${url}"
done
