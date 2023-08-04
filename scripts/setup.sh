#!/bin/bash

# Check if Python3 is installed as python3, python, or py
if command -v python3 &> /dev/null
then
    PYTHON_CMD=python3
elif command -v python &> /dev/null
then
    PYTHON_CMD=python
elif command -v py &> /dev/null
then
    PYTHON_CMD=py
else
    echo "Python is not installed. Please install Python first."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create a Python virtual environment and activate it
echo "[WORKING] Creating your python virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Install the required Python packages
echo "[WORKING] Installing required Python packages, this might take a few minutes..."
pip install -r requirements.txt

# Download the spaCy model
echo "[WORKING] Downloading spaCy model 'en_core_web_md'..."
$PYTHON_CMD -m spacy download en_core_web_md

# Create a directory for the data files
echo "[WORKING] Creating the directory (./input-amazon) for storing all the data..."
mkdir -p input-amazon

# List of data file URLs
urls=(
    "https://archive.org/download/amazon_reviews_dump/input-apparel.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-automotive.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-electronics.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-furniture.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-grocery.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-personal-care.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-shoes.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-software.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-sports.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-tools.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-video-games.tsv.gz"
    "https://archive.org/download/amazon_reviews_dump/input-watches.tsv.gz"
)

# Download each data file
echo "[WORKING] Downloading data files from archive.org..."
for url in "${urls[@]}"
do
    curl -L -o "input-amazon/$(basename "${url}")" "${url}"
done

# Set the environment variables (if the env variables have already been set then it won't override them)
echo "[WORKING] Setting your environment variables..."
export SPARK_SAMPLE_LIMIT="${SPARK_SAMPLE_LIMIT:=100_000}"
export PANDAS_SAMPLE_LIMIT="${PANDAS_SAMPLE_LIMIT:=95000}"
export SPARK_SAMPLE_FRACTION="${SPARK_SAMPLE_FRACTION:=0.7}"
export AMAZON_BIGDATA_INPUT_DIRECTORY="${AMAZON_BIGDATA_INPUT_DIRECTORY:=./input-amazon}"
export AMAZON_BIGDATA_OUTPUT_DIRECTORY="${AMAZON_BIGDATA_OUTPUT_DIRECTORY:=./output-amazon}"

# Ask the user which environment they want to run
read -p "Do you want to run the transformations in PySpark or Pandas? (pyspark/pandas): " ENV_CHOICE

if [ "$ENV_CHOICE" = "pyspark" ]; then
    # Prepare nlp package for PySpark
    echo "[WORKING] Preparing nlp package for PySpark..."
    mkdir -p package-zips/pyspark_package
    cp -r nlp/* package-zips/pyspark_package
    cd package-zips/pyspark_package
    zip -r ../nlp.zip *
    cd ../..
    rm -rf package-zips/pyspark_package

    # Run PySpark script
    echo "[WORKING] Running spark-submit to process the output..."
    spark-submit processing/process_amazon_data_spark.py
elif [ "$ENV_CHOICE" = "pandas" ]; then
    # Run Pandas script
    echo "[WORKING] Running Pandas script to process the output..."
    $PYTHON_CMD processing/process_amazon_reviews_pandas.py
else
    echo "Invalid choice. Please enter 'pyspark' or 'pandas'."
    exit 1
fi
