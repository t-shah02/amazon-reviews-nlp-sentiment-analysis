# Sentiment Unleashed: Amazon Reviews NLP Analysis 🚀

Welcome to the exciting journey of unraveling sentiments hidden within Amazon reviews! This project leverages the power of natural language processing (NLP) to categorize sentiments as positive, negative, or neutral across various product categories.

## 🧰 What's Inside the Box?

### Data Processing 🔄

Utilize PySpark or Pandas to clean, transform, and prepare the Amazon reviews data. Explore the magic in:
- `processing/process_amazon_reviews_pandas.py`
- `processing/process_amazon_data_spark.py`

### Analysis 📊

Dive into detailed analysis with Jupyter notebooks:
- Exploratory Data Analysis (EDA)
- Predictions: Star Ratings, Product Categories, Sentiment Labels
- Statistical Significance Testing

### Natural Language Processing (NLP) 📝

Explore the NLP techniques used:
- Text Cleaning: Lowercasing, HTML unescaping, punctuation removal
- Sentiment Analysis: Vader Sentiment library

## 🚀 Get Started

### Recommended: Download Preprocessed Data and Models 📥

Before diving into the code (especially the notebooks), we strongly recommend downloading the preprocessed output dataset directory and models hosted on archive.org. This step will save you time by avoiding the need to rerun the entire data processing phase.

- **Output Datasets**: Available in `.csv.gz` format: [Download Output Datasets](https://archive.org/download/amazon_reviews_dump)
- **Models**: Available in `.joblib.gz` format: [Download Models](https://archive.org/download/amazon_reviews_dump)

The processing package, although insightful, is mainly included to showcase our data cleaning and preparation process.

### Requirements 🛠️

Just make sure you have Python 3.11 installed, and we'll take care of the rest!

### Setting Script Permissions 🔑

Before proceeding with either the automatic setup or manual exploration, please ensure that the scripts have the necessary permissions to execute. This can be done by navigating to the root directory of the project and running the following commands:

```bash
chmod +x ./scripts/setup.sh
chmod +x ./scripts/download-data.sh
```

Make sure you are in the root directory of the project when running these commands. These will grant execute permissions for the `setup.sh` and `download-data.sh` scripts, enabling them to run on your system.

Now you're ready to continue with the setup process, as outlined in the sections below!

### Automatic Setup 🎩✨

1. Clone the repository.
2. Navigate to the root directory of the project.
3. Run the magical setup script:

```bash
./scripts/setup.sh
```

### Manual Exploration 🧐

If you prefer to explore manually, you'll need to set some environment variables. Here's the default `.env` skeleton:

```env
SPARK_SAMPLE_LIMIT=50000
PANDAS_SAMPLE_LIMIT=95000
SPARK_SAMPLE_FRACTION=0.90
AMAZON_BIGDATA_INPUT_DIRECTORY=./input-amazon/
AMAZON_BIGDATA_OUTPUT_DIRECTORY=./output-amazon/
ML_MODEL_FOLDER=./models/
ML_MODEL_TESTING_FOLDER=validation_data/
```

1. **Download Data**: Run the `download-data.sh` script from the root directory to download Amazon reviews data and pre-trained models:

```bash
./scripts/download-data.sh
```

2. **Create Virtual Environment**:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download spaCy Model**:

```bash
python3 -m spacy download en_core_web_md
```

5. **Process Data**: Choose either Pandas or PySpark:

```bash
python3 processing/process_amazon_reviews_pandas.py # For Pandas
```
or
```bash
spark-submit processing/process_amazon_data_spark.py # For PySpark
```

6. **Explore Analysis Notebooks**: Navigate to the `analysis` directory to explore Jupyter notebooks.

### Model Evaluation 🧪

Explore the performance of the Bayes models through classification reports and scoring with the `analysis/run_models.py` script. This script provides insights into how well the models are performing on the validation data.

To run the script, navigate to the `analysis` directory and execute:

```bash
python3 run_models.py
```

This will generate classification reports and scores for the Bayes models, displaying them in the standard output. Make sure you have the required models and validation data available before running this script.

## ⚠️ Spark Warning

Running Spark jobs requires significant memory and may not be suitable for machines with limited resources. If you are an SFU student, faculty member, or staff, consider using CSIL if you wish to run Spark jobs. Otherwise, you can choose the Pandas option for data processing.
