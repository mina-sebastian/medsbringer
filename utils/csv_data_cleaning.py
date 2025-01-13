import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess_text(text):
    """
    Clean and preprocess the text.
    Steps:
    1. Convert to lowercase.
    2. Remove punctuation.
    3. Remove extra whitespace.
    4. Remove stopwords.
    5. Lemmatize words.
    """
    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


def clean_csv_data(input_csv_path):
    """
    Load, clean, and return the CSV data with `id` and `extracted_text` columns.

    Args:
    - input_csv_path (str): Path to the input CSV file.

    Returns:
    - pd.DataFrame: DataFrame with cleaned `extracted_text`.
    """
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(input_csv_path)

        # Check for required columns
        if 'id' not in data.columns or 'extracted_text' not in data.columns:
            raise ValueError("CSV must contain 'id' and 'extracted_text' columns.")

        # Clean the `extracted_text` column
        data['extracted_text'] = data['extracted_text'].apply(preprocess_text)

        return data
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None


if __name__ == "__main__":
    # Input CSV file path
    input_csv_path = '../data/raw_data_csv/en_meds.csv'

    # Clean and process the data
    cleaned_data = clean_csv_data(input_csv_path)

    # Print the cleaned data
    if cleaned_data is not None:
        print("Cleaned Data (first few rows):")
        print(cleaned_data.head())
