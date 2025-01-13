import re
import nltk  # Ensure nltk is imported
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


""" Bronze data cleaning """
input_file = "../data/raw_data_txt/en_meds.txt"
output_file = "../data/clean_data_txt/en_meds.txt"

def clean_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        # Read the file content
        text = infile.read()
    #remove leading spaces, lines or tabs from each line
    cleaned_text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(cleaned_text)

    print(f"Text cleaned and saved to {output_file}")


""" Silver data cleaning"""

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess the text."""
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

def clean_silver_data(file_path):
    """Load, clean, and return the silver data - only for english"""
    #read text data
    with open(file_path, 'r', encoding='utf-8') as file:
        leaflets_txt = file.read()

    #create a df and preprocess
    data = pd.DataFrame([leaflets_txt], columns=['EN'])
    data['EN'] = data['EN'].apply(preprocess_text)
    return data


if __name__ == "__main__":
    file_path = '../data/clean_data_txt/en_meds.txt'
    cleaned_data = clean_silver_data(file_path)
    print("Silver data:")
    print(cleaned_data)