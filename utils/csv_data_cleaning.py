import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Language-specific resources
LANGUAGE_RESOURCES = {
    'en': {
        'stopwords': set(stopwords.words('english')),
        'lemmatizer': WordNetLemmatizer()
    },
    'fr': {
        'stopwords': set(stopwords.words('french')),
        'lemmatizer': None  # French needs different lemmatization
    },
    'ro': {
        'stopwords': set(stopwords.words('romanian')),
        'lemmatizer': None  # Romanian needs different lemmatization
    },
    'da': {
        'stopwords': set(stopwords.words('danish')),
        'lemmatizer': None
    },
    'de': {
        'stopwords': set(stopwords.words('german')),
        'lemmatizer': None
    },
    'nl': {
        'stopwords': set(stopwords.words('dutch')),
        'lemmatizer': None
    }
}


# def preprocess_text(text, language='en'):
#     """
#     Clean and preprocess the text based on language.
#     """
#     resources = LANGUAGE_RESOURCES.get(language)
#     if not resources:
#         raise ValueError(f"Unsupported language: {language}")
#
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove punctuation
#     text = re.sub(r'[^\w\s.!?]', '', text)
#
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     # Remove stopwords if available for the language
#     if resources['stopwords']:
#         text = ' '.join([word for word in text.split() if word not in resources['stopwords']])
#
#     # Lemmatize if available for the language
#     if resources['lemmatizer']:
#         text = ' '.join([resources['lemmatizer'].lemmatize(word) for word in text.split()])
#
#     return text
def preprocess_text(text, language='en'):
    """
    Clean and preprocess the text based on language.
    """
    resources = LANGUAGE_RESOURCES.get(language)
    if not resources:
        raise ValueError(f"Unsupported language: {language}")

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text using nltk
    tokens = word_tokenize(text)

    # Remove punctuation and unwanted characters
    tokens = [word for word in tokens if word.isalnum() or word in ['.', '!', '?']]

    # Remove stopwords if available for the language
    if resources['stopwords']:
        tokens = [word for word in tokens if word not in resources['stopwords']]

    # Lemmatize if available for the language
    if resources['lemmatizer']:
        tokens = [resources['lemmatizer'].lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text

def clean_csv_data(input_csv_path, language='en'):
    """
    Load, clean, and return the CSV data with language-specific processing.
    """
    try:
        data = pd.read_csv(input_csv_path)

        if 'id' not in data.columns or 'extracted_text' not in data.columns:
            raise ValueError("CSV must contain 'id' and 'extracted_text' columns.")

        data['extracted_text'] = data['extracted_text'].apply(lambda x: preprocess_text(x, language))
        return data

    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None


if __name__ == "__main__":
    languages = ['en', 'fr', 'ro']
    for lang in languages:
        input_csv_path = f'../data/raw_data_csv/{lang}_meds.csv'
        cleaned_data = clean_csv_data(input_csv_path, lang)
        if cleaned_data is not None:
            print(f"\nCleaned Data for {lang} (first few rows):")
            print(cleaned_data.head())
