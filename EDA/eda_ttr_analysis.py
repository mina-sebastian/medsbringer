import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import pos_tag

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize lemmatizers for additional languages
spacy_models = {
    'fr': 'fr_core_news_sm',
    'ro': 'ro_core_news_sm',
    'da': 'da_core_news_sm',
    'de': 'de_core_news_sm',
    'nl': 'nl_core_news_sm'
}

# Load spaCy models
spacy_lemmatizers = {lang: spacy.load(model) for lang, model in spacy_models.items()}

# Language-specific resources
LANGUAGE_RESOURCES = {
    'en': {
        'stopwords': set(stopwords.words('english')),
        'lemmatizer': WordNetLemmatizer()
    },
    'fr': {
        'stopwords': set(stopwords.words('french')),
        'lemmatizer': spacy_lemmatizers.get('fr')
    },
    'ro': {
        'stopwords': set(stopwords.words('romanian')),
        'lemmatizer': spacy_lemmatizers.get('ro')
    },
    'da': {
        'stopwords': set(stopwords.words('danish')),
        'lemmatizer': spacy_lemmatizers.get('da')
    },
    'de': {
        'stopwords': set(stopwords.words('german')),
        'lemmatizer': spacy_lemmatizers.get('de')
    },
    'nl': {
        'stopwords': set(stopwords.words('dutch')),
        'lemmatizer': spacy_lemmatizers.get('nl')
    }
}


def preprocess_text(text, language='en'):
    """
    Clean and preprocess the text based on language with enhanced filtering.
    """
    resources = LANGUAGE_RESOURCES.get(language)
    if not resources:
        raise ValueError(f"Unsupported language: {language}")

    # Convert to lowercase
    text = text.lower()

    # Remove numbers, special characters, and extra whitespace
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Create comprehensive stopwords
    extended_stopwords = resources['stopwords'].union({
        'may', 'must', 'will', 'can', 'could', 'would', 'should',
        'mg', 'ml', 'tel', 'tell', 'ask',
        'doctor', 'medicine', 'using', 'use', 'take', 'taking',
        'get', 'need', 'call', 'see', 'help', 'stop', 'keep',
        'know', 'used', 'using', 'include', 'including',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'be', 'is', 'are', 'was', 'were', 'been', 'being',
        'he', 'she', 'it', 'they', 'we', 'you', 'me', 'him', 'her',
        'do', 'does', 'did', 'doing', 'done',
        'have', 'has', 'had', 'having'
    })

    # Tokenize and apply enhanced filtering
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if (len(word) > 2 and  # Remove short words
            word.isalpha() and  # Keep only alphabetic words
            word not in extended_stopwords and  # Remove stopwords
            zipf_frequency(word, language) < 6.0)  # Filter out extremely common words
    ]

    # Lemmatize
    lemmatizer = resources['lemmatizer']
    if lemmatizer:
        if isinstance(lemmatizer, WordNetLemmatizer):
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        else:
            doc = lemmatizer(" ".join(tokens))
            tokens = [token.lemma_ for token in doc if len(token.lemma_) > 2]

    return tokens

def calculate_ttr(tokens):
    """
    Calculate the Type-Token Ratio (TTR).
    """
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0


def chunk_and_calculate_ttr(tokens, chunk_size=1000, shuffle_sentences=False):
    """
    Split tokens into chunks of size `chunk_size` and calculate TTR for each chunk.
    Optionally shuffle sentences before chunking.
    """
    if shuffle_sentences:
        sentences = sent_tokenize(" ".join(tokens))
        random.shuffle(sentences)
        tokens = word_tokenize(" ".join(sentences))

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    ttr_values = [calculate_ttr(chunk) for chunk in chunks]
    return ttr_values


def calculate_word_frequencies(tokens, language='en'):
    """
    Calculate word frequencies using language-specific POS tagging.
    """
    if language == 'en':
        # Use NLTK for English
        tagged_words = pos_tag(tokens)
        content_words = [word for word, tag in tagged_words
                         if tag.startswith(('NN', 'VB', 'JJ', 'RB'))]
    else:
        # Use spaCy for other languages
        nlp = spacy_lemmatizers[language]
        # Process in chunks of 100k characters
        content_words = []
        chunk_size = 100000
        text = " ".join(tokens)

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = nlp(chunk)
            chunk_words = [token.text for token in doc
                           if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
            content_words.extend(chunk_words)

    word_freqs = {
        word: zipf_frequency(word, language)
        for word in set(content_words)
        if len(word) > 2
    }

    return dict(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True))

def calculate_noun_frequencies(tokens, language='en'):
    """
    Calculate word frequencies using language-specific POS tagging for nouns only.
    """
    if language == 'en':
        # Use NLTK for English - only nouns (NN*)
        tagged_words = pos_tag(tokens)
        # Print some tagged words for verification
        print("Sample tagged words:", tagged_words[:10])
        content_words = [word for word, tag in tagged_words
                         if tag.startswith('NN')]
        # Print some identified nouns
        print("Sample nouns:", content_words[:10])
    else:
        # Use spaCy for other languages
        nlp = spacy_lemmatizers[language]
        content_words = []
        chunk_size = 100000
        text = " ".join(tokens)

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = nlp(chunk)
            chunk_words = [token.text for token in doc
                           if token.pos_ == 'NOUN']
            content_words.extend(chunk_words)
            # Print some identified nouns from this chunk
            print(f"Sample nouns from chunk: {chunk_words[:10]}")

    word_freqs = {
        word: zipf_frequency(word, language)
        for word in set(content_words)
        if len(word) > 2
    }

    return dict(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True))

def preprocess_text_temp(text, language='en'):
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
    lemmatizer = resources['lemmatizer']
    if lemmatizer:
        if isinstance(lemmatizer, WordNetLemmatizer):
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        else:
            doc = lemmatizer(" ".join(tokens))
            tokens = [token.lemma_ for token in doc if len(token.lemma_) > 2]
    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text

def clean_csv_data(input_csv_path, language='ro'):
    """
    Load, clean, and return the CSV data with language-specific processing.
    """
    try:
        data = pd.read_csv(input_csv_path)

        if 'id' not in data.columns or 'extracted_text' not in data.columns:
            raise ValueError("CSV must contain 'id' and 'extracted_text' columns.")

        data['extracted_text'] = data['extracted_text'].apply(lambda x: preprocess_text_temp(x, language))
        return data

    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None

def plot_ttr_distributions(ttr_data, languages):
    """
    Plot TTR distributions with enhanced statistical indicators.
    """
    # Create DataFrame
    ttr_df = pd.DataFrame({'TTR': sum(ttr_data.values(), []),
                           'Language': sum([[lang] * len(ttrs) for lang, ttrs in ttr_data.items()], [])})

    # Calculate statistics
    medians = ttr_df.groupby('Language')['TTR'].median()
    means = ttr_df.groupby('Language')['TTR'].mean()

    # Create enhanced plot
    plt.figure(figsize=(12, 8))

    # Main violin plot
    sns.violinplot(x='Language', y='TTR', data=ttr_df, palette='muted')

    # Add mean markers
    plt.plot(range(len(languages)), means.values, 'ro', label='Mean')

    # Add median lines
    plt.hlines(y=medians.values, xmin=range(len(languages)),
               xmax=[x + 0.5 for x in range(len(languages))],
               color='green', label='Median', linestyles='dashed')

    # Add threshold line at global mean
    plt.axhline(y=ttr_df['TTR'].mean(), color='blue',
                linestyle='--', label='Global Mean')

    # Enhance labels and styling
    plt.title('Type-Token Ratio (TTR) Distributions by Language',
              fontsize=14, pad=20)
    plt.ylabel('Type-Token Ratio (TTR)', fontsize=12)
    plt.xlabel('Language', fontsize=12)

    # Add statistical annotations
    for i, lang in enumerate(languages):
        plt.text(i, means[lang], f'μ={means[lang]:.3f}',
                 horizontalalignment='center', verticalalignment='bottom')

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def analyze_csv_data(input_csv_path, language='en'):
    """
    Analyze the CSV data for word frequencies and TTR distributions.
    """
    data = pd.read_csv(input_csv_path)

    if 'extracted_text' not in data.columns:
        raise ValueError("CSV must contain 'extracted_text' column.")

    tokens_list = []
    for text in data['extracted_text']:
        tokens = preprocess_text(text, language)
        tokens_list.extend(tokens)

    # Calculate metrics
    ttr = calculate_ttr(tokens_list)
    ttr_chunks = chunk_and_calculate_ttr(tokens_list, chunk_size=1000)
    word_freqs = calculate_word_frequencies(tokens_list, language)

    print(f"TTR (overall): {ttr}")
    print(f"TTR (per 1000 tokens chunk): {ttr_chunks[:5]}")  # Show first 5 chunk TTR values
    print(f"Word Frequencies (top 10): {dict(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:10])}")

    return ttr_chunks


if __name__ == "__main__":
    languages = ['en', 'fr', 'ro', 'nl', 'da', 'de']
    ttr_data = {}

    for lang in languages:
        input_csv_path = f'../data/raw_data_csv/{lang}_meds.csv'
        print(f"\nProcessing language: {lang}")
        ttr_chunks = analyze_csv_data(input_csv_path, lang)
        ttr_data[lang] = ttr_chunks

    # Plot TTR distributions across languages
    plot_ttr_distributions(ttr_data, languages)
