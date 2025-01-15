import pandas as pd
import numpy as np
import re
from textstat import textstat
from sklearn.preprocessing import MinMaxScaler
from utils.csv_data_cleaning import clean_csv_data
import os

def preprocess_for_readability(text):
    """Properly segment text into sentences for readability metrics."""
    # Add proper sentence boundaries
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    return text

def calculate_readability_metrics(text):
    """Calculate readability metrics with proper text preprocessing."""
    try:
        text = preprocess_for_readability(text)
        metrics = {
            'flesch_reading_ease': max(-100, min(100, textstat.flesch_reading_ease(text))),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'sentence_count': max(1, textstat.sentence_count(text)),
            'word_count': textstat.lexicon_count(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text)
        }
        return metrics
    except Exception as e:
        print(f"Error processing text: {text[:50]}... - {e}")
        # Return default values for missing metrics
        return {
            'flesch_reading_ease': None,
            'flesch_kincaid_grade': None,
            'gunning_fog': None,
            'sentence_count': None,
            'word_count': None,
            'avg_sentence_length': None,
            'avg_syllables_per_word': None
        }

def classify_difficulty(df, text_column='extracted_text'):
    """
    Classify text difficulty into levels:
    Easy, Moderate, Difficult.
    """
    # Calculate readability metrics for each text
    metrics = []
    for text in df[text_column]:
        metrics.append(calculate_readability_metrics(text))

    metrics_df = pd.DataFrame(metrics)

    # Normalize metrics where applicable
    metrics_df['flesch_reading_ease'] = metrics_df['flesch_reading_ease'] / 100
    metrics_df['flesch_kincaid_grade'] = metrics_df['flesch_kincaid_grade'] / 12
    metrics_df['gunning_fog'] = metrics_df['gunning_fog'] / 20
    metrics_df['avg_sentence_length'] = metrics_df['avg_sentence_length'] / 50
    metrics_df['avg_syllables_per_word'] = metrics_df['avg_syllables_per_word'] / 3

    # Fill missing values with 0 for normalized scores
    metrics_df = metrics_df.fillna(0)

    # Create composite score using weights
    weights = {
        'flesch_reading_ease': -0.3,  # Negative weight for easier texts
        'flesch_kincaid_grade': 0.2,
        'gunning_fog': 0.2,
        'avg_sentence_length': 0.1,
        'avg_syllables_per_word': 0.2
    }

    difficulty_scores = np.zeros(len(df))
    for metric, weight in weights.items():
        difficulty_scores += metrics_df[metric] * weight

    # Classify into three levels
    difficulty_levels = pd.qcut(difficulty_scores, q=3, labels=['Easy', 'Moderate', 'Difficult'])

    # Add results to dataframe
    result_df = df.copy()
    result_df['difficulty_score'] = difficulty_scores
    result_df['difficulty_level'] = difficulty_levels

    # Expand metrics into separate columns
    result_df = pd.concat([result_df, metrics_df], axis=1)

    return result_df

if __name__ == "__main__":
    languages = ['en', 'fr', 'ro']
    for lang in languages:
        print(f"\nAnalyzing {lang.upper()} language leaflets...")
        input_csv_path = f'../data/raw_data_csv/{lang}_meds.csv'

        # Check if file exists
        if not os.path.exists(input_csv_path):
            print(f"File not found: {input_csv_path}")
            continue

        # Load and clean data
        data = clean_csv_data(input_csv_path, lang)
        if data is not None:
            # Classify difficulty
            classified_data = classify_difficulty(data)

            # Print summary
            print("\nDifficulty Level Distribution:")
            print(classified_data['difficulty_level'].value_counts())

            # Example of detailed metrics for first document
            print("\nDetailed metrics for first document:")
            print(classified_data.iloc[0][['difficulty_level', 'difficulty_score', 'flesch_reading_ease',
                                           'flesch_kincaid_grade', 'gunning_fog', 'avg_sentence_length',
                                           'avg_syllables_per_word']])
