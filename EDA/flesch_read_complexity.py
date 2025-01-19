import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import textstat
import numpy as np


def calculate_readability_metrics(text):
    """Calculate readability metrics for text with difficulty score."""

    flesch = textstat.flesch_reading_ease(text)
    gunning = textstat.gunning_fog(text)
    avg_sent_len = textstat.avg_sentence_length(text)
    syllables_per_word = textstat.avg_syllables_per_word(text)

    #normalize metrics as per existing codebase
    normalized_metrics = {
        'flesch_reading_ease': flesch / 100,
        'gunning_fog': gunning / 20,
        'avg_sentence_length': avg_sent_len / 50,
        'avg_syllables_per_word': syllables_per_word / 3
    }
    #Calculate difficulty score using complete weights from codebase
    weights = {
        'flesch_reading_ease': -0.3,
        'gunning_fog': 0.2,
        'avg_sentence_length': 0.1,
        'avg_syllables_per_word': 0.2
    }
    difficulty_score = sum(normalized_metrics[metric] * weight
                           for metric, weight in weights.items())

    #Return all metrics including raw and difficulty score
    return {
        'flesch_reading_ease': flesch,
        'gunning_fog': gunning,
        'avg_sentence_length': avg_sent_len,
        'difficulty_score': difficulty_score
    }


def analyze_readability_by_language():
    languages = ['en', 'fr', 'ro', 'de', 'da', 'nl']
    metrics_by_lang = {}

    for lang in languages:
        file_path = f"./data/clean_data_txt/{lang}_meds.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            metrics = calculate_readability_metrics(text)
            metrics_by_lang[lang] = metrics

    return pd.DataFrame(metrics_by_lang).T


def plot_readability_metrics(metrics_df):
    metrics_df = metrics_df.sort_values('difficulty_score')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.2
    r1 = np.arange(len(metrics_df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Main metrics bars
    ax1.bar(r1, metrics_df['flesch_reading_ease'], width=bar_width, label='Flesch Reading Ease', color='skyblue')
    ax1.bar(r2, metrics_df['gunning_fog'], width=bar_width, label='Gunning Fog', color='lightgreen')
    ax1.bar(r3, metrics_df['avg_sentence_length'], width=bar_width, label='Avg Sentence Length', color='salmon')
    ax1.set_ylabel('Main Metrics Scores')

    #y-axis for difficulty score
    ax2 = ax1.twinx()

    difficulty_ranges = [
        (0.03, 'Easy', 'green'),
        (0.06, 'Moderate', 'orange'),
        (0.1, 'Hard', 'red')
    ]

    for level, label, color in difficulty_ranges:
        ax2.axhline(y=level, color=color, linestyle='--', alpha=0.8)
        ax2.text(-0.5, level, label, verticalalignment='bottom', color=color, fontweight='bold')

    ax2.plot(r2, metrics_df['difficulty_score'], 'o-', label='Difficulty Score', linewidth=2, color='purple')
    ax2.set_ylabel('Difficulty Score')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Readability Metrics by Language (Sorted by Difficulty)')
    plt.xlabel('Languages')
    plt.xticks([r + bar_width for r in range(len(metrics_df))], metrics_df.index)

    plt.tight_layout()
    plt.show()


def main():
    metrics_df = analyze_readability_by_language()
    plot_readability_metrics(metrics_df)
    print("Readability metrics by language:")
    print(metrics_df.round(2))


main()
