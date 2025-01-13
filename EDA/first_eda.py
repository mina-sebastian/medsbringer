import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from scipy.stats import poisson
from utils.csv_data_cleaning import clean_csv_data
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import re

# Frequency analysis
def make_freq_analysis(df, text_column="extracted_text", title="Word Cloud", num_top_words=10):
    """generate and display a word cloud and top word freq from the specified text column in df"""
    #all text into a single string
    all_text = " ".join(df[text_column].dropna())

    #tokenize the text and count word freq
    words = all_text.split()
    word_counts = Counter(words)

    #display top frequent words
    print(f"Top {num_top_words} most frequent words:")
    for word, count in word_counts.most_common(num_top_words):
        print(f"{word}: {count}")

    #gen word cloud
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

    #freq distribution bar plot
    freq_data = word_counts.most_common(num_top_words)
    words, counts = zip(*freq_data)
    sns.barplot(x=list(counts), y=list(words))
    plt.title('Top Words Frequency')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

# Topic Modeling
def perform_topic_modeling(df, text_column="extracted_text", num_topics=5, max_features=1000):
    #all text into a single string
    all_text = " ".join(df[text_column].dropna())

    #vectorize the text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform([all_text])

    #LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    print("Topics:")
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx + 1}: ", [terms[i] for i in topic.argsort()[-10:]])

# Clustering
def perform_clustering(df, text_column="extracted_text", num_clusters=5, max_features=1000):
    """Perform clustering on the specified text column in a DataFrame."""
    #convert column to a list of texts
    texts = df[text_column].dropna().tolist()

    #Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(texts)

    #KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Add clusters in df
    df['cluster'] = pd.Series(clusters, index=df.index)
    return df


def visualize_clusters(df, cluster_column="cluster"):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[cluster_column])
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()

def identify_discriminatory_words(df, text_column="extracted_text", max_features=1000):
    """
    Identify words with significant deviation using a Poisson distribution
    from the specified text column, filtering out irrelevant tokens.
    """
    #all text into a single string
    all_text = " ".join(df[text_column].dropna())

    #avoid numbers and special characters, and convert to lowercase
    all_text = re.sub(r"[^a-zA-Z\s]", "", all_text.lower())

    #vectorize the text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform([all_text])
    word_counts = X.sum(axis=0).A1

    #poisson distribution
    poisson_fit = poisson(mu=np.mean(word_counts))
    terms = vectorizer.get_feature_names_out()

    #words with significant deviation,filter out irrelevant tokens
    discriminatory_words = [
        terms[i] for i in np.where(poisson_fit.pmf(word_counts) < 0.05)[0]
        if len(terms[i]) > 2  # Exclude short tokens (e.g., single letters)
    ]

    print("Discriminatory Words:")
    print(len(discriminatory_words), discriminatory_words)
    print("Total Words:", len(terms))
    print("Total Discriminatory Words:", len(discriminatory_words) / len(terms))
    print("Discriminatory Words Percentage:", round(len(discriminatory_words) / len(terms) * 100, 2), "%")
    return discriminatory_words


if __name__ == "__main__":
    file_path = "../data/raw_data_csv/en_meds.csv"
    # Load and clean data
    data = clean_csv_data(file_path)
    """
    print("Testing word cloud generation...")
    generate_wordcloud(data['EN'].iloc[0], title="Sample Word Cloud")
"""
    # Perform frequency analysis
    print("Generating word cloud...")
    make_freq_analysis(data)

    """
    # Perform topic modeling
    print("Performing topic modeling...")
    perform_topic_modeling(data)

    
    # Perform clustering
    clusters = perform_clustering(data)
    print("Cluster assignments:", clusters)
    
    # Identify discriminatory words
    print("Identifying discriminatory words...")
    identify_discriminatory_words(data)
"""
