from spacy.lang.de.stop_words import STOP_WORDS
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
import string
import spacy
import nltk
import json

nltk.download('punkt')
nlp = spacy.load("de_core_news_sm")

ASPECT_CATEGORIES = ["FOOD", "SERVICE",
                     "PRICE", "AMBIENCE", "GENERAL-IMPRESSION"]

POLARITIES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


def count_tokens(texts):
    token_counts = []
    for text in texts:
        tokens = word_tokenize(text)
        token_counts.append(len(tokens))
    return token_counts


def count_unique_lemmas(texts):
    unique_lemmas = set()
    for text in texts:
        doc = nlp(text)
        for token in doc:
            unique_lemmas.add(token.lemma_)
    return len(unique_lemmas)


def remove_stopwords_and_punctuation(text):
    doc = nlp(text)
    cleaned_tokens = [token.lemma_ for token in doc if token.text.lower(
    ) not in STOP_WORDS and token.text not in string.punctuation]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text


def count_top_n_lemmas(texts, n):
    lemma_counts = {}
    for text in texts:
        cleaned_text = remove_stopwords_and_punctuation(text)
        doc = nlp(cleaned_text)
        for token in doc:
            lemma = token.lemma_
            if lemma in lemma_counts:
                lemma_counts[lemma] += 1
            else:
                lemma_counts[lemma] = 1

    sorted_lemmas = sorted(
        lemma_counts, key=lambda lemma: lemma_counts[lemma], reverse=True)
    top_n_lemmas = sorted_lemmas[:n]

    return ', '.join(top_n_lemmas)


def df_statistics(dataset_json, synthetic=False):
    statistics = {}

    if synthetic == True:
        statistics = get_statistics_for_synthetic_data(
            statistics, dataset_json)

    # Count Most Frequent Aspect Phrases in Text for each Category
    for ac in ASPECT_CATEGORIES:
        statistics[f"most_frequent_aspect_terms_{ac}"] = Counter(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac).most_common(10)
        statistics[f"unique_aspect_terms_{ac}"] = len(set(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac))
        statistics[f"number_of_implicit_aspects_{ac}"] = len([tag for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-implicit' and tag["label"] == ac]
                                                             )
        statistics[f"number_of_explicit_aspects_{ac}"] = len([tag for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac]
                                                             )
        statistics[f"number_of_aspects_{ac}"] = len(
            [tag for entry in dataset_json for tag in entry['tags'] if tag["label"] == ac])

    # 1. Count Most Frequent Words in Text
    statistics["most_frequent_tokens_in_text"] = Counter(
        tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit').most_common(10)

    # 2. Count Most Frequent Aspect Phrases in Text
    statistics["most_frequent_aspect_terms"] = Counter(
        tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit').most_common(10)

    # 3. Count Number of Unique Aspect Phrases
    statistics["unique_aspect_terms"] = len(set(
        tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit'))

    # 4. Count Number of Implicit Aspects
    statistics["number_of_implicit_aspects"] = len(
        [tag for entry in dataset_json for tag in entry['tags']
            if tag['type'] == 'label-implicit']
    )

    # 5. Count Number of Explicit Aspects
    statistics["number_of_explicit_aspects"] = len(
        [tag for entry in dataset_json for tag in entry['tags']
            if tag['type'] == 'label-explicit']
    )

    # 6. Count Number of all Aspects
    statistics["number_of_aspects"] = len(
        [tag for entry in dataset_json for tag in entry['tags']])

    # 7. Count AVG Number of Explicit Aspect Phrases in an Example
    statistics["avg_number_of_explicit_aspects"] = statistics["number_of_explicit_aspects"] / \
        len(dataset_json)

    # 8. Count AVG Number of Implicit Aspects in an Example
    statistics["avg_number_of_implicit_aspects"] = statistics["number_of_implicit_aspects"] / \
        len(dataset_json)

    # 9. Count AVG Number of All Aspects in an Example
    statistics["avg_number_of_aspects"] = statistics["number_of_aspects"] / \
        len(dataset_json)

    # 10. Create Statistics on how often 1,2,3 etc. number of aspects in an example
    statistics["statistics_frequency_of_n_aspects_for_an_example"] = dict(Counter(
        [len(entry["tags"]) for entry in dataset_json]).items())

    # 11. Most Frequent Aspect Categories
    most_frequent_ac = Counter(tag['label']
                               for entry in dataset_json for tag in entry['tags'])
    for key in most_frequent_ac.keys():
        statistics[f"ac_count_{key}"] = most_frequent_ac[key]

    # 12. Most Frequent Polarities
    most_frequent_pol = Counter(tag['polarity']
                                for entry in dataset_json for tag in entry['tags'])
    for key in most_frequent_pol.keys():
        statistics[f"polarity_count_{key}"] = most_frequent_pol[key]

    # 13. Combination Polarity and Aspect Category
    aspect_category_and_polarity = Counter(
        tag['tag_with_polarity'] for entry in dataset_json for tag in entry['tags'])

    # Calculate frequency of each combination of aspect category and polarity
    for ac in ASPECT_CATEGORIES:
        for polarity in POLARITIES:
            statistics[f"{ac}-{polarity}-frequency"] = aspect_category_and_polarity[f'{ac}-{polarity}']

    # Calculate frequency of each combination of aspect category, polarity and implicit vs explicit
    for ac in ASPECT_CATEGORIES:
        for polarity in POLARITIES:
            for mention_type in ["label-explicit", "label-implicit"]:
                statistics[f"{ac}-{polarity}-{mention_type}-frequency"] = len([tag for example in dataset_json for tag in example["tags"] if tag["polarity"] == polarity and tag["label"] == ac and tag["type"] == mention_type])

    # 14. Count Average Number of Words in Text
    word_counts = [len(word_tokenize(entry['text'], language='german'))
                   for entry in dataset_json]

    average_word_count = np.mean(word_counts)
    std_word_count = np.std(word_counts)
    max_word_count = np.max(word_counts)
    min_word_count = np.min(word_counts)

    statistics[f"avg_words_in_text"] = average_word_count
    statistics[f"std_words_in_text"] = std_word_count
    statistics[f"max_words_in_text"] = max_word_count
    statistics[f"min_words_in_text"] = min_word_count

    return statistics


def get_statistics_for_synthetic_data(statistics, dataset_json):
    # 1. Store reasons for retry errors
    for retry_reason in ["llm_invalid_xml_schema", "llm_invalid_xml_tags", "llm_aspect_polarity_in_text_but_not_in_label", "llm_more_than_one_sentences"]:
        statistics[f"avg_{retry_reason}"] = np.mean(
            [example[retry_reason] for example in dataset_json])
        statistics[f"sd_{retry_reason}"] = np.std(
            [example[retry_reason] for example in dataset_json])
        statistics[f"min_{retry_reason}"] = np.min(
            [example[retry_reason] for example in dataset_json])
        statistics[f"max_{retry_reason}"] = np.max(
            [example[retry_reason] for example in dataset_json])

    return statistics
