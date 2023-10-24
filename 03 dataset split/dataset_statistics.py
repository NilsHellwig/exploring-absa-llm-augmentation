import nltk
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import pandas as pd
import json

nltk.download('punkt')

ASPECT_CATEGORIES = ["FOOD", "SERVICE",
                     "PRICE", "AMBIENCE", "GENERAL-IMPRESSION"]

POLARITIES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


def df_statistics(dataset_json):
    statistics = {}

    # 5. Count Most Frequent Aspect Phrases in Text for each Category
    for ac in ASPECT_CATEGORIES:
        statistics[f"most_frequent_aspect_terms_{ac}"] = Counter(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac).most_common(10)
        statistics[f"most_frequent_tokens_in_text_{ac}"] = Counter(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac).most_common(10)
        statistics[f"most_frequent_aspect_terms_{ac}"] = Counter(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac).most_common(10)
        statistics[f"unique_aspect_terms_{ac}"] = len(set(
            tag['text'] for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac))
        statistics[f"number_of_implicit_aspects_{ac}"] = len([tag for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-implicit' and tag["label"] == ac]
                                                             )
        statistics[f"number_of_explicit_aspects_{ac}"] = len([tag for entry in dataset_json for tag in entry['tags'] if tag['type'] == 'label-explicit' and tag["label"] == ac]
                                                             )
        statistics[f"number_of_aspects_{ac}"] = len([tag for entry in dataset_json for tag in entry['tags'] if tag["label"] == ac])

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

    # 6. Count AVG Number of Explicit Aspect Phrases in an Example
    statistics["avg_number_of_explicit_aspects"] = statistics["number_of_explicit_aspects"] / \
        len(dataset_json)

    # 7. Count AVG Number of Implicit Aspects in an Example
    statistics["avg_number_of_implicit_aspects"] = statistics["number_of_implicit_aspects"] / \
        len(dataset_json)

    # 8. Count AVG Number of All Aspects in an Example
    statistics["avg_number_of_aspects"] = statistics["number_of_aspects"] / \
        len(dataset_json)

    # 9. Create Statistics on how often 1,2,3 etc. number of aspects in an example
    statistics["statistics_frequency_of_n_aspects_for_an_example"] = dict(Counter(
        [len(entry["tags"]) for entry in dataset_json]).items())

    # 10. Most Frequent Aspect Categories
    most_frequent_ac = Counter(tag['label']
                               for entry in dataset for tag in entry['tags'])
    for key in most_frequent_ac.keys():
        statistics[f"ac_count_{key}"] = most_frequent_ac[key]

    # 11. Most Frequent Polarities
    most_frequent_pol = Counter(tag['polarity']
                                for entry in dataset for tag in entry['tags'])
    for key in most_frequent_pol.keys():
        statistics[f"polarity_count_{key}"] = most_frequent_pol[key]

    # 12. Combination Polarity and Aspect Category
    aspect_category_and_polarity = Counter(
        tag['tag_with_polarity'] for entry in dataset for tag in entry['tags'])

    for ac in ASPECT_CATEGORIES:
        for polarity in POLARITIES:
            statistics[f"{ac}-{polarity}-frequency"] = aspect_category_and_polarity[f'{ac}-{polarity}']

    # 13. Count Average Number of Words in Text
    word_counts = [len(word_tokenize(entry['text'], language='german'))
                   for entry in dataset]
    average_word_count = np.mean(word_counts)
    std_word_count = np.std(word_counts)
    max_word_count = np.max(word_counts)
    min_word_count = np.min(word_counts)

    statistics[f"avg_words_in_text"] = average_word_count
    statistics[f"std_words_in_text"] = std_word_count
    statistics[f"max_words_in_text"] = max_word_count
    statistics[f"min_words_in_text"] = min_word_count

    return statistics


# Load Data

DATASET_PATH = "dataset_total/filtered_dataset.json"

with open(DATASET_PATH, 'r', encoding='utf-8') as json_file:
    dataset = json.load(json_file)


print(df_statistics(dataset))
